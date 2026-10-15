"""LLM-клиент: OpenAI-совместимые chat completions с tool-calling.

Исторически модуль назывался deepseek.py; теперь это универсальный клиент для
любого вендора из реестра `config.LLM_PROVIDERS` (DeepSeek, MiniMax, Qwen,
OpenAI, OpenRouter, Ollama, custom). Провайдера/модель можно переключать
в рантайме через `switch()` (команда /model).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from ..config import LLM_PROVIDERS, LLMProvider, settings

log = logging.getLogger(__name__)
tokens_log = logging.getLogger("bot.tokens")


def extract_usage(resp: dict) -> tuple[int, int, int]:
    """Вернуть (prompt, completion, cached) из usage любого вендора.

    DeepSeek отдаёт `prompt_cache_hit_tokens`, OpenAI/Qwen —
    `prompt_tokens_details.cached_tokens`, остальные — ничего.
    """
    usage = resp.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(usage.get("prompt_cache_hit_tokens") or 0)
    if not cached:
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
    return prompt, completion, cached


def _completions_url(base_url: str) -> str:
    """Собрать URL chat/completions из базового.

    Если base_url — голый хост (https://api.deepseek.com, http://localhost:11434),
    добавляем /v1. Если путь уже есть (/v1, /compatible-mode/v1, /api/v1) —
    используем как есть.
    """
    base = base_url.rstrip("/")
    tail = base.split("://", 1)[-1]
    if "/" not in tail:
        base += "/v1"
    return f"{base}/chat/completions"


class LLMClient:
    def __init__(
        self,
        provider_key: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        self.provider: LLMProvider
        self.api_key: str = ""
        self.base_url: str = ""
        self.model: str = ""
        self._configure(provider_key, model, api_key=api_key, base_url=base_url)
        # connect — быстрый (ловим DNS/network), read — долгий
        # (модели иногда думают >2 минут на tool-calling).
        if timeout is None:
            tmo = httpx.Timeout(connect=15.0, read=1200.0, write=60.0, pool=10.0)
        else:
            tmo = httpx.Timeout(timeout)
        # Форсируем IPv4: на некоторых машинах дефолтный сокет висит ConnectTimeout
        # к api.deepseek.com (curl -4 ок, голый httpx — нет). Bind на 0.0.0.0 чинит.
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1)
        self._client = httpx.AsyncClient(timeout=tmo, transport=transport)

    def _configure(
        self, provider_key: str | None, model: str | None,
        *, api_key: str | None = None, base_url: str | None = None,
    ) -> None:
        provider, key, url, mdl = settings.resolve_llm(provider_key, model)
        self.provider = provider
        self.api_key = api_key or key
        self.base_url = (base_url or url).rstrip("/")
        self.model = mdl
        log.info("LLMClient: provider=%s model=%s base_url=%s key=%s",
                 provider.key, self.model, self.base_url,
                 "set" if self.api_key else "MISSING")

    # --- runtime switching ---

    @staticmethod
    def available_providers() -> list[dict[str, Any]]:
        """Список вендоров с признаком наличия ключа (для /model)."""
        out = []
        for p in LLM_PROVIDERS.values():
            out.append({
                "key": p.key,
                "title": p.title,
                "default_model": p.default_model,
                "models": list(p.models),
                "has_key": bool(settings.provider_api_key(p)),
            })
        return out

    def switch(self, provider_key: str, model: str | None = None) -> str:
        """Переключить вендора/модель. Возвращает человекочитаемое описание.

        Бросает ValueError для неизвестного провайдера или отсутствующего ключа.
        """
        pkey = provider_key.strip().lower()
        if pkey not in LLM_PROVIDERS:
            raise ValueError(
                f"неизвестный провайдер '{pkey}'. Доступны: {', '.join(LLM_PROVIDERS)}"
            )
        provider, key, url, mdl = settings.resolve_llm(pkey, model)
        if not key and pkey != "ollama":
            raise ValueError(
                f"нет API-ключа для {provider.title}: задай {provider.api_key_env} в .env"
            )
        if not url:
            raise ValueError(f"для провайдера {pkey} нужен LLM_BASE_URL")
        self._configure(pkey, model)
        return self.describe()

    def describe(self) -> str:
        return f"{self.provider.title} · {self.model} · {self.base_url}"

    def cost_usd(self, prompt: int, completion: int, cached: int = 0) -> float:
        """Оценка расхода по ценам провайдера ($/1M токенов)."""
        p = self.provider
        return (
            (prompt - cached) * p.price_in
            + cached * p.price_cached
            + completion * p.price_out
        ) / 1_000_000

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_balance(self) -> dict:
        """Запросить баланс аккаунта (есть только у DeepSeek)."""
        if not self.provider.has_balance_api:
            return {"error": f"{self.provider.title} не отдаёт баланс через API"}
        root = self.base_url
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        url = f"{root}/user/balance"
        try:
            resp = await self._client.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_balance failed: %s", exc)
            return {"error": str(exc)}

    async def describe_image(self, base64_data: str, mime: str = "image/jpeg") -> str:
        """Описание изображения: MiniMax vision (MiniMax-M3), если включён;
        иначе локальная Ollama moondream."""
        mode = (settings.vision_provider or "auto").lower()
        if mode == "off":
            return "(vision выключен: VISION_PROVIDER=off)"
        if mode != "ollama":
            try:
                from ..media_ai import get_media_ai  # noqa: PLC0415
                ai = get_media_ai()
                if ai.vision_enabled:
                    return await ai.describe_image_b64(base64_data, mime)
            except Exception as exc:  # noqa: BLE001
                log.warning("minimax vision failed: %s", exc)
                if mode == "minimax":
                    return f"(vision error: {exc})"
        return await self._describe_image_ollama(base64_data, mime)

    async def _describe_image_ollama(self, base64_data: str, mime: str = "image/jpeg") -> str:
        """Описание изображения через локальную Ollama moondream (нативный API)."""
        payload: dict[str, Any] = {
            "model": "moondream",
            "prompt": (
                "Describe this image briefly in Russian. "
                "What is shown? Is there text, diagrams, tables? "
                "If it's a UI screenshot — what is displayed?"
            ),
            "images": [base64_data],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                )
            if resp.status_code >= 400:
                log.warning("Ollama moondream HTTP %s: %s", resp.status_code, resp.text[:200])
                return f"(vision error {resp.status_code})"
            data = resp.json()
            return data.get("response", "(пустой ответ)")
        except Exception as exc:  # noqa: BLE001
            log.warning("Ollama moondream failed: %s", exc)
            return f"(vision unavailable: {exc})"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.provider.supports_thinking_flag:
            # DeepSeek: явно отключаем thinking mode — нам нужен быстрый
            # tool-calling. Другим вендорам этот параметр не шлём (400).
            payload["thinking"] = {"type": "disabled"}
        prefixes = self.provider.extra_payload_prefixes
        if not prefixes or self.model.startswith(tuple(prefixes)):
            payload.update(self.provider.extra_payload)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        url = _completions_url(self.base_url)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider.key == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/ducksidian"
            headers["X-Title"] = "DuckSidian"
        _RETRYABLE = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                      httpx.ConnectTimeout, httpx.NetworkError, httpx.WriteError)
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                resp = await self._client.post(
                    url,
                    headers=headers,
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )
                break
            except _RETRYABLE as exc:
                if attempt == max_attempts - 1:
                    log.error("LLM request failed after %d attempts: %s: %r",
                              max_attempts, type(exc).__name__, exc)
                    raise
                wait = 1 + (attempt % 5)  # 1..5s, без экспоненты — не душим бота
                log.warning("LLM transient error (attempt %d/%d): %s: %r — retry in %ds",
                            attempt + 1, max_attempts, type(exc).__name__, exc, wait)
                await asyncio.sleep(wait)
        if resp.status_code == 429:
            # Rate limit — одна длинная пауза и повтор.
            retry_after = resp.headers.get("retry-after")
            try:
                wait = min(float(retry_after), 60.0) if retry_after else 10.0
            except ValueError:
                wait = 10.0
            log.warning("LLM 429 rate limit, waiting %.0fs", wait)
            await asyncio.sleep(wait)
            resp = await self._client.post(
                url, headers=headers,
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )
        if resp.status_code >= 400:
            log.error("LLM HTTP %s [%s]: %s", resp.status_code, self.provider.key, resp.text[:500])
            resp.raise_for_status()
        data = resp.json()
        resp_model = data.get("model", self.model)
        prompt, completion, cached = extract_usage(data)
        if prompt or completion:
            log.info(
                "LLM usage: provider=%s model=%s prompt=%d completion=%d total=%d cached=%d",
                self.provider.key, resp_model, prompt, completion, prompt + completion, cached,
            )
            tokens_log.info(
                "call provider=%s model=%s prompt=%d compl=%d total=%d cached=%d",
                self.provider.key, resp_model, prompt, completion, prompt + completion, cached,
            )
        return data


# Backward compat: старое имя класса.
DeepSeekClient = LLMClient
