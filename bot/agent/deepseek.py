"""DeepSeek client (OpenAI-compatible chat completions с tool-calling)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)
tokens_log = logging.getLogger("bot.tokens")


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float | None = None):
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.model = model or settings.deepseek_model
        log.info("DeepSeekClient init: model=%s base_url=%s", self.model, self.base_url)
        # connect — быстрый (ловим DNS/network), read — долгий
        # (DeepSeek-chat иногда думает >2 минут на tool-calling).
        if timeout is None:
            tmo = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=10.0)
        else:
            tmo = httpx.Timeout(timeout)
        # Форсируем IPv4: на этой машине дефолтный сокет иногда висит ConnectTimeout 15s
        # к api.deepseek.com (curl/curl -4 ок, голый httpx — нет). Bind на 0.0.0.0 чинит.
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1)
        self._client = httpx.AsyncClient(timeout=tmo, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_balance(self) -> dict:
        """Запросить текущий баланс аккаунта DeepSeek."""
        url = f"{self.base_url}/user/balance"
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
            # Явно отключаем thinking mode — нам нужен быстрый tool-calling,
            # а не размышления. Без этого v4-flash иногда сам включает thinking,
            # что требует особой обработки reasoning_content.
            "thinking": {"type": "disabled"},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        url = f"{self.base_url}/v1/chat/completions"
        _RETRYABLE = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                      httpx.ConnectTimeout, httpx.NetworkError, httpx.WriteError)
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                resp = await self._client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )
                break
            except _RETRYABLE as exc:
                if attempt == max_attempts - 1:
                    log.error("DeepSeek request failed after %d attempts: %s: %r",
                              max_attempts, type(exc).__name__, exc)
                    raise
                wait = 1 + (attempt % 5)  # 1..5s, без экспоненты — не душим бота
                log.warning("DeepSeek transient error (attempt %d/%d): %s: %r — retry in %ds",
                            attempt + 1, max_attempts, type(exc).__name__, exc, wait)
                await asyncio.sleep(wait)
        if resp.status_code >= 400:
            log.error("DeepSeek HTTP %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        data = resp.json()
        resp_model = data.get("model", self.model)
        usage = data.get("usage") or {}
        if usage:
            log.info(
                "DeepSeek usage: model=%s prompt=%d completion=%d total=%d cached=%d",
                resp_model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
                usage.get("prompt_cache_hit_tokens", 0),
            )
            tokens_log.info(
                "call model=%s prompt=%d compl=%d total=%d cached=%d",
                resp_model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
                usage.get("prompt_cache_hit_tokens", 0),
            )
        return data
