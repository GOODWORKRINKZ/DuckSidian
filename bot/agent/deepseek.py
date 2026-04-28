"""DeepSeek client (OpenAI-compatible chat completions с tool-calling)."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.model = model or settings.deepseek_model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def describe_image(self, base64_data: str, mime: str = "image/jpeg") -> str:
        """Отправить изображение в DeepSeek VL для описания."""
        payload: dict[str, Any] = {
            "model": "deepseek-vl2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{base64_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Опиши изображение кратко и по делу на русском языке. "
                                "Что изображено? Есть ли текст, схемы, таблицы? "
                                "Если это скриншот интерфейса — что именно показано?"
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        url = f"{self.base_url}/v1/chat/completions"
        resp = await self._client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        if resp.status_code >= 400:
            log.warning("DeepSeek VL HTTP %s: %s", resp.status_code, resp.text[:200])
            return f"(vision API error {resp.status_code})"
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "(vision: unexpected response)"

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
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        url = f"{self.base_url}/v1/chat/completions"
        resp = await self._client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        if resp.status_code >= 400:
            log.error("DeepSeek HTTP %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        return resp.json()
