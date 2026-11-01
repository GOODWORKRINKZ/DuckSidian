"""Vision через MiniMax-M3 (инцидент 2026-11-01): с включённым thinking модель
тратит весь max_tokens на <think> по фото чека/страницы и возвращает пустой
ответ. Thinking выключаем, а пустой ответ считаем сбоем — он не должен
попадать в кэш describe_media и в sidecar."""
from __future__ import annotations

import json

import httpx
import pytest

from bot.config import settings
from bot.media_ai import MediaAI, MediaAIError


def _client(handler) -> MediaAI:
    ai = MediaAI()
    ai.api_key = "mmk"
    ai._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ai


def _reply(content: str, finish: str = "stop") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 2700, "completion_tokens": 700},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


async def test_vision_disables_thinking():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=_reply("МАГНИТ, итог 619,12 ₽"))

    text = await _client(handler).describe_image_b64("AAAA")
    assert text == "МАГНИТ, итог 619,12 ₽"
    assert seen["thinking"] == {"type": "disabled"}
    assert seen["max_tokens"] == settings.minimax_vision_max_tokens


async def test_vision_thinking_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "minimax_vision_thinking", True)
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=_reply("<think>...</think>ок"))

    assert await _client(handler).describe_image_b64("AAAA") == "ок"
    assert "thinking" not in seen


async def test_vision_think_only_answer_is_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply("<think>We need transcribe…", "length"))

    with pytest.raises(MediaAIError, match="finish=length"):
        await _client(handler).describe_image_b64("AAAA")
