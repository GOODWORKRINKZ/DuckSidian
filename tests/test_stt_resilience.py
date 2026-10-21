"""STT не должен ронять машину (инцидент 2026-10-21): MiniMax ASR — по одному
запросу и с повтором на 429/5xx, локальный Whisper — одна загрузка модели и
отказ от загрузки при нехватке памяти."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

from bot import media_ai, media_parser
from bot.config import settings
from bot.media_ai import MediaAI, MediaAIError


def _client(handler) -> MediaAI:
    ai = MediaAI()
    ai.api_key = "mmk"
    ai._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ai


@pytest.fixture
def voice(tmp_path: Path) -> Path:
    p = tmp_path / "voice_1.ogg"
    p.write_bytes(b"OggS fake")
    return p


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch):
    monkeypatch.setattr(media_ai, "ASR_RETRY_DELAY", 0)


async def test_asr_retries_429_then_succeeds(voice):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "conn limit (1041)"}})
        return httpx.Response(200, json={"text": "750 рублей, стрижка", "duration": 2.9})

    assert await _client(handler).transcribe(voice) == "750 рублей, стрижка"
    assert len(calls) == 2


async def test_asr_retries_transport_error(voice):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, json={"text": "ok"})

    assert await _client(handler).transcribe(voice) == "ok"
    assert len(calls) == 2


async def test_asr_gives_up_after_attempts(voice):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text="conn limit")

    with pytest.raises(MediaAIError, match="429"):
        await _client(handler).transcribe(voice)
    assert len(calls) == media_ai.ASR_ATTEMPTS


async def test_asr_client_error_not_retried(voice):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad file")

    with pytest.raises(MediaAIError, match="400"):
        await _client(handler).transcribe(voice)
    assert len(calls) == 1


async def test_asr_requests_run_one_at_a_time(voice):
    in_flight = 0
    peak = 0

    async def handler(req: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json={"text": "ok"})

    ai = _client(handler)
    assert await asyncio.gather(*(ai.transcribe(voice) for _ in range(3))) == ["ok"] * 3
    assert peak == 1


# ── Whisper ──────────────────────────────────────────────────────────────────

class _FakeModel:
    def transcribe(self, path: str) -> dict:
        return {"text": "пец 70 рублей", "language": "ru"}


@pytest.fixture
def fake_whisper(monkeypatch):
    loads = []

    def load_model(name: str) -> _FakeModel:
        loads.append(name)
        time.sleep(0.05)  # окно для гонки двух загрузок
        return _FakeModel()

    monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(media_parser, "_WHISPER_MODEL", None)
    monkeypatch.setattr(media_parser, "_WHISPER_MODEL_NAME", "")
    monkeypatch.setattr(settings, "whisper_model", "small")
    return loads


def test_whisper_skips_load_on_low_memory(monkeypatch, fake_whisper, voice):
    monkeypatch.setattr(settings, "whisper_min_free_mb", 2500)
    monkeypatch.setattr(media_parser, "_mem_available_mb", lambda: 900)
    out = media_parser._transcribe_audio_sync(voice)
    assert out.startswith("(голосовое не распознано: мало памяти")
    assert fake_whisper == []


def test_whisper_loads_when_memory_ok(monkeypatch, fake_whisper, voice):
    monkeypatch.setattr(settings, "whisper_min_free_mb", 2500)
    monkeypatch.setattr(media_parser, "_mem_available_mb", lambda: 8000)
    assert media_parser._transcribe_audio_sync(voice) == "пец 70 рублей"
    assert fake_whisper == ["small"]


def test_whisper_loaded_once_for_concurrent_voices(monkeypatch, fake_whisper, voice):
    monkeypatch.setattr(settings, "whisper_min_free_mb", 0)
    out: list[str] = []
    threads = [threading.Thread(target=lambda: out.append(media_parser._transcribe_audio_sync(voice)))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == ["пец 70 рублей"] * 3
    assert fake_whisper == ["small"]


async def test_minimax_429_does_not_fall_back_to_whisper(monkeypatch, fake_whisper, voice):
    """Сценарий инцидента: первый ответ 429 — повтор в MiniMax, Whisper не грузится."""
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="conn limit (1041)")
        return httpx.Response(200, json={"text": "750 рублей, стрижка Кирилл"})

    ai = _client(handler)
    monkeypatch.setattr(media_ai, "get_media_ai", lambda: ai)
    monkeypatch.setattr(settings, "stt_provider", "auto")
    assert await media_parser.transcribe_audio(voice) == "750 рублей, стрижка Кирилл"
    assert fake_whisper == []
