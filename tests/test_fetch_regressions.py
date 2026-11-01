"""Падения инструментов из bot.log (2026-10-26, 2026-10-30): fetch_url на
странице, где <title> содержит теги, и аудио-транскрипт YouTube (NameError Path)."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bot.agent import tools
from bot.agent.tools import ToolExecutor
from bot.config import settings
from bot.personal import PersonalSpace


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolExecutor:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    (tmp_path / "AGENTS.md").write_text("# schema", encoding="utf-8")

    async def ask(question, options):  # pragma: no cover - не вызывается
        return options[0]

    return ToolExecutor(PersonalSpace(42, "Иван").wiki, ask)


async def test_fetch_url_title_with_nested_tags(executor, monkeypatch):
    html = ("<html><head><title><span>Мотор-колесо</span> 6.5\"</title></head>"
            "<body><main>Цена 1 990 ₽</main></body></html>")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    real = httpx.AsyncClient
    monkeypatch.setattr(tools.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler)))
    out = await executor._fetch_url("https://www.aliexpress.com/item/1.html")
    assert "Мотор-колесо" in out
    assert "Цена 1 990 ₽" in out


async def test_yt_audio_transcript_reaches_ytdlp(executor, monkeypatch):
    calls: list = []

    class _Proc:
        returncode = 1

        async def communicate(self):
            return b"", b"ERROR: video unavailable"

    async def fake_exec(*cmd, **kw):
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/yt-dlp")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    text, err = await executor._yt_audio_transcript("https://youtu.be/x")
    assert calls and calls[0][0] == "yt-dlp"
    assert text == "" and "video unavailable" in err
