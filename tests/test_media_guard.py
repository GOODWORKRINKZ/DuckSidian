"""Защита от выдуманных картинок: блок IMAGES / «посмотрел картинку» без fetch_image,
describe_media, generate_image и без реального файла в vault."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bot.agent.loop import (
    MEDIA_NUDGE,
    MEDIA_TOOLS,
    UNCONFIRMED_MEDIA_NOTE,
    WRITE_NUDGE,
    claims_media,
    run_agent,
)
from bot.agent.tools import PERSONAL_WRITE_TOOLS, ToolExecutor, personal_tool_schemas
from bot.config import settings
from bot.orchestrator import UNCONFIRMED_WRITE_NOTE, Orchestrator, mark_unconfirmed_media
from bot.personal import PersonalSpace

# Ответ из инцидента 2026-10-15: ни одного инструмента, выдуманный файл и описание.
FAKE_ANDI = (
    "Посмотрел изображение 大陆智源 ANDI — на картинке видно белого робота-дезинфектора "
    "с УФ-лампами на колёсной базе.\n\n"
    "**IMAGES:**\n`raw/assets/2026-10-15/web_andi-disinfection.jpg`"
)


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            return {"choices": [{"message": {"content": "(no more)"}}]}
        return self._responses.pop(0)


def _call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": cid, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }]}}]}


def _text(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    monkeypatch.setattr(settings, "git_autocommit", False)
    (tmp_path / "AGENTS.md").write_text("# schema", encoding="utf-8")
    return tmp_path


@pytest.fixture
def space(vault: Path) -> PersonalSpace:
    return PersonalSpace(42, "Иван")


def _executor(space: PersonalSpace) -> ToolExecutor:
    async def ask(_q, _o):
        return ""
    return ToolExecutor(space.wiki, ask)


def _count(client: _FakeClient, content: str) -> int:
    hits = [m for m in client.calls[-1] if m.get("content") == content]
    assert all(m["role"] == "system" for m in hits)
    return len(hits)


async def _run(space: PersonalSpace, client: _FakeClient):
    return await run_agent(client=client, executor=_executor(space), system_prompt="s",
                           user_prompt="u", max_steps=8, write_tools=PERSONAL_WRITE_TOOLS,
                           media_tools=MEDIA_TOOLS)


def test_claims_media_detection() -> None:
    for s in ("Посмотрел картинку ANDI", "Скачал фото робота", "Прикрепляю изображение",
              "Отправляю картинку", "Пришлю фото", "На картинке видно робота",
              "Что увидел по картинке: робот", "Фото прикрепил ниже", "Вот фото ANDI",
              "Вижу на изображении дезинфектор", "Текст\n\nIMAGES:\nraw/assets/a.jpg",
              "Текст\n\n**IMAGES:**\n`raw/assets/a.jpg`", FAKE_ANDI):
        assert claims_media(s), s
    for s in ("Не нашёл картинку ANDI", "Картинку так и не нашёл", "Посмотрел в заметках — ничего",
              "Записал ссылку на фото", "Не скачал фото: сайт закрыт", "Могу прислать картинку?",
              "Записал в wiki/notes/x.md", ""):
        assert not claims_media(s), s


def test_media_tools_are_real() -> None:
    names = {t["function"]["name"] for t in personal_tool_schemas()}
    assert MEDIA_TOOLS == {"fetch_image", "describe_media", "generate_image"}
    assert MEDIA_TOOLS <= names


@pytest.mark.asyncio
async def test_fake_image_nudged_then_fetched(space: PersonalSpace, monkeypatch) -> None:
    async def fake_fetch(self, url: str) -> str:
        p = space.root / "raw/assets/2026-10-15/web_abc.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8jpg")
        return "OK: raw/assets/2026-10-15/web_abc.jpg (1 KB, image/jpeg)"
    monkeypatch.setattr(ToolExecutor, "_fetch_image", fake_fetch)
    client = _FakeClient([
        _text(FAKE_ANDI),
        _call("fetch_image", {"url": "https://example.com/andi.jpg"}, "c1"),
        _call("finish", {"summary": "Вот ANDI.\n\nIMAGES:\nraw/assets/2026-10-15/web_abc.jpg"}, "c2"),
    ])
    run = await _run(space, client)
    assert run.media_nudged and not run.unconfirmed_media and run.finished
    assert "fetch_image" in run.tools_called
    assert _count(client, MEDIA_NUDGE) == 1


@pytest.mark.asyncio
async def test_fake_image_after_nudge_unconfirmed(space: PersonalSpace) -> None:
    client = _FakeClient([_text(FAKE_ANDI), _text(FAKE_ANDI)])
    run = await _run(space, client)
    assert run.media_nudged and run.unconfirmed_media
    assert len(client.calls) == 2 and _count(client, MEDIA_NUDGE) == 1


@pytest.mark.asyncio
async def test_honest_answer_after_media_nudge(space: PersonalSpace) -> None:
    client = _FakeClient([_text(FAKE_ANDI), _text("Картинку ANDI не нашёл — вот ссылка на сайт.")])
    run = await _run(space, client)
    assert run.media_nudged and not run.unconfirmed_media


@pytest.mark.asyncio
async def test_rephrase_after_write_nudge_still_marked(space: PersonalSpace) -> None:
    """Инцидент: write-guard поймал «сохранил», модель перефразировала в «посмотрел».
    Второй подсказки нет, но выдумка помечается."""
    client = _FakeClient([
        _text("Скачал и сохранил изображение ANDI в raw/assets/2026-10-15/web_andi.jpg."),
        _text(FAKE_ANDI),
    ])
    run = await _run(space, client)
    assert run.write_nudged and run.media_nudged      # оба guard'а сработали на первом ответе
    assert len(client.calls) == 2                     # подсказки досланы один раз
    assert _count(client, WRITE_NUDGE) == 1 and _count(client, MEDIA_NUDGE) == 1
    assert run.unconfirmed_media and not run.unconfirmed_write


@pytest.mark.asyncio
async def test_media_rephrase_after_pure_write_nudge(space: PersonalSpace) -> None:
    client = _FakeClient([_text("Записал заметку про ANDI."), _text(FAKE_ANDI)])
    run = await _run(space, client)
    assert run.write_nudged and not run.media_nudged  # вторую подсказку не даём
    assert len(client.calls) == 2
    assert run.unconfirmed_media


@pytest.mark.asyncio
async def test_existing_vault_file_not_nudged(space: PersonalSpace) -> None:
    p = space.root / "raw/assets/2026-10-01/photo_1.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"jpg")
    client = _FakeClient([
        _call("list_dir", {"path": "raw/assets/2026-10-01"}, "c1"),
        _call("finish", {"summary": "Вот фото из архива.\n\n**IMAGES:**\n- `raw/assets/2026-10-01/photo_1.jpg`"}, "c2"),
    ])
    run = await _run(space, client)
    assert not run.media_nudged and not run.unconfirmed_media
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_describe_media_counts(space: PersonalSpace, monkeypatch) -> None:
    async def fake_describe(self, rel: str) -> str:
        return "робот"
    monkeypatch.setattr(ToolExecutor, "_describe_media", fake_describe)
    client = _FakeClient([
        _call("describe_media", {"path": "raw/assets/x.jpg"}, "c1"),
        _call("finish", {"summary": "На картинке видно робота."}, "c2"),
    ])
    run = await _run(space, client)
    assert not run.media_nudged and not run.unconfirmed_media


@pytest.mark.asyncio
async def test_media_guard_off_by_default(space: PersonalSpace) -> None:
    client = _FakeClient([_text(FAKE_ANDI)])
    run = await run_agent(client=client, executor=_executor(space), system_prompt="s",
                          user_prompt="u", max_steps=3)
    assert run.finished and not run.media_nudged and not run.unconfirmed_media


def test_mark_unconfirmed_media_strips_images() -> None:
    out = mark_unconfirmed_media(FAKE_ANDI)
    assert "IMAGES" not in out and "web_andi-disinfection" not in out
    assert out.endswith(UNCONFIRMED_MEDIA_NOTE)
    assert out.startswith("Посмотрел изображение")


def _orch(client: _FakeClient) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.client = client  # type: ignore[assignment]
    orch.db = None  # type: ignore[assignment]
    orch._run_stats = []
    orch._spaces = {}
    orch._record_run = lambda *a, **k: None  # type: ignore[method-assign]
    return orch


@pytest.mark.asyncio
async def test_personal_turn_marks_fake_image(space: PersonalSpace,
                                              caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient([
        _text("Сохранил картинку ANDI.\n\nIMAGES:\nraw/assets/2026-10-15/web_andi.jpg"),
        _text(FAKE_ANDI),
    ])
    with caplog.at_level(logging.WARNING, logger="bot.orchestrator"):
        answer = await _orch(client).personal_turn(space, "пришли изображение 大陆智源 ANDI",
                                                   message_id=9)
    assert answer.endswith(UNCONFIRMED_MEDIA_NOTE)
    assert "IMAGES" not in answer and UNCONFIRMED_WRITE_NOTE not in answer
    assert any("never fetched" in r.getMessage() and "recovered=False" in r.getMessage()
               for r in caplog.records)
