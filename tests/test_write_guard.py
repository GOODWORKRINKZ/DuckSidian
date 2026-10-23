"""Защита от выдуманных записей: ответ «записал/готово» без вызова write-инструментов."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bot.agent.loop import WRITE_NUDGE, claims_write, run_agent
from bot.agent.tools import PERSONAL_WRITE_TOOLS, ToolExecutor, personal_tool_schemas
from bot.config import settings
from bot.orchestrator import UNCONFIRMED_WRITE_NOTE, Orchestrator, mark_unconfirmed_write
from bot.personal import PersonalSpace


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


def _nudges(client: _FakeClient) -> int:
    nudges = [m for m in client.calls[-1] if m.get("content") == WRITE_NUDGE]
    assert all(m["role"] == "system" for m in nudges)   # служебное, не реплика пользователя
    return len(nudges)


def test_claims_write_detection() -> None:
    for s in ("Записал в wiki/projects/X.md", "добавила чек 74,99 ₽", "Сохранили.",
              "Обновил раздел 6–9", "Создал страницу", "Дополнил историю", "✅ Готово!",
              "Записано: firstvds", "внёс правку"):
        assert claims_write(s), s
    for s in ("Ничего не записал — нужен номер раздела.", "Не сохранил, уточни сумму",
              "Нашёл 3 упоминания в raw/daily", "Готовность 80%", "добавить?", ""):
        assert not claims_write(s), s


def test_write_tools_constant() -> None:
    names = {t["function"]["name"] for t in personal_tool_schemas()}
    assert {"edit_file", "write_file", "append_file", "create_workspace",
            "set_workspace_icon", "generate_image"} <= PERSONAL_WRITE_TOOLS
    assert PERSONAL_WRITE_TOOLS <= names           # только реальные инструменты
    assert not {"read_file", "search_wiki", "finish", "list_dir"} & PERSONAL_WRITE_TOOLS


@pytest.mark.asyncio
async def test_fake_finish_nudged_then_writes(space: PersonalSpace) -> None:
    client = _FakeClient([
        _call("finish", {"summary": "Записал разделы 6–9 в проект."}, "c1"),
        _call("append_file", {"path": "wiki/notes/x.md", "content": "разделы 6–9"}, "c2"),
        _call("finish", {"summary": "Записал разделы 6–9 в wiki/notes/x.md."}, "c3"),
    ])
    run = await run_agent(client=client, executor=_executor(space), system_prompt="s",
                          user_prompt="u", max_steps=8, write_tools=PERSONAL_WRITE_TOOLS)
    assert run.write_nudged and not run.unconfirmed_write and run.finished
    assert run.summary.endswith("wiki/notes/x.md.")
    assert "разделы 6–9" in (space.root / "wiki/notes/x.md").read_text(encoding="utf-8")
    assert _nudges(client) == 1


@pytest.mark.asyncio
async def test_plain_text_claim_still_unconfirmed(space: PersonalSpace) -> None:
    client = _FakeClient([_text("Добавил чек 74,99 ₽. Готово."), _text("Добавил, всё готово.")])
    run = await run_agent(client=client, executor=_executor(space), system_prompt="s",
                          user_prompt="u", max_steps=8, write_tools=PERSONAL_WRITE_TOOLS)
    assert run.write_nudged and run.unconfirmed_write
    assert len(client.calls) == 2 and _nudges(client) == 1   # подсказка только один раз


@pytest.mark.asyncio
async def test_honest_answer_after_nudge(space: PersonalSpace) -> None:
    client = _FakeClient([_text("Сохранил заметку."), _text("Ничего не записал: уточни проект.")])
    run = await run_agent(client=client, executor=_executor(space), system_prompt="s",
                          user_prompt="u", max_steps=8, write_tools=PERSONAL_WRITE_TOOLS)
    assert run.write_nudged and not run.unconfirmed_write
    assert run.summary.startswith("Ничего не записал")


@pytest.mark.asyncio
async def test_real_write_not_nudged(space: PersonalSpace) -> None:
    client = _FakeClient([
        _call("write_file", {"path": "wiki/notes/y.md", "content": "y"}, "c1"),
        _call("finish", {"summary": "Записал в wiki/notes/y.md"}, "c2"),
    ])
    run = await run_agent(client=client, executor=_executor(space), system_prompt="s",
                          user_prompt="u", max_steps=8, write_tools=PERSONAL_WRITE_TOOLS)
    assert not run.write_nudged and not run.unconfirmed_write
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_guard_off_without_write_tools(space: PersonalSpace) -> None:
    client = _FakeClient([_text("Готово, записал.")])
    run = await run_agent(client=client, executor=_executor(space), system_prompt="s",
                          user_prompt="u", max_steps=3)
    assert run.finished and not run.write_nudged and not run.unconfirmed_write


def test_mark_unconfirmed_write_before_images() -> None:
    assert mark_unconfirmed_write("Записал.") == f"Записал.\n\n{UNCONFIRMED_WRITE_NOTE}"
    out = mark_unconfirmed_write("Сделал картинку.\n\nIMAGES:\nraw/assets/a.png")
    assert out.index(UNCONFIRMED_WRITE_NOTE) < out.index("IMAGES:")
    assert out.endswith("IMAGES:\nraw/assets/a.png")


def _orch(client: _FakeClient) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.client = client  # type: ignore[assignment]
    orch.db = None  # type: ignore[assignment]
    orch._run_stats = []
    orch._spaces = {}
    orch._record_run = lambda *a, **k: None  # type: ignore[method-assign]
    return orch


@pytest.mark.asyncio
async def test_personal_turn_marks_and_logs(space: PersonalSpace, caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient([_text("Записал перевод 90к."), _text("Записал перевод 90к в заметки.")])
    with caplog.at_level(logging.WARNING, logger="bot.orchestrator"):
        answer = await _orch(client).personal_turn(space, "перевод 90к", message_id=7)
    assert answer.endswith(UNCONFIRMED_WRITE_NOTE)
    rec = [r for r in caplog.records if r.name == "bot.orchestrator" and r.levelno == logging.WARNING]
    assert rec and "Иван" in rec[0].getMessage() and "recovered=False" in rec[0].getMessage()


@pytest.mark.asyncio
async def test_personal_turn_recovered_after_nudge(space: PersonalSpace,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient([
        _text("Записал."),
        _call("append_file", {"path": "wiki/notes/z.md", "content": "z"}, "c1"),
        _call("finish", {"summary": "Записал в wiki/notes/z.md"}, "c2"),
    ])
    with caplog.at_level(logging.WARNING, logger="bot.orchestrator"):
        answer = await _orch(client).personal_turn(space, "z", message_id=8)
    assert answer == "Записал в wiki/notes/z.md\n\n✍️ Изменено: `wiki/notes/z.md`"
    assert any("recovered=True" in r.getMessage() for r in caplog.records)


def test_strip_footers() -> None:
    from bot.orchestrator import strip_footers
    text = ("Записала: 21.10 · 70 ₽\n\n✍️ Изменено: `wiki/projects/Траты.md`\n\n"
            "✍ Изменено: `log.md`\n\n⚠️ в vault ничего не записано")
    assert strip_footers(text) == "Записала: 21.10 · 70 ₽"
    assert strip_footers("Изменено расписание — без футера") == "Изменено расписание — без футера"


@pytest.mark.asyncio
async def test_personal_turn_single_real_footer(space: PersonalSpace) -> None:
    # Модель копирует футер из истории — в ответе остаётся только футер от кода.
    client = _FakeClient([
        _call("append_file", {"path": "wiki/notes/z.md", "content": "z"}, "c1"),
        _call("finish", {"summary": "Записал z\n\n✍️ Изменено: `wiki/notes/z.md`\n\n"
                                    "✍️ Изменено: `wiki/notes/z.md`"}, "c2"),
    ])
    answer = await _orch(client).personal_turn(space, "z", message_id=9)
    assert answer == "Записал z\n\n✍️ Изменено: `wiki/notes/z.md`"


@pytest.mark.asyncio
async def test_personal_turn_drops_fake_footer(space: PersonalSpace) -> None:
    # Ничего не записано, а модель приписала «Изменено» — футер убираем.
    client = _FakeClient([_text("Не разобрал картинку, напиши сумму.\n\n"
                                "✍️ Изменено: `wiki/projects/Семейные-траты.md`")])
    answer = await _orch(client).personal_turn(space, "фото", message_id=10)
    assert answer == "Не разобрал картинку, напиши сумму."
