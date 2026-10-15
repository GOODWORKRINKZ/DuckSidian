"""Совместные воркспейсы shared/<slug>: автор реплики, общая история, якоря inbox."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.config import settings
from bot.db import DB
from bot.handlers.private import history_messages
from bot.orchestrator import Orchestrator
from bot.personal import PersonalSpace

SLUG = "shared/Семейные-траты"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    monkeypatch.setattr(settings, "git_autocommit", False)
    (tmp_path / "AGENTS.md").write_text("# schema", encoding="utf-8")
    return tmp_path


def test_inbox_anchor_includes_chat_in_shared(vault: Path) -> None:
    personal = PersonalSpace(111, "Иван")
    assert personal.msg_anchor(5, 111) == "msg-5"            # личный vault — как раньше
    rel = personal.append_inbox(who="Иван", text="x", message_id=5, chat_id=111)
    assert "^msg-5\n" in (personal.root / rel).read_text(encoding="utf-8")

    shared = PersonalSpace.shared("Семейные-траты", "Иван")
    ts = datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)
    shared.append_inbox(who="Иван", text="хлеб 74,99", ts=ts, message_id=5, chat_id=111)
    shared.append_inbox(who="Tatyana Grishko", text="молоко", ts=ts, message_id=5, chat_id=222)
    text = (shared.root / "raw/inbox/2026-10-15.md").read_text(encoding="utf-8")
    assert "^msg-111-5" in text and "^msg-222-5" in text      # одинаковый message_id не сливается
    assert shared.msg_anchor(5) == "msg-5"                     # без chat_id — старый формат


@pytest.mark.asyncio
async def test_workspace_history_across_members(tmp_path: Path) -> None:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    await db.upsert_workspace(111, 7, "Семейные траты", SLUG)   # тема у владельца
    await db.upsert_workspace(222, 9, "Семейные траты", SLUG)   # тема у жены
    await db.upsert_workspace(111, 8, "Другое", "Другое")
    t0 = datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)
    rows = [
        (111, 1, 7, 111, "Иван", "хлеб 74,99"),
        (222, 1, 9, 222, "Tatyana Grishko", "молоко 120"),
        (111, 2, 8, 111, "Иван", "не про траты"),
        (222, 2, 9, 999, "DuckSidian", "Записал молоко"),
        (111, 3, 7, 111, "Иван", "итого?"),
    ]
    for i, (chat, mid, topic, uid, name, text) in enumerate(rows):
        await db.save_message(chat_id=chat, message_id=mid, topic_id=topic, user_id=uid, username=None,
                              full_name=name, text=text, reply_to_message_id=None,
                              ts=t0 + timedelta(minutes=i))
    got = await db.fetch_last_messages_in_workspace(SLUG, 10)
    assert [r["text"] for r in got] == ["хлеб 74,99", "молоко 120", "Записал молоко", "итого?"]
    assert [r["text"] for r in await db.fetch_last_messages_in_workspace(SLUG, 2)] == ["Записал молоко", "итого?"]

    hist = history_messages(got, bot_id=999, skip=(111, 3), limit=10, label_authors=True)
    assert hist == [
        {"role": "user", "content": "[Иван]: хлеб 74,99"},
        {"role": "user", "content": "[Tatyana Grishko]: молоко 120"},
        {"role": "assistant", "content": "Записал молоко"},
    ]
    # личный режим: без меток, skip по (chat_id, message_id) — не задевает чужой msg 1
    plain = history_messages(got, bot_id=999, skip=(222, 1), limit=2)
    assert plain == [{"role": "assistant", "content": "Записал молоко"},
                     {"role": "user", "content": "итого?"}]


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append([dict(m) for m in messages])
        return {"choices": [{"message": {"content": "Ок, вижу."}}]}


def _orch(client: _FakeClient) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.client = client  # type: ignore[assignment]
    orch.db = None  # type: ignore[assignment]
    orch._run_stats = []
    orch._spaces = {}
    orch._record_run = lambda *a, **k: None  # type: ignore[method-assign]
    return orch


@pytest.mark.asyncio
async def test_personal_turn_shared_prompt_has_real_author(vault: Path) -> None:
    client = _FakeClient()
    shared = PersonalSpace.shared("Семейные-траты", "Tatyana Grishko")
    await _orch(client).personal_turn(
        shared, "молоко 120", message_id=5, workspace={"name": "Семейные траты", "project_slug": "Семейные-траты"},
        author_name="Tatyana Grishko", author_id=222, chat_id=222,
    )
    prompt = client.calls[0][-1]["content"]
    assert "Автор этой реплики: Tatyana Grishko (user_id=222)" in prompt
    assert "— Tatyana Grishko" in prompt and "user_id=0" not in prompt
    assert "#msg-222-5]]" in prompt


@pytest.mark.asyncio
async def test_personal_turn_personal_prompt_unchanged(vault: Path) -> None:
    client = _FakeClient()
    await _orch(client).personal_turn(PersonalSpace(42, "Иван"), "привет", message_id=3)
    prompt = client.calls[0][-1]["content"]
    assert "Владелец: Иван (user_id=42)" in prompt and "#msg-3]]" in prompt
    assert "Автор этой реплики" not in prompt
