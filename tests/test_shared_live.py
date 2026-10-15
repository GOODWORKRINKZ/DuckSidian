"""Совместные воркспейсы: что видят партнёры, пока агент отвечает коллеге."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot.db import DB
from bot.handlers.shared_live import (
    LiveNotice,
    RunningRequest,
    clear_running,
    mark_running,
    member_targets,
    queue_wait_text,
    request_snippet,
    running_request,
    typing_many,
)

SLUG = "семейные-траты"
PS = "shared/" + SLUG


class _FakeBot:
    def __init__(self) -> None:
        self.actions: list[tuple[int, int | None, str]] = []

    async def send_chat_action(self, chat_id, action, message_thread_id=None):
        self.actions.append((chat_id, message_thread_id, action))


async def _db(tmp_path: Path) -> DB:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    await db.register_shared_vault(SLUG, 111)
    await db.add_shared_member(SLUG, 222, 111)
    await db.upsert_workspace(111, 7, "Семейные траты", PS)
    await db.upsert_workspace(222, 9, "Семейные траты", PS)
    await db.upsert_workspace(333, 5, "Семейные траты", PS)   # тема есть, но в участниках не числится
    await db.upsert_workspace(111, 8, "Другое", "Другое")
    return db


@pytest.mark.asyncio
async def test_member_targets_other_members_only(tmp_path: Path) -> None:
    db = await _db(tmp_path)
    assert await member_targets(db, PS, exclude_chat_id=111) == [(222, 9)]
    assert await member_targets(db, PS, exclude_chat_id=222) == [(111, 7)]
    assert sorted(await member_targets(db, PS)) == [(111, 7), (222, 9)]
    assert await member_targets(db, "Другое", 111) == []        # не shared — никого


@pytest.mark.asyncio
async def test_typing_many_hits_every_topic_until_stopped() -> None:
    bot = _FakeBot()
    stop = asyncio.Event()
    task = asyncio.create_task(typing_many(bot, [(222, 9), (333, 5)], stop, period=0.01))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, 1)
    assert (222, 9, "typing") in bot.actions and (333, 5, "typing") in bot.actions
    assert len(bot.actions) >= 4                                 # повторяется по кругу


@pytest.mark.asyncio
async def test_typing_many_without_targets_returns_immediately() -> None:
    bot = _FakeBot()
    await asyncio.wait_for(typing_many(bot, [], asyncio.Event()), 1)
    assert bot.actions == []


class _NoticeBot:
    """send_message / edit_message_text; edit падает для чатов из fail_edit."""

    def __init__(self, fail_edit: set[int] = frozenset()) -> None:
        self.fail_edit = set(fail_edit)
        self.sent: list[tuple[int, int | None, str]] = []
        self.edited: list[tuple[int, int, str]] = []
        self._next = 100

    async def send_message(self, chat_id, text, message_thread_id=None, parse_mode=None):
        self._next += 1
        self.sent.append((chat_id, message_thread_id, text))
        return SimpleNamespace(message_id=self._next)

    async def edit_message_text(self, text, chat_id, message_id, parse_mode=None):
        if chat_id in self.fail_edit:
            raise RuntimeError("Bad Request: message can't be edited")
        self.edited.append((chat_id, message_id, text))


def test_request_snippet_is_one_line_and_cut() -> None:
    assert request_snippet("  хлеб\n74,99  ") == "хлеб 74,99"
    assert request_snippet("а" * 100) == "а" * 80 + "…"
    assert request_snippet(None) == ""


@pytest.mark.asyncio
async def test_live_notice_one_message_edited_in_place() -> None:
    bot = _NoticeBot()
    n = LiveNotice(bot, [(222, 9), (333, 5)], author="Иван", ws_name="Семейные траты",
                   request="хлеб 74,99 <b>")
    await n.start()
    assert [(c, t) for c, t, _ in bot.sent] == [(222, 9), (333, 5)]
    assert bot.sent[0][2] == "⏳ <b>Иван</b>: «хлеб 74,99 &lt;b&gt;» — работаю"
    await n.finish("Записал: хлеб 74,99\nИтого за день 120")
    assert len(bot.sent) == 2                                   # новых сообщений нет
    assert [(c, m) for c, m, _ in bot.edited] == [(222, 101), (333, 102)]
    assert bot.edited[0][2] == ("👥 <b>Иван</b> в «Семейные траты»:\n"
                                "<i>хлеб 74,99 &lt;b&gt; → Записал: хлеб 74,99 Итого за день 120</i>")


@pytest.mark.asyncio
async def test_live_notice_falls_back_to_new_message() -> None:
    bot = _NoticeBot(fail_edit={333})
    n = LiveNotice(bot, [(222, 9), (333, 5)], author="Иван", ws_name="ws", request="итого?")
    await n.start()
    await n.finish("120")
    assert [(c, m) for c, m, _ in bot.edited] == [(222, 101)]
    assert bot.sent[-1][:2] == (333, 5) and bot.sent[-1][2].startswith("👥 <b>Иван</b> в «ws»")


@pytest.mark.asyncio
async def test_live_notice_fail_and_no_targets() -> None:
    bot = _NoticeBot()
    n = LiveNotice(bot, [(222, 9)], author="Иван", ws_name="ws", request="x")
    await n.start()
    await n.fail("TimeoutError")
    assert bot.edited[0][2] == "❌ <b>Иван</b>: «x» — ошибка агента (TimeoutError)"
    empty = LiveNotice(bot, [], author="Иван", ws_name="ws", request="x")
    await empty.start()
    await empty.finish("ok")
    assert len(bot.sent) == 1                                   # личный режим — тишина


def test_queue_wait_text_names_running_author() -> None:
    assert queue_wait_text(None, 222) == "⏳ Принял, отвечу сразу после текущего запроса."
    rr = RunningRequest("Иван", 111, "сколько потратили на еду за неделю, разбей по дням и магазинам, " * 3)
    t = queue_wait_text(rr, 222)
    assert t.startswith("⏳ Сейчас выполняется запрос Иван («сколько потратили")
    assert t.endswith("…»), твой следующий в очереди.")
    assert len(rr.snippet) == 81                                 # 80 символов + «…»
    own = queue_wait_text(RunningRequest("Иван", 111, "итого?"), 111)
    assert own == "⏳ Сейчас выполняется твой предыдущий запрос («итого?»), этот следующий в очереди."
    assert "(вложение)" in queue_wait_text(RunningRequest("Иван", 111, ""), 222)


def test_running_registry_per_lock_key() -> None:
    key = "shared/тест-очереди"
    clear_running(key)
    assert running_request(key) is None
    rr = mark_running(key, "Иван", 111, "хлеб")
    assert running_request(key) is rr and rr.elapsed() < 5
    nxt = mark_running(key, "Таня", 222, "молоко")        # следующий запрос перехватил ключ
    clear_running(key, rr)                                 # старый finally не снимает чужую отметку
    assert running_request(key) is nxt
    clear_running(key, nxt)
    assert running_request(key) is None
