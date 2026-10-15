"""Совместные воркспейсы: что видят партнёры, пока агент отвечает коллеге."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bot.db import DB
from bot.handlers.shared_live import member_targets, typing_many

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
