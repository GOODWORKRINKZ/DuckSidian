"""Совместные vault'ы: владелец, участники, занятые имена, исключение."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from bot.db import DB
from bot.handlers.private import claim_shared_slug, shared_role

SLUG = "семейные-траты"
PS = "shared/" + SLUG


@pytest.mark.asyncio
async def test_register_owner_and_members(tmp_path: Path) -> None:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    assert await db.shared_role(SLUG, 111) == "none"
    await db.register_shared_vault(SLUG, 111)
    assert await db.shared_role(SLUG, 111) == "owner"
    assert await db.shared_role(SLUG, 222) == "none"
    await db.add_shared_member(SLUG, 222, 111)
    assert await db.shared_role(SLUG, 222) == "member"
    assert await db.shared_members(SLUG) == [111, 222]
    # повторная регистрация не меняет владельца
    await db.register_shared_vault(SLUG, 333)
    assert await db.shared_role(SLUG, 333) == "none"
    assert int((await db.get_shared_vault(SLUG))["owner_user_id"]) == 111


@pytest.mark.asyncio
async def test_claim_slug_refuses_foreign_owner(tmp_path: Path) -> None:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    assert await claim_shared_slug(db, SLUG, 111) is None          # первый занял
    assert await claim_shared_slug(db, SLUG, 111) is None          # свой — можно снова
    err = await claim_shared_slug(db, SLUG, 999)                   # чужой — отказ
    assert err and "занято" in err and "111" in err
    assert await db.shared_role(SLUG, 999) == "none"               # ничего не записалось


@pytest.mark.asyncio
async def test_legacy_vault_without_owner_goes_to_first_user(tmp_path: Path) -> None:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    # vault создан до учёта владельцев: записи нет → первый, кто работает, становится владельцем
    assert await shared_role(db, "старый", 111) == "owner"
    assert await shared_role(db, "старый", 222) == "none"


@pytest.mark.asyncio
async def test_migration_assigns_owner_to_earliest_topic(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    db = DB(path)
    await db.init()
    # эмулируем старую БД: темы двух людей привязаны к shared/<slug>, таблиц владельцев нет
    async with aiosqlite.connect(path) as raw:
        await raw.execute("DROP TABLE shared_vaults")
        await raw.execute("DROP TABLE shared_members")
        await raw.execute(
            "INSERT INTO workspaces (chat_id, topic_id, name, project_slug, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (222, 9, "Семейные траты", PS, "2026-10-14T16:25:00+00:00", "2026-10-14T16:25:00+00:00"),
        )
        await raw.execute(
            "INSERT INTO workspaces (chat_id, topic_id, name, project_slug, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (111, 7, "Семейные траты", PS, "2026-10-14T16:07:00+00:00", "2026-10-14T16:07:00+00:00"),
        )
        await raw.execute(
            "INSERT INTO workspaces (chat_id, topic_id, name, project_slug, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (111, 8, "Личное", "Личное", "2026-10-14T13:00:00+00:00", "2026-10-14T13:00:00+00:00"),
        )
        await raw.commit()
    await DB(path).init()   # повторный старт бота → миграция
    db = DB(path)
    assert await db.shared_role(SLUG, 111) == "owner"    # чья тема раньше
    assert await db.shared_role(SLUG, 222) == "member"
    assert await db.shared_role(SLUG, 333) == "none"
    assert await db.get_shared_vault("Личное") is None   # личные проекты не трогаем


@pytest.mark.asyncio
async def test_remove_member_unbinds_topic(tmp_path: Path) -> None:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    await db.register_shared_vault(SLUG, 111)
    await db.add_shared_member(SLUG, 222, 111)
    await db.upsert_workspace(222, 9, "Семейные траты", PS)
    await db.upsert_workspace(222, 10, "Своё", "Своё")
    await db.remove_shared_member(SLUG, 222)
    assert await db.shared_role(SLUG, 222) == "none"
    assert (await db.get_workspace(222, 9))["project_slug"] is None   # тема больше не ведёт в чужой vault
    assert (await db.get_workspace(222, 10))["project_slug"] == "Своё"
    assert await db.find_workspaces_by_slug(PS) == []


@pytest.mark.asyncio
async def test_user_label(tmp_path: Path) -> None:
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    assert await db.user_label(555) == "555"
    await db.save_message(chat_id=555, message_id=1, topic_id=None, user_id=555, username="tanya",
                          full_name="Tatyana Grishko", text="x", reply_to_message_id=None,
                          ts=datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert await db.user_label(555) == "Tatyana Grishko (@tanya, 555)"
