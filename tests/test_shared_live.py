"""Совместные воркспейсы: что видят партнёры, пока агент отвечает коллеге."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot.config import settings
from bot.db import DB
from bot.handlers.shared_live import (
    LiveNotice,
    _fmt_ago,
    RunningRequest,
    clear_running,
    is_busy,
    lock_for,
    mark_running,
    member_targets,
    members_report,
    queue_wait_text,
    request_snippet,
    running_request,
    serialized_turn,
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


@pytest.mark.asyncio
async def test_serialized_turn_queues_second_and_names_first() -> None:
    key = "shared/тест-serialized"
    clear_running(key)
    log: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    seen: list[RunningRequest | None] = []

    async def body_a() -> str:
        log.append("a:start")
        first_started.set()
        await release_first.wait()
        log.append("a:end")
        return "a"

    async def body_b() -> str:
        log.append("b:start")
        return "b"

    async def on_queued(rr) -> None:
        seen.append(rr)
        log.append("b:queued")

    ta = asyncio.create_task(serialized_turn(key, author="Иван", author_id=111, text="хлеб", body=body_a))
    await first_started.wait()
    assert is_busy(key) and running_request(key).author == "Иван"
    # «Долгая расшифровка» второго участника идёт параллельно, пока Иван держит ключ.
    log.append("b:stt")
    tb = asyncio.create_task(serialized_turn(key, author="Таня", author_id=222, text="молоко",
                                             body=body_b, on_queued=on_queued))
    await asyncio.sleep(0.02)
    assert log == ["a:start", "b:stt", "b:queued"]            # b ждёт, не стартовал
    assert seen and seen[0].author == "Иван" and seen[0].snippet == "хлеб"
    release_first.set()
    assert await asyncio.wait_for(asyncio.gather(ta, tb), 1) == ["a", "b"]
    assert log == ["a:start", "b:stt", "b:queued", "a:end", "b:start"]
    assert not is_busy(key) and running_request(key) is None


@pytest.mark.asyncio
async def test_serialized_turn_clears_marker_on_error() -> None:
    key = "shared/тест-ошибка"

    async def boom() -> None:
        raise RuntimeError("agent down")

    with pytest.raises(RuntimeError):
        await serialized_turn(key, author="Иван", author_id=111, text="x", body=boom)
    assert running_request(key) is None and not is_busy(key)
    assert lock_for(key) is lock_for(key)


@pytest.mark.asyncio
async def test_last_activity_in_workspace(tmp_path: Path) -> None:
    db = await _db(tmp_path)
    t0 = datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)
    rows = [(111, 1, 7, 111, "хлеб"), (222, 1, 9, 222, "молоко"), (111, 2, 8, 111, "не сюда"),
            (222, 2, 9, 999, "бот"), (111, 3, 7, 111, "итого?")]
    for i, (chat, mid, topic, uid, text) in enumerate(rows):
        await db.save_message(chat_id=chat, message_id=mid, topic_id=topic, user_id=uid, username=None,
                              full_name="x", text=text, reply_to_message_id=None, ts=t0 + timedelta(minutes=i))
    seen = await db.last_activity_in_workspace(PS)
    assert set(seen) == {111, 222, 999}
    assert datetime.fromisoformat(seen[111]) == t0 + timedelta(minutes=4)   # последнее в теме 7, не из темы 8
    assert datetime.fromisoformat(seen[222]) == t0 + timedelta(minutes=1)
    assert await db.last_activity_in_workspace("нет-такого") == {}


def test_members_report_shows_last_seen_and_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tz", "Europe/Moscow")
    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    members = [(111, "Иван (@ivan, 111)", True), (222, "Таня (222)", False), (333, "333", False)]
    seen = {111: (now - timedelta(seconds=30)).isoformat(),
            222: (now - timedelta(hours=1, days=1)).isoformat()}
    rr = RunningRequest("Иван", 111, "хлеб <b>", started_at=now - timedelta(seconds=12))
    text = members_report("Семейные траты", members, seen, rr, now)
    assert text.splitlines()[0] == "<b>Участники «Семейные траты»:</b>"
    assert "• Иван (@ivan, 111) 👑 владелец — писал только что" in text
    assert "• Таня (222) — писал вчера 14:00" in text          # 11:00 UTC вчера = 14:00 МСК
    assert "• 333 — ещё не писал здесь" in text
    assert "⏳ Сейчас выполняется запрос <b>Иван</b> («хлеб &lt;b&gt;»), 12 с." in text
    assert text.rstrip().endswith("/ws kick <user_id|@username>")
    idle = members_report("ws", members, {}, None, now)
    assert "💤 Сейчас никто ничего не выполняет." in idle


def test_fmt_ago_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tz", "UTC")
    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    assert _fmt_ago((now - timedelta(minutes=5)).isoformat(), now) == "5 мин назад"
    assert _fmt_ago((now - timedelta(hours=3)).isoformat(), now) == "3 ч назад"
    assert _fmt_ago("2026-10-12T09:15:00+00:00", now) == "12.10 09:15"
    assert _fmt_ago("2026-10-15T08:00:00", now) == "4 ч назад"      # naive = UTC
    assert _fmt_ago("мусор", now) == "мусор"
