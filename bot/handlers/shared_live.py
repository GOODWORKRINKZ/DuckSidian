"""Живая картина совместной работы в shared-воркспейсах (личка каждого участника —
своя тема, привязанная к общему vault shared/<slug>).

Здесь всё, что видно партнёрам, пока агент отвечает коллеге: индикатор «печатает»
во всех темах участников, живое сообщение «⏳ … — работаю» → «👥 … → ответ»,
текст ожидания в очереди с именем автора текущего запроса и сводка для /ws members.
Функции вынесены из замыкания setup() в bot/handlers/private.py, чтобы их можно
было тестировать без aiogram-роутера.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape as _esc
from typing import Any, Awaitable, Callable, TypeVar

from ..personal import shared_slug_of

log = logging.getLogger(__name__)

TYPING_PERIOD = 4.5     # Telegram показывает chat action ~5 с
T = TypeVar("T")


async def member_targets(db: Any, project_slug: str, exclude_chat_id: int | None = None
                         ) -> list[tuple[int, int]]:
    """(chat_id, topic_id) тем всех участников воркспейса shared/<slug>, кроме exclude_chat_id.
    Тема учитывается, только если её владелец числится в shared_members (исключённый
    участник с отвязанной темой или чужая тема с тем же именем не попадут)."""
    sslug = shared_slug_of(project_slug)
    if not sslug:
        return []
    try:
        rows = await db.find_workspaces_by_slug(project_slug)
        members = set(await db.shared_members(sslug))
    except Exception as exc:  # noqa: BLE001
        log.debug("member_targets(%s) failed: %s", project_slug, exc)
        return []
    out: list[tuple[int, int]] = []
    for r in rows:
        chat_id, topic_id = int(r["chat_id"]), int(r["topic_id"])
        if chat_id in members and chat_id != exclude_chat_id and (chat_id, topic_id) not in out:
            out.append((chat_id, topic_id))
    return out


async def typing_many(bot: Any, targets: list[tuple[int, int]], stop: asyncio.Event,
                      action: str = "typing", period: float = TYPING_PERIOD) -> None:
    """Слать chat action во все темы targets, пока не выставлен stop."""
    if not targets:
        return
    while not stop.is_set():
        for chat_id, topic_id in targets:
            try:
                await bot.send_chat_action(chat_id, action, message_thread_id=topic_id)
            except Exception as exc:  # noqa: BLE001
                log.debug("typing to %s/%s failed: %s", chat_id, topic_id, exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass


def request_snippet(text: str | None, limit: int = 80) -> str:
    """Первые limit символов запроса в одну строку (для уведомлений и очереди)."""
    s = " ".join((text or "").split())
    return s[:limit] + ("…" if len(s) > limit else "")


class LiveNotice:
    """Одно живое сообщение на запрос в темах остальных участников:
    при старте «⏳ <автор>: «<запрос>» — работаю», по завершении оно же редактируется
    в «👥 <автор> в «<ws>»: запрос → ответ». Не удалось отредактировать — шлём новое."""

    def __init__(self, bot: Any, targets: list[tuple[int, int]], *, author: str,
                 ws_name: str, request: str | None) -> None:
        self.bot = bot
        self.targets = list(targets)
        self.author = author
        self.ws_name = ws_name
        self.request = (request or "").strip()
        self.message_ids: dict[tuple[int, int], int] = {}

    def start_text(self) -> str:
        return f"⏳ <b>{_esc(self.author)}</b>: «{_esc(request_snippet(self.request))}» — работаю"

    def done_text(self, answer: str) -> str:
        body = (self.request + "\n→ " + (answer or "").strip()).strip().replace("\n", " ")
        body = body[:300] + ("…" if len(body) > 300 else "")
        return f"👥 <b>{_esc(self.author)}</b> в «{_esc(self.ws_name)}»:\n<i>{_esc(body)}</i>"

    def fail_text(self, reason: str) -> str:
        return (f"❌ <b>{_esc(self.author)}</b>: «{_esc(request_snippet(self.request))}» — "
                f"ошибка агента ({_esc(reason)})")

    async def start(self) -> None:
        for chat_id, topic_id in self.targets:
            try:
                m = await self.bot.send_message(chat_id, self.start_text(), message_thread_id=topic_id,
                                                parse_mode="HTML")
                self.message_ids[(chat_id, topic_id)] = int(m.message_id)
            except Exception as exc:  # noqa: BLE001
                log.debug("live notice start to %s/%s failed: %s", chat_id, topic_id, exc)

    async def _update(self, text: str) -> None:
        for chat_id, topic_id in self.targets:
            mid = self.message_ids.get((chat_id, topic_id))
            if mid is not None:
                try:
                    await self.bot.edit_message_text(text, chat_id=chat_id, message_id=mid, parse_mode="HTML")
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.debug("live notice edit %s/%s failed, sending anew: %s", chat_id, mid, exc)
            try:
                await self.bot.send_message(chat_id, text, message_thread_id=topic_id, parse_mode="HTML")
            except Exception as exc:  # noqa: BLE001
                log.debug("live notice send to %s/%s failed: %s", chat_id, topic_id, exc)

    async def finish(self, answer: str) -> None:
        await self._update(self.done_text(answer))

    async def fail(self, reason: str) -> None:
        await self._update(self.fail_text(reason))


@dataclass
class RunningRequest:
    """Чей запрос сейчас держит блокировку (ключ — project_slug shared-воркспейса или user_id)."""
    author: str
    author_id: int | None
    text: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def snippet(self) -> str:
        return request_snippet(self.text)

    def elapsed(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(timezone.utc)) - self.started_at).total_seconds()


_running: dict[Any, RunningRequest] = {}


def mark_running(key: Any, author: str, author_id: int | None, text: str | None) -> RunningRequest:
    rr = _running[key] = RunningRequest(author, author_id, (text or "").strip())
    return rr


def clear_running(key: Any, rr: RunningRequest | None = None) -> None:
    """Снять отметку; если передан rr — только если это всё ещё он (а не следующий запрос)."""
    if rr is None or _running.get(key) is rr:
        _running.pop(key, None)


def running_request(key: Any) -> RunningRequest | None:
    return _running.get(key)


def queue_wait_text(rr: RunningRequest | None, user_id: int | None) -> str:
    """Текст «принял, жди»: называет, чей запрос выполняется и о чём он."""
    if rr is None:
        return "⏳ Принял, отвечу сразу после текущего запроса."
    what = f"(«{rr.snippet}»)" if rr.snippet else "(вложение)"
    if rr.author_id is not None and rr.author_id == user_id:
        return f"⏳ Сейчас выполняется твой предыдущий запрос {what}, этот следующий в очереди."
    return f"⏳ Сейчас выполняется запрос {rr.author} {what}, твой следующий в очереди."


_locks: dict[Any, asyncio.Lock] = {}


def lock_for(key: Any) -> asyncio.Lock:
    """Блокировка на ключ: project_slug совместного воркспейса или user_id личного режима."""
    lk = _locks.get(key)
    if lk is None:
        lk = _locks[key] = asyncio.Lock()
    return lk


def is_busy(key: Any) -> bool:
    lk = _locks.get(key)
    return bool(lk and lk.locked())


async def serialized_turn(key: Any, *, author: str, author_id: int | None, text: str | None,
                          body: Callable[[], Awaitable[T]],
                          on_queued: Callable[[RunningRequest | None], Awaitable[None]] | None = None) -> T:
    """Выполнить body под блокировкой key — только вызов агента и запись ответа; всё
    долгое (скачивание и расшифровка медиа) делается до вызова и партнёра не держит.
    Если ключ занят, сначала on_queued(кто сейчас выполняется), затем ждём очереди."""
    lk = lock_for(key)
    if lk.locked() and on_queued is not None:
        await on_queued(running_request(key))
    async with lk:
        rr = mark_running(key, author, author_id, text)
        try:
            return await body()
        finally:
            clear_running(key, rr)
