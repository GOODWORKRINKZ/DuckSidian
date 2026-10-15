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
from typing import Any

from ..personal import shared_slug_of

log = logging.getLogger(__name__)

TYPING_PERIOD = 4.5     # Telegram показывает chat action ~5 с


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
