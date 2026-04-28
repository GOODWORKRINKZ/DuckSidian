"""Управление служебным форум-топиком бота (один топик на чат).

Бот пишет в свой топик "🤖 DuckSidian", не засоряя General.
"""
from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from .db import DB

log = logging.getLogger(__name__)

_BOT_TOPIC_NAME = "🤖 DuckSidian"
_BOT_TOPIC_ICON_COLOR = 0x6FB9F0  # голубой


async def ensure_bot_topic(bot: Bot, db: DB, chat_id: int) -> int | None:
    """Вернуть message_thread_id бот-топика, создав его при необходимости.

    Работает только в forum-супергруппах. В обычных группах/каналах
    возвращает None и бот пишет в основной чат.
    """
    state_key = f"bot_topic_{chat_id}"
    stored = await db.get_state(state_key)
    if stored:
        tid = int(stored)
        log.debug("bot topic for chat %s: thread_id=%s (cached)", chat_id, tid)
        return tid

    try:
        result = await bot.create_forum_topic(
            chat_id=chat_id,
            name=_BOT_TOPIC_NAME,
            icon_color=_BOT_TOPIC_ICON_COLOR,
        )
        topic_id = result.message_thread_id
        await db.set_state(state_key, str(topic_id))
        log.info("created bot topic thread_id=%s in chat %s", topic_id, chat_id)
        return topic_id
    except TelegramAPIError as exc:
        err = str(exc).lower()
        if "not a forum" in err or "supergroup" in err:
            # Не форум — кэшируем чтобы не ломиться каждый раз
            await db.set_state(state_key, "0")
        log.warning("can't create forum topic in chat %s: %s", chat_id, exc)
        return None


async def reset_bot_topic(db: DB, chat_id: int) -> None:
    """Сбросить кэш топика (для пересоздания)."""
    await db.set_state(f"bot_topic_{chat_id}", "")
