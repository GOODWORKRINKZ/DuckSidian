"""Хэндлер слушающий ВСЕ сообщения целевой группы и записывающий их в SQLite.

Также ловит ответы (reply / callback) на вопросы агента и доставляет их
в Orchestrator._waiters.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.types import CallbackQuery, Message

from ..config import settings
from ..db import DB
from ..orchestrator import Orchestrator

log = logging.getLogger(__name__)

router = Router(name="listener")


def setup(db: DB, orch: Orchestrator) -> Router:
    target_chat = settings.telegram_chat_id

    @router.callback_query(F.data.startswith("ans:"))
    async def on_callback(cb: CallbackQuery) -> None:
        try:
            _, qid_s, idx_s = (cb.data or "").split(":", 2)
            qid = int(qid_s)
            idx = int(idx_s)
        except ValueError:
            await cb.answer("bad payload")
            return
        row = await db.get_question(qid)
        if not row:
            await cb.answer("вопрос не найден")
            return
        opts_raw = row["options"]
        options = json.loads(opts_raw) if opts_raw else []
        if idx >= len(options):
            await cb.answer("bad option")
            return
        chosen = options[idx]
        delivered = await orch.deliver_answer(qid, chosen)
        await cb.answer("принято" if delivered else "уже отвечено")
        if cb.message:
            try:
                await cb.message.edit_text(
                    (cb.message.text or "") + f"\n\n✅ *Ответ:* {chosen}",
                    parse_mode="Markdown",
                )
            except Exception:  # noqa: BLE001
                pass

    @router.message()
    async def on_any_message(msg: Message) -> None:
        if msg.chat.id != target_chat:
            return

        # Reply на вопрос агента в служебном топике → доставка.
        if (
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.is_bot
            and msg.text
        ):
            replied_text = msg.reply_to_message.text or ""
            if "id=" in replied_text:
                try:
                    qid = int(
                        replied_text.split("id=", 1)[1].split(".", 1)[0].strip()
                    )
                except ValueError:
                    qid = -1
                if qid > 0:
                    delivered = await orch.deliver_answer(qid, msg.text)
                    if delivered:
                        await msg.reply("✅ ответ передан агенту")
                        return

        # Команды обрабатываются commands-роутером — не пишем их в архив.
        if msg.text and msg.text.startswith("/"):
            return

        # Сообщения из служебного forum-топика бота не архивируем.
        if (
            settings.telegram_topic_id is not None
            and msg.message_thread_id == settings.telegram_topic_id
        ):
            return

        ts = msg.date or datetime.now(timezone.utc)
        user = msg.from_user
        await db.save_message(
            chat_id=msg.chat.id,
            message_id=msg.message_id,
            topic_id=msg.message_thread_id,
            user_id=user.id if user else None,
            username=user.username if user else None,
            full_name=user.full_name if user else None,
            text=msg.text or msg.caption,
            reply_to_message_id=(
                msg.reply_to_message.message_id if msg.reply_to_message else None
            ),
            ts=ts,
        )
