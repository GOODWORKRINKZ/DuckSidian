"""Хэндлер слушающий ВСЕ сообщения отслеживаемых групп и записывающий их в SQLite.

Поддерживает несколько чатов (ChatConfig). Медиафайлы (фото, видео,
документы, аудио, голосовые) скачиваются в raw/assets/<chat>/<date>/ и
сохраняются в БД.

Также ловит ответы (reply / callback) на вопросы агента и доставляет их
в Orchestrator._waiters.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, Message

from ..config import ChatConfig, settings
from ..db import DB
from ..media_parser import AUDIO_EXTENSIONS, extract_text_from_file
from ..orchestrator import Orchestrator
from ..topic_manager import ensure_bot_topic

log = logging.getLogger(__name__)

router = Router(name="listener")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


async def _download_media(
    bot: Bot, msg: Message, vault_root: Path, chat_name: str
) -> tuple[str | None, str | None]:
    """Определить тип медиа и скачать файл.

    Возвращает (media_type, rel_path относительно chat wiki root) или (None, None).
    Медиа сохраняется в vault/<chat_name>/raw/assets/<date>/.
    """
    date = (msg.date or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    # Путь внутри chat wiki root (vault/<chat_name>/)
    chat_wiki_root = vault_root / chat_name
    assets_dir = chat_wiki_root / "raw" / "assets" / date
    assets_dir.mkdir(parents=True, exist_ok=True)

    file_id: str | None = None
    media_type: str | None = None
    filename: str | None = None

    if msg.photo:
        ph = msg.photo[-1]
        file_id = ph.file_id
        media_type = "photo"
        filename = f"photo_{msg.message_id}.jpg"
    elif msg.video:
        file_id = msg.video.file_id
        media_type = "video"
        ext = (msg.video.mime_type or "video/mp4").split("/")[-1]
        filename = msg.video.file_name or f"video_{msg.message_id}.{ext}"
    elif msg.document:
        file_id = msg.document.file_id
        media_type = "document"
        filename = msg.document.file_name or f"doc_{msg.message_id}.bin"
    elif msg.audio:
        file_id = msg.audio.file_id
        media_type = "audio"
        filename = msg.audio.file_name or f"audio_{msg.message_id}.mp3"
    elif msg.voice:
        file_id = msg.voice.file_id
        media_type = "voice"
        filename = f"voice_{msg.message_id}.ogg"
    elif msg.video_note:
        file_id = msg.video_note.file_id
        media_type = "video_note"
        filename = f"vidnote_{msg.message_id}.mp4"
    elif msg.sticker:
        file_id = msg.sticker.file_id
        media_type = "sticker"
        filename = f"sticker_{msg.message_id}.webp"
    elif msg.animation:
        file_id = msg.animation.file_id
        media_type = "animation"
        filename = msg.animation.file_name or f"anim_{msg.message_id}.gif"

    if not file_id or not filename:
        return media_type, None

    # Санируем имя файла
    safe_filename = "".join(
        c for c in filename
        if c.isalnum() or c in "._-"
    ) or f"file_{msg.message_id}.bin"

    dst = assets_dir / safe_filename
    try:
        tg_file = await bot.get_file(file_id)
        await bot.download_file(tg_file.file_path, destination=str(dst))  # type: ignore[arg-type]
        # rel — относительно chat wiki root (vault/<chat_name>/)
        rel = str(dst.relative_to(chat_wiki_root)).replace("\\", "/")
        return media_type, rel
    except Exception as exc:  # noqa: BLE001
        log.warning("media download failed: %s: %r", type(exc).__name__, exc, exc_info=True)
        return media_type, None


def setup(db: DB, orch: Orchestrator, bot: Bot) -> Router:
    chat_ids: set[int] = {cfg.chat_id for cfg in settings.get_chats()}
    chat_name_by_id: dict[int, str] = {
        cfg.chat_id: cfg.name for cfg in settings.get_chats()
    }

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
                from html import escape as _esc
                await cb.message.edit_text(
                    _esc(cb.message.text or "") + f"\n\n✅ <b>Ответ:</b> {_esc(chosen)}",
                    parse_mode="HTML",
                )
            except Exception:  # noqa: BLE001
                pass

    @router.message()
    async def on_any_message(msg: Message) -> None:
        # DEBUG: всегда логируем входящие чтобы можно было найти chat_id
        log.info(
            "MSG chat_id=%s title=%r thread=%s user=%s text=%r",
            msg.chat.id, msg.chat.title, msg.message_thread_id,
            msg.from_user.username if msg.from_user else None,
            (msg.text or msg.caption or "")[:60],
        )
        if msg.chat.id not in chat_ids:
            return

        chat_name = chat_name_by_id[msg.chat.id]

        bot_topic_id = await ensure_bot_topic(bot, db, msg.chat.id)

        # Reply на сообщение бота с "id=N" — это ответ на вопрос агента.
        # Проверяем ДО фильтра бот-топика: вопросы агента уходят именно в
        # бот-топик, поэтому ответы там тоже должны ловиться.
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

        # Прочие сообщения из бот-топика не архивируем (только ответы выше).
        if bot_topic_id and msg.message_thread_id == bot_topic_id:
            return

        # Команды обрабатываются commands-роутером
        if msg.text and msg.text.startswith("/"):
            return

        ts = msg.date or datetime.now(timezone.utc)

        # Скачиваем медиа
        media_type: str | None = None
        media_path: str | None = None
        extracted_text: str | None = None
        if not msg.text or (msg.photo or msg.video or msg.document
                            or msg.audio or msg.voice or msg.video_note
                            or msg.sticker or msg.animation):
            media_type, media_path = await _download_media(
                bot, msg, settings.vault_path, chat_name
            )
            # Авто-извлечение текста: STT для голосовых и аудио, парсинг для документов
            # Видео анализируется позже агентом через describe_media (STT + кейфреймы)
            if media_path and media_type in {"voice", "audio", "document"}:
                abs_path = settings.vault_path / chat_name / media_path
                try:
                    extracted_text = await extract_text_from_file(abs_path)
                    if extracted_text:
                        log.info(
                            "extracted text from %s (%s): %r...",
                            media_path, media_type, extracted_text[:80],
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("media parse failed for %s: %s", media_path, exc)

        user = msg.from_user
        # Если текста нет, но удалось распознать/распарсить медиа — сохраняем как text
        text_to_save = msg.text or msg.caption or extracted_text
        await db.save_message(
            chat_id=msg.chat.id,
            message_id=msg.message_id,
            topic_id=msg.message_thread_id,
            user_id=user.id if user else None,
            username=user.username if user else None,
            full_name=user.full_name if user else None,
            text=text_to_save,
            media_type=media_type,
            media_path=media_path,
            reply_to_message_id=(
                msg.reply_to_message.message_id if msg.reply_to_message else None
            ),
            ts=ts,
        )

    return router
