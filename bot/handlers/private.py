"""Личный режим (DM): персональное пространство сотрудника.

Роутер ловит ВСЕ сообщения из приватных чатов (включается в Dispatcher первым,
чтобы групповой commands-роутер со своим chat_guard их не проглатывал).

Что умеет:
- доступ по allowlist (админы + TELEGRAM_PRIVATE_USERS + /allow);
- любое сообщение (текст / голос / видео-кружок / фото / документ) →
  скачивание в личный vault, STT (MiniMax ASR → Whisper), запись в raw/inbox
  и БД, затем агент (запись мысли / поиск фактов / управление артефактами);
- воркспейсы: тема (topic) в личке с ботом = проект. Сообщения в теме
  привязываются к странице wiki/projects/<Slug>.md, история — своя на тему;
- голос: ответы голосом (/voice on|off|auto), /say — озвучить текст,
  /voices — голоса; /imagine — сгенерировать картинку;
- /ask /search /note /projects /list /show /del /share /log /space /help;
- кнопки под ответами агента: 📤 поделиться результатом с групповым чатом.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime, timezone
from html import escape as _esc
from typing import Any

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from ..config import settings
from ..db import DB
from ..intake import FILE_URL_RE, IntakeResult, download_url_to_vault, process_attachment
from ..media_ai import MediaAIError, get_media_ai
from ..media_parser import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    extract_text_from_file,
    transcribe_video,
)
from ..orchestrator import Orchestrator
from ..personal import (
    ARTIFACT_KINDS,
    Artifact,
    PersonalSpace,
    artifact_markdown_table,
    describe_space,
    slugify,
)
from ..telegraph import TelegraphClient, page_title
from ..tg_format import extract_tables, md_to_tg_html, render_table_png
from .admin_shared import (
    format_model_list,
    handle_allow_command,
    handle_deny_command,
    handle_model_command,
    handle_users_command,
    is_admin,
    is_private_allowed,
)
from .commands import _download_ask_media, _extract_image_refs, _react, _reply_html
from .listener import _download_media, handle_answer_callback, question_id_from_reply

log = logging.getLogger(__name__)

router = Router(name="private")
router.message.filter(F.chat.type == "private")
router.callback_query.filter(F.message.chat.type == "private")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
_GIF_EXTS = {".gif"}
_TG_MAX_BYTES = 50 * 1024 * 1024
_TTS_REPLY_MAX_CHARS = 1500

HELP_TEXT = (
    "<b>DuckSidian — личный режим</b>\n\n"
    "Просто пиши или говори голосом — я веду твоё личное пространство:\n"
    "• мысли, идеи, статусы по проектам → записываю в заметки/проекты;\n"
    "• «посмотри / проверь / найди / сравни …» → ищу в твоих данных, в wiki "
    "команды и в интернете, собираю подборку с источниками;\n"
    "• «покажи / удали / отправь в чат / нарисуй …» → управляю артефактами, рисую.\n\n"
    "<b>Воркспейсы</b> — создай тему (topic) в этом чате: тема = проект. "
    "Всё, что пишешь в теме, ложится в её страницу проекта. "
    "/ws — текущий воркспейс, /ws new &lt;имя&gt; — создать, /ws list — все.\n\n"
    "<b>Команды</b>\n"
    "/ask &lt;вопрос&gt; — вопрос к твоему личному vault\n"
    "/note &lt;текст&gt; — быстрая заметка без агента\n"
    "/search &lt;строка&gt; — поиск по личному vault\n"
    "/projects — твои проекты\n"
    "/list [projects|notes|ideas|research|files] — список артефактов\n"
    "/show &lt;№|имя|путь&gt; — показать артефакт\n"
    "/share &lt;№|имя|путь&gt; [чат] [комментарий] — отправить в групповой чат\n"
    "/share (реплаем на мой ответ) [чат] — поделиться ответом\n"
    "/del &lt;№|имя|путь&gt; — удалить артефакт\n"
    "/say &lt;текст&gt; (или реплаем) — озвучить голосом\n"
    "/voice on|off|auto — голосовые ответы; /voice set &lt;voice_id&gt;; /voices — список\n"
    "/imagine &lt;описание&gt; [16:9] — сгенерировать картинку\n"
    "/space — что лежит в пространстве · /log — операции · /chats — куда шарить\n"
    "/model — модель LLM <i>(admin)</i> · /allow /deny /users — доступ <i>(admin)</i>\n"
)

# ---- постоянное меню (reply keyboard)
BTN_WS, BTN_ART, BTN_PROJ, BTN_SPACE = "📌 Воркспейсы", "📂 Артефакты", "📁 Проекты", "📊 Пространство"
BTN_VOICE, BTN_HELP, BTN_ADMIN = "🎙 Голос", "❓ Помощь", "🛡 Админ"
MENU_BUTTONS = {BTN_WS, BTN_ART, BTN_PROJ, BTN_SPACE, BTN_VOICE, BTN_HELP, BTN_ADMIN}


def _menu_kb(admin: bool) -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(text=BTN_WS), KeyboardButton(text=BTN_ART)],
        [KeyboardButton(text=BTN_PROJ), KeyboardButton(text=BTN_SPACE)],
        [KeyboardButton(text=BTN_VOICE), KeyboardButton(text=BTN_HELP)],
    ]
    if admin:
        rows.append([KeyboardButton(text=BTN_ADMIN)])
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True, is_persistent=True,
                               input_field_placeholder="Пиши или говори — я запишу / найду")


# user_id → что ждём следующим сообщением ("ws_name")
_awaiting: dict[int, str] = {}

INBOX_TOPIC_NAME = "📥 Входящие"
# Имена тем, которые Telegram даёт сам при первом сообщении «из корня» чата.
_AUTO_TOPIC_NAMES = {
    "media", "медиа", "voice message", "голосовое сообщение", "photo", "фото", "video", "видео",
    "document", "файл", "документ", "sticker", "стикер", "gif", "video message", "видеосообщение",
    "audio", "аудио", "new topic", "новая тема",
}


def _looks_auto_topic(name: str, first_text: str) -> bool:
    """Тема создана Telegram автоматически (а не пользователем через «Новая тема»)?"""
    n = (name or "").strip().lower().rstrip("…. ")
    if not n or n in _AUTO_TOPIC_NAMES:
        return True
    t = (first_text or "").strip().lower()
    if t and (t.startswith(n[:20]) or n.startswith(t[:20])):
        return True
    return False
# user_id → последний показанный список артефактов (для /show 3, /share 2 …)
_last_listing: dict[int, list[Artifact]] = {}
# token → (user_id, text_markdown | "@file:<rel>", title|comment)
_pending_share: dict[str, tuple[int, str, str]] = {}
# token → (user_id, rel_path) — подтверждение удаления
_pending_delete: dict[str, tuple[int, str]] = {}
# token → (user_id, rel_path) — кнопки «📰 статьёй / 📄 файлом» под /show
_pending_view: dict[str, tuple[int, str]] = {}
# user_id → фоновая задача ingest (разбор импортированных дней в wiki)
_ingest_tasks: dict[int, asyncio.Task] = {}
_PENDING_MAX = 300
# Кому уже сказали «нет доступа» (чтобы не спамить на каждое сообщение)
_denied_notified: set[int] = set()
# Последовательная обработка сообщений одного пользователя
_user_locks: dict[int, asyncio.Lock] = {}


def _trim(d: dict) -> None:
    while len(d) > _PENDING_MAX:
        d.pop(next(iter(d)))


def _lock(user_id: int) -> asyncio.Lock:
    lk = _user_locks.get(user_id)
    if lk is None:
        lk = _user_locks[user_id] = asyncio.Lock()
    return lk


def _display_name(msg: Message) -> str:
    u = msg.from_user
    if not u:
        return "anon"
    return u.full_name or (f"@{u.username}" if u.username else str(u.id))


def _thread_id(msg: Message) -> int | None:
    return msg.message_thread_id if (msg.is_topic_message and msg.message_thread_id) else None


async def _typing_loop(bot: Bot, chat_id: int, stop: asyncio.Event,
                       thread_id: int | None = None, action: str = "typing") -> None:
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, action, message_thread_id=thread_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            pass


def _strip_images_block(answer: str) -> tuple[str, bool]:
    m = re.search(r"(?im)^\s*IMAGES\s*:\s*$", answer or "")
    if not m:
        return (answer or "").strip(), False
    return answer[: m.start()].rstrip(), True


async def _attach_media(msg: Message, answer: str, wiki_root) -> None:
    """Прикрепить файлы из блока IMAGES: ответа агента."""
    try:
        paths = _extract_image_refs(answer or "", wiki_root, max_n=5)
        paths = [p for p in paths if os.path.getsize(p) <= _TG_MAX_BYTES]
        for p in paths:
            ext = "." + p.lower().rsplit(".", 1)[-1]
            f = FSInputFile(p)
            if ext in _VID_EXTS:
                await msg.answer_video(f)
            elif ext in _GIF_EXTS:
                await msg.answer_animation(f)
            else:
                await msg.answer_photo(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("private: attach media failed: %s", exc)


def _share_keyboard(token: str) -> InlineKeyboardMarkup | None:
    chats = settings.get_chats()
    if not chats:
        return None
    row = [
        InlineKeyboardButton(text=f"📤 в {c.name}", callback_data=f"pshr:{token}:{c.name}"[:64])
        for c in chats[:4]
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row])


def _resolve_ref(space: PersonalSpace, user_id: int, ref: str) -> str | None:
    """№ из последнего списка, либо имя/путь."""
    ref = (ref or "").strip()
    if ref.isdigit():
        listing = _last_listing.get(user_id) or space.list_artifacts()
        idx = int(ref) - 1
        if 0 <= idx < len(listing):
            return listing[idx].path
        return None
    return space.resolve_artifact(ref)


def _split_share_args(args: str) -> tuple[str, str | None, str]:
    """`<ref> [chat] [comment...]` → (ref, chat|None, comment)."""
    parts = (args or "").split()
    if not parts:
        return "", None, ""
    ref = parts[0]
    chat: str | None = None
    rest = parts[1:]
    names = {c.name for c in settings.get_chats()}
    if rest and rest[0].lstrip("#") in names:
        chat = rest[0].lstrip("#")
        rest = rest[1:]
    return ref, chat, " ".join(rest)


_STEP_LABELS = {
    "search_wiki": "🔎 ищу в твоих записях",
    "search_team_wiki": "🔎 ищу в wiki команды",
    "web_search": "🌐 ищу в интернете",
    "fetch_url": "🌐 читаю страницу",
    "fetch_git_log": "🐙 смотрю коммиты",
    "read_file": "📖 читаю", "read_lines": "📖 читаю", "grep_file": "📖 читаю",
    "read_team_file": "📖 читаю в wiki команды", "list_team_dir": "📂 смотрю wiki команды",
    "list_dir": "📂 смотрю папку",
    "write_file": "✍️ пишу", "append_file": "✍️ дописываю", "edit_file": "✍️ правлю",
    "move_file": "📁 переношу", "delete_file": "🗑 удаляю",
    "describe_media": "👁 смотрю файл", "generate_image": "🎨 рисую",
    "share_to_chat": "📤 делюсь с чатом", "get_current_time": "🕐 смотрю время",
}


def _step_line(name: str, args: dict) -> str:
    label = _STEP_LABELS.get(name, f"⚙️ {name}")
    arg = ""
    for key in ("query", "path", "url", "chat", "prompt", "repo", "src"):
        v = args.get(key)
        if v:
            arg = str(v)
            break
    if name in ("search_team_wiki", "read_team_file", "list_team_dir") and args.get("chat"):
        arg = f"{args.get('chat')}: {args.get('query') or args.get('path') or ''}".strip(": ")
    arg = arg.replace("\n", " ")
    return f"{label}" + (f": {arg[:60]}" if arg else "")


def _parse_imagine_args(args: str) -> tuple[str, str]:
    """`<prompt> [16:9]` → (prompt, aspect)."""
    from ..media_ai import IMAGE_ASPECTS
    parts = (args or "").split()
    aspect = "1:1"
    if parts and parts[-1] in IMAGE_ASPECTS:
        aspect = parts.pop()
    return " ".join(parts).strip(), aspect


def setup(db: DB, orch: Orchestrator, bot: Bot) -> Router:

    # ------------------------------------------------------------- helpers

    async def _access_requests() -> list[dict]:
        try:
            return json.loads(await db.get_state("access_requests", "[]"))
        except Exception:  # noqa: BLE001
            return []

    async def _set_access_requests(items: list[dict]) -> None:
        await db.set_state("access_requests", json.dumps(items[-50:], ensure_ascii=False))

    async def _register_access_request(msg: Message) -> None:
        """Запомнить запрос доступа и разослать админам карточку с кнопками ✅/❌."""
        u = msg.from_user
        if not u:
            return
        items = await _access_requests()
        if any(int(i.get("id", 0)) == u.id for i in items):
            return
        items.append({"id": u.id, "name": u.full_name, "username": u.username or "",
                      "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")})
        await _set_access_requests(items)
        card = (
            f"🔔 <b>Запрос доступа к личному режиму</b>\n"
            f"{_esc(u.full_name)}" + (f" (@{_esc(u.username)})" if u.username else "") +
            f" · id <code>{u.id}</code>\n"
            f"Сообщение: <i>{_esc((msg.text or msg.caption or '(медиа)')[:200])}</i>"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✅ Разрешить", callback_data=f"pacc:{u.id}:y"),
            InlineKeyboardButton(text="❌ Отклонить", callback_data=f"pacc:{u.id}:n"),
        ]])
        for admin_id in settings.telegram_admins:
            try:
                await bot.send_message(admin_id, card, parse_mode="HTML", reply_markup=kb)
            except Exception as exc:  # noqa: BLE001
                log.info("access request: can't notify admin %s: %s", admin_id, exc)

    async def _guard(msg: Message) -> PersonalSpace | None:
        uid = msg.from_user.id if msg.from_user else None
        if msg.from_user and msg.from_user.is_bot:
            return None
        if not await is_private_allowed(db, uid):
            if uid not in _denied_notified:
                _denied_notified.add(uid or 0)
                await msg.reply(
                    "⛔ Личный режим пока не включён для тебя.\n"
                    f"Твой id: <code>{uid}</code>. Я отправил запрос админу — как только он "
                    "подтвердит, я напишу.",
                    parse_mode="HTML",
                )
                await _register_access_request(msg)
            return None
        return orch.personal_space(uid, _display_name(msg))  # type: ignore[arg-type]

    async def _ensure_inbox_topic(chat_id: int) -> int | None:
        """Тема «📥 Входящие» для всего, что не по проекту: иначе каждое сообщение из
        корня чата открывает новую авто-тему. Создаётся один раз на пользователя."""
        key = f"inbox_topic_{chat_id}"
        stored = await db.get_state(key, "")
        if stored:
            return int(stored) if stored.isdigit() else None
        try:
            topic = await bot.create_forum_topic(chat_id=chat_id, name=INBOX_TOPIC_NAME)
        except Exception as exc:  # noqa: BLE001
            log.info("inbox topic not created for %s: %s", chat_id, exc)
            await db.set_state(key, "0")
            return None
        tid = topic.message_thread_id
        await db.set_state(key, str(tid))
        await db.upsert_workspace(chat_id, tid, INBOX_TOPIC_NAME, None)
        try:
            await bot.send_message(
                chat_id,
                "📥 Это <b>Входящие</b> — пиши сюда всё, что не относится к конкретному проекту: "
                "мысли, вопросы, «посмотри/проверь…». Для проекта создай тему (/ws new &lt;имя&gt; "
                "или попроси меня) — тогда всё в ней ляжет в его страницу.\n"
                "Сообщения, отправленные из списка тем, Telegram сам оборачивает в новую тему — "
                "лучше писать внутри Входящих или темы проекта.",
                message_thread_id=tid, parse_mode="HTML",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("inbox greet failed: %s", exc)
        return tid

    async def _offer_archive_ingest(msg: Message, space: PersonalSpace, intake: IntakeResult) -> None:
        info = intake.archive
        if info is None:
            return
        token = uuid.uuid4().hex[:10]
        _pending_share[token] = (space.user_id, "@ingest", "")
        _trim(_pending_share)
        rows = [[InlineKeyboardButton(text=f"🧠 Разобрать в wiki ({len(info.dates)} дн.)", callback_data=f"ping:{token}")]]
        if is_admin(space.user_id):
            for c in settings.get_chats()[:3]:
                rows.append([InlineKeyboardButton(text=f"📤 Импортировать в чат {c.name}", callback_data=f"pimp:{token}:{c.name}")])
        _pending_view[token] = (space.user_id, intake.rel)
        await msg.reply(md_to_tg_html(info.summary_md()) + "\n\nЧто делаем дальше?",
                        parse_mode="HTML", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))

    async def _run_ingest_bg(chat_id: int, thread_id: int | None, space: PersonalSpace) -> None:
        """Фоновый разбор всех необработанных raw/daily/*.md личного vault (INGEST-агент)."""
        done = 0
        try:
            while True:
                out = await orch.ingest_raw_pending(space.cfg, limit=3)
                if out.startswith("Все raw/daily") or out.startswith("Нет raw/daily"):
                    break
                done += 3
                if done % 6 == 0:
                    try:
                        await bot.send_message(chat_id, f"⏳ Разобрано ~{done} дней, продолжаю… (/stop — остановить)",
                                               message_thread_id=thread_id)
                    except Exception:  # noqa: BLE001
                        pass
                if "Ещё" not in out:
                    break
            await bot.send_message(chat_id, "✅ Импорт разобран: знания разложены по wiki/. "
                                   "Спроси меня о чём-нибудь из переписки или смотри /list.",
                                   message_thread_id=thread_id)
        except asyncio.CancelledError:
            await bot.send_message(chat_id, "⏹ Разбор остановлен.", message_thread_id=thread_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("bg ingest failed")
            await bot.send_message(chat_id, f"⚠️ Разбор упал: {type(exc).__name__}: {exc}",
                                   message_thread_id=thread_id)
        finally:
            _ingest_tasks.pop(space.user_id, None)

    @router.callback_query(F.data.startswith("ping:"))
    async def cb_ingest_imported(cb: CallbackQuery) -> None:
        token = (cb.data or "").split(":", 1)[1]
        entry = _pending_view.get(token)
        if not entry or not cb.from_user or cb.from_user.id != entry[0]:
            await cb.answer("⌛ устарело")
            return
        space = orch.personal_space(entry[0])
        if space.user_id in _ingest_tasks and not _ingest_tasks[space.user_id].done():
            await cb.answer("уже идёт разбор — /stop чтобы остановить", show_alert=True)
            return
        await cb.answer("запускаю разбор в фоне")
        tid = cb.message.message_thread_id if cb.message and cb.message.is_topic_message else None
        _ingest_tasks[space.user_id] = asyncio.create_task(_run_ingest_bg(cb.message.chat.id, tid, space))  # type: ignore[union-attr]
        try:
            await cb.message.edit_reply_markup(reply_markup=None)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        await cb.message.answer("🧠 Разбираю импорт в wiki в фоне — это небыстро (LLM по каждому дню). "  # type: ignore[union-attr]
                                "Напишу, когда закончу. /stop — остановить.")

    @router.callback_query(F.data.startswith("pimp:"))
    async def cb_import_to_chat(cb: CallbackQuery) -> None:
        parts = (cb.data or "").split(":", 2)
        if len(parts) != 3:
            await cb.answer("bad data")
            return
        token, chat_name = parts[1], parts[2]
        entry = _pending_view.get(token)
        if not entry or not cb.from_user or cb.from_user.id != entry[0] or not is_admin(cb.from_user.id):
            await cb.answer("⌛ устарело / нет прав")
            return
        target = settings.get_chat_by_name(chat_name)
        if target is None:
            await cb.answer("чат не найден")
            return
        space = orch.personal_space(entry[0])
        await cb.answer("импортирую…")
        try:
            from ..orchestrator import _chat_wiki  # noqa: PLC0415
            from ..archive_import import import_archive  # noqa: PLC0415
            import shutil  # noqa: PLC0415
            twiki = _chat_wiki(target)
            # копируем архив в vault чата и импортируем туда же
            src = space.root / entry[1]
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            dst_rel = f"raw/assets/shared/{date}/u{space.user_id}_{src.name}"
            (twiki.root / dst_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, twiki.root / dst_rel)
            info = import_archive(twiki, dst_rel)
            text = md_to_tg_html(info.summary_md()) + (
                "\n\nДни попадут в wiki чата при ближайшем авто-ingest (каждые 20 минут) или по /ingestfiles в группе."
                if info.dates else "")
        except Exception as exc:  # noqa: BLE001
            log.exception("import to chat failed")
            text = f"❌ {_esc(str(exc))}"
        if cb.message:
            await cb.message.answer(text, parse_mode="HTML")

    @router.message(Command("stop"))
    async def cmd_stop(msg: Message) -> None:
        uid = msg.from_user.id if msg.from_user else 0
        task = _ingest_tasks.get(uid)
        if task and not task.done():
            task.cancel()
            await msg.reply("⏹ Останавливаю разбор…")
        else:
            await msg.reply("Фоновых задач нет.")

    async def _create_workspace(space: PersonalSpace, chat_id: int, name: str) -> tuple[bool, str, int | None]:
        """Создать тему в личке + проект. Возвращает (ok, текст, thread_id)."""
        name = (name or "").strip()[:128]
        if not name:
            return False, "пустое имя", None
        try:
            topic = await bot.create_forum_topic(chat_id=chat_id, name=name)
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"не удалось создать тему: {exc}. Включи у бота в @BotFather режим тем для "
                "личных чатов (Bot Settings → Threaded mode / Topics)."
            ), None
        slug = space.ensure_project(name)
        await db.upsert_workspace(chat_id, topic.message_thread_id, name, slug)
        try:
            await bot.send_message(
                chat_id,
                f"📌 Воркспейс <b>{_esc(name)}</b> → <code>wiki/projects/{_esc(slug)}.md</code>.\n"
                "Пиши сюда всё по этому проекту — я буду вести его страницу.",
                message_thread_id=topic.message_thread_id, parse_mode="HTML",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("ws greet failed: %s", exc)
        return True, f"тема «{name}» создана, проект wiki/projects/{slug}.md", topic.message_thread_id

    def _workspace_tool_fn(msg: Message, space: PersonalSpace):
        """Коллбек для агентских инструментов create_workspace / list_workspaces / rename_workspace."""
        async def fn(action: str, args: dict) -> str:
            if action == "create_workspace":
                ok, text, _tid = await _create_workspace(space, msg.chat.id, str(args.get("name", "")))
                return ("OK: " if ok else "ERROR: ") + text
            if action == "list_workspaces":
                rows = await db.list_workspaces(msg.chat.id)
                if not rows:
                    return "Воркспейсов нет."
                return "\n".join(f"- {r['name']} → wiki/projects/{r['project_slug']}.md" for r in rows)
            if action == "rename_workspace":
                tid = _thread_id(msg)
                if tid is None:
                    return "ERROR: разговор идёт не в теме — переименовывать нечего"
                new_name = str(args.get("name", "")).strip()[:128]
                if not new_name:
                    return "ERROR: пустое имя"
                slug = await _bind_workspace(space, msg.chat.id, tid, new_name)
                return f"OK: воркспейс → «{new_name}» (wiki/projects/{slug}.md)"
            return f"ERROR: unknown action {action}"
        return fn

    def _bind_kb(tid: int, name: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=f"📌 Да, воркспейс «{name[:24]}»", callback_data=f"pwsbind:{tid}")],
            [InlineKeyboardButton(text="✏️ Другое имя", callback_data=f"pwsname:{tid}"),
             InlineKeyboardButton(text="💬 Нет, просто чат", callback_data=f"pwsno:{tid}")],
        ])

    async def _topic_name(msg: Message, tid: int) -> str:
        r = msg.reply_to_message
        if r is not None and r.forum_topic_created and r.forum_topic_created.name:
            return r.forum_topic_created.name
        row = await db.get_workspace(msg.chat.id, tid)
        if row is not None and row["name"]:
            return row["name"]
        return f"тема {tid}"

    async def _workspace(msg: Message, space: PersonalSpace,
                         name_hint: str | None = None) -> dict[str, Any] | None:
        """Воркспейс для сообщения в теме.

        Тема становится воркспейсом ТОЛЬКО явно: кнопкой «📌 Сделать воркспейсом»,
        через /ws new, /ws <имя> или инструмент агента. Telegram сам создаёт темы при
        первом сообщении и называет их по содержимому («Медиа», первые слова текста) —
        такие темы остаются обычным чатом, пока пользователь не решит иначе.
        Для незнакомой темы один раз показываем подсказку с кнопками.
        """
        tid = _thread_id(msg)
        if tid is None:
            return None
        row = await db.get_workspace(msg.chat.id, tid)
        if row is not None:
            return dict(row) if row["project_slug"] else None
        name = (name_hint or await _topic_name(msg, tid)).strip()[:128]
        await db.upsert_workspace(msg.chat.id, tid, name, None)  # известна, не привязана
        if _looks_auto_topic(name, msg.text or msg.caption or ""):
            return None  # Telegram сам открыл тему по первому сообщению — просто чат
        try:
            await msg.answer(
                f"🆕 Новая тема «{_esc(name)}». Сделать её воркспейсом? Тогда всё в ней "
                "будет относиться к проекту с таким именем.",
                parse_mode="HTML", reply_markup=_bind_kb(tid, name),
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("bind hint failed: %s", exc)
        return None

    async def _bind_workspace(space: PersonalSpace, chat_id: int, tid: int, name: str) -> str:
        """Привязать тему к проекту (создать страницу при необходимости). Возвращает slug."""
        name = (name or "").strip()[:128] or f"topic-{tid}"
        slug = space.ensure_project(name)
        await db.upsert_workspace(chat_id, tid, name, slug)
        async with db.conn() as conn:
            await conn.execute("UPDATE workspaces SET project_slug=? WHERE chat_id=? AND topic_id=?",
                               (slug, chat_id, tid))
            await conn.commit()
        try:
            await bot.edit_forum_topic(chat_id=chat_id, message_thread_id=tid, name=name)
        except Exception as exc:  # noqa: BLE001
            log.debug("edit_forum_topic failed: %s", exc)
        return slug

    async def _voice_mode(user_id: int) -> str:
        return (await db.get_state(f"voice_mode_{user_id}", settings.personal_voice_reply or "auto")).lower()

    async def _voice_id(user_id: int) -> str:
        return await db.get_state(f"voice_id_{user_id}", settings.minimax_tts_voice)

    async def _send_table_images(msg: Message, markdown: str) -> None:
        """Широкие markdown-таблицы (3+ колонок) дополнительно шлём картинкой:
        в тексте они превращаются в карточки, а PNG даёт привычный табличный вид."""
        try:
            tables = extract_tables(markdown, min_cols=3)
        except Exception:  # noqa: BLE001
            return
        for i, rows in enumerate(tables[:3], 1):
            try:
                png = await asyncio.get_running_loop().run_in_executor(None, render_table_png, rows)
            except Exception as exc:  # noqa: BLE001
                log.debug("table png failed: %s", exc)
                png = None
            if not png:
                return
            try:
                from aiogram.types import BufferedInputFile  # noqa: PLC0415
                cap = f"📊 таблица {i}" if len(tables) > 1 else "📊 таблица"
                await msg.answer_photo(BufferedInputFile(png, filename=f"table_{i}.png"), caption=cap)
            except Exception as exc:  # noqa: BLE001
                log.warning("send table png failed: %s", exc)

    async def _send_voice(msg: Message, space: PersonalSpace, text: str, *, summarize: bool = False) -> bool:
        """Озвучить текст и отправить голосовым. summarize=True — сначала сжать в короткий
        устный пересказ сути (полный ответ уже ушёл текстом). True если получилось."""
        ai = get_media_ai()
        if not ai.tts_enabled:
            return False
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dst = space.root / ".cache" / "tts" / f"tts_{ts}_{msg.message_id}.ogg"
        stop = asyncio.Event()
        typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop, _thread_id(msg), "record_voice"))
        try:
            speech = await orch.summarize_for_voice(text) if summarize else text[:_TTS_REPLY_MAX_CHARS]
            if not speech.strip():
                return False
            path = await ai.tts_to_file(speech, dst, voice_id=await _voice_id(space.user_id))
            if path.suffix == ".ogg":
                await msg.answer_voice(FSInputFile(path))
            else:
                await msg.answer_audio(FSInputFile(path))
            return True
        except MediaAIError as exc:
            log.warning("tts failed: %s", exc)
            await msg.reply(f"⚠️ Не удалось озвучить: {_esc(str(exc))}", parse_mode="HTML")
        except Exception as exc:  # noqa: BLE001
            log.warning("tts send failed: %s", exc)
        finally:
            stop.set()
            typing.cancel()
        return False

    async def _save_bot_reply(space: PersonalSpace, msg: Message, text: str,
                              sent_ids: list[int], ws_name: str | None) -> None:
        now = datetime.now(timezone.utc)
        mid = sent_ids[0] if sent_ids else -int(now.timestamp() * 1000)
        try:
            await db.save_message(
                chat_id=msg.chat.id, message_id=mid, topic_id=_thread_id(msg),
                user_id=bot.id, username="bot", full_name="DuckSidian",
                text=text, media_type=None, media_path=None,
                reply_to_message_id=None, ts=now,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("private: save bot reply failed: %s", exc)
        who = "🤖 DuckSidian" + (f" · 📌 {ws_name}" if ws_name else "")
        space.append_inbox(who=who, text=text, ts=now, message_id=mid)

    def _draft_progress(msg: Message):
        """Стриминг прогресса агента черновиком (Bot API sendMessageDraft, только личка)."""
        draft_id = random.randint(1, 2**31 - 1)
        steps: list[str] = []
        state = {"ok": True}

        async def progress(name: str, args: dict) -> None:
            if not state["ok"]:
                return
            steps.append(_step_line(name, args))
            text = "⏳ " + "\n".join(steps[-6:])
            try:
                await bot.send_message_draft(
                    msg.chat.id, draft_id, message_thread_id=_thread_id(msg), text=text[:3500],
                )
            except Exception as exc:  # noqa: BLE001
                state["ok"] = False
                log.debug("send_message_draft unavailable: %s", exc)
        return progress

    async def _history(msg: Message) -> list[dict[str, str]]:
        rows = await db.fetch_last_messages_in_topic(
            msg.chat.id, _thread_id(msg), settings.personal_history_size + 1,
        )
        out: list[dict[str, str]] = []
        for r in rows:
            if r["message_id"] == msg.message_id:
                continue
            text = (r["text"] or "").strip()
            if not text:
                continue
            role = "assistant" if r["user_id"] == bot.id else "user"
            out.append({"role": role, "content": text[:2000]})
        return out[-settings.personal_history_size:]

    async def _send_agent_answer(msg: Message, space: PersonalSpace, answer: str, *,
                                 title_hint: str, ws_name: str | None = None,
                                 voice: bool = False) -> None:
        display, has_images = _strip_images_block(answer)
        sent_ids = await _reply_html(msg, display or "(пусто)")
        if has_images:
            await _attach_media(msg, answer, space.wiki.root)
        await _send_table_images(msg, display)
        await _save_bot_reply(space, msg, display, sent_ids, ws_name)
        if voice and display:
            await _send_voice(msg, space, display, summarize=True)
        # Кнопка «поделиться ответом» — только для содержательных ответов.
        if len(display) >= 200 and settings.get_chats():
            token = uuid.uuid4().hex[:10]
            _pending_share[token] = (space.user_id, display, title_hint[:80])
            _trim(_pending_share)
            kb = _share_keyboard(token)
            if kb:
                try:
                    await msg.answer("Поделиться этим ответом с командой?", reply_markup=kb)
                except Exception as exc:  # noqa: BLE001
                    log.warning("private: share-button failed: %s", exc)

    # ------------------------------------------------------------- commands

    @router.message(Command("start", "help"))
    async def cmd_help(msg: Message) -> None:
        uid = msg.from_user.id if msg.from_user else None
        if not await is_private_allowed(db, uid):
            await msg.reply(
                "👋 Это DuckSidian. Личный режим доступен по списку.\n"
                f"Твой id: <code>{uid}</code> — отправь его админу, он выполнит "
                f"<code>/allow {uid}</code>.",
                parse_mode="HTML",
            )
            return
        orch.personal_space(uid, _display_name(msg))  # type: ignore[arg-type]
        await msg.reply(HELP_TEXT, parse_mode="HTML", reply_markup=_menu_kb(is_admin(uid)))
        await _ensure_inbox_topic(msg.chat.id)

    @router.message(Command("menu"))
    async def cmd_menu(msg: Message) -> None:
        space = await _guard(msg)
        if not space:
            return
        await msg.reply("Меню внизу 👇", reply_markup=_menu_kb(is_admin(space.user_id)))

    @router.message(Command("model"))
    async def cmd_model(msg: Message, command: CommandObject) -> None:
        await handle_model_command(msg, command.args or "", orch)

    @router.message(Command("allow"))
    async def cmd_allow(msg: Message, command: CommandObject) -> None:
        await handle_allow_command(msg, command.args or "", db)

    @router.message(Command("deny"))
    async def cmd_deny(msg: Message, command: CommandObject) -> None:
        await handle_deny_command(msg, command.args or "", db)

    @router.message(Command("users"))
    async def cmd_users(msg: Message) -> None:
        await handle_users_command(msg, db)

    @router.message(Command("chats"))
    async def cmd_chats(msg: Message) -> None:
        if not await _guard(msg):
            return
        lines = ["<b>Групповые чаты (куда можно /share):</b>"]
        for c in settings.get_chats():
            lines.append(f"• <code>{c.name}</code>")
        await msg.reply("\n".join(lines), parse_mode="HTML")

    @router.message(Command("space"))
    async def cmd_space(msg: Message) -> None:
        space = await _guard(msg)
        if not space:
            return
        info = describe_space(space)
        c = info["counts"]
        st = get_media_ai().status()
        ws = await db.list_workspaces(msg.chat.id)
        text = (
            f"<b>Твоё пространство</b> <code>users/{space.user_id}/</code>\n"
            f"📁 проекты: {c['projects']}  📝 заметки: {c['notes']}  💡 идеи: {c['ideas']}\n"
            f"🔎 подборки: {c['research']}  ❓ queries: {c['queries']}  📎 файлы: {c['files']}\n"
            f"📌 воркспейсов (тем): {len(ws)}\n"
            f"🧠 модель: {_esc(orch.model_info())}\n"
            f"🎙 STT: {'MiniMax' if st['stt'] else 'Whisper'} · "
            f"👁 vision: {'MiniMax ' + st['vision_model'] if st['vision'] else 'ollama/нет'} · "
            f"🔊 TTS: {('MiniMax ' + st['tts_model']) if st['tts'] else 'выкл'} · "
            f"🎨 image: {st['image_model'] if st['image'] else 'выкл'}\n"
            f"🗣 голосовые ответы: {await _voice_mode(space.user_id)}, голос: "
            f"<code>{_esc(await _voice_id(space.user_id))}</code>"
        )
        await msg.reply(text, parse_mode="HTML")

    @router.message(Command("log"))
    async def cmd_log(msg: Message) -> None:
        space = await _guard(msg)
        if not space:
            return
        try:
            content = space.wiki.read_file("log.md")
        except Exception as exc:  # noqa: BLE001
            await msg.reply(f"log.md недоступен: {exc}")
            return
        lines = [ln for ln in content.splitlines() if ln.startswith("## [")]
        await msg.reply(f"<pre>{_esc(chr(10).join(lines[-15:]) or '(пусто)')}</pre>",
                        parse_mode="HTML")

    @router.message(Command("projects"))
    async def cmd_projects(msg: Message) -> None:
        space = await _guard(msg)
        if not space:
            return
        items = orch.list_projects(space.cfg)
        if not items:
            await msg.reply("Проектов пока нет. Напиши про проект или создай тему — я заведу страницу.")
            return
        emoji = {"active": "🟢", "paused": "🟡", "done": "✅", "archive": "📁"}
        lines = ["<b>Твои проекты:</b>"]
        for it in items:
            upd = f" <i>{it['updated']}</i>" if it["updated"] else ""
            lines.append(f"{emoji.get(it['status'], '⚪️')} <b>{_esc(it['name'])}</b>{upd}\n"
                         f" {_esc(it['summary'])}")
        await msg.reply("\n\n".join(lines)[:4000], parse_mode="HTML")

    # ---- workspaces (темы)

    @router.message(Command("ws", "workspace"))
    async def cmd_ws(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        args = (command.args or "").strip()
        sub, _, rest = args.partition(" ")
        sub = sub.lower()
        if sub == "share":
            tid = _thread_id(msg)
            if tid is None:
                await msg.reply("Команда /ws share работает внутри темы-воркспейса.")
                return
            ws = await _workspace(msg, space)
            if not ws:
                await msg.reply("Эта тема не привязана к проекту — сначала /ws <имя>.")
                return
            parts = rest.split()
            chats = settings.get_chats()
            chat = None
            as_topic = False
            comment_parts: list[str] = []
            for p in parts:
                if p.lstrip("#") in {c.name for c in chats} and chat is None:
                    chat = p.lstrip("#")
                elif p.lower() in ("topic", "тема", "--topic"):
                    as_topic = True
                else:
                    comment_parts.append(p)
            comment = " ".join(comment_parts)
            if chat is None and len(chats) == 1:
                chat = chats[0].name
            if chat is None:
                token = uuid.uuid4().hex[:10]
                _pending_share[token] = (space.user_id, f"@ws:{ws['project_slug']}:{int(as_topic)}", comment)
                _trim(_pending_share)
                await msg.reply(f"Куда отправить воркспейс <b>{_esc(ws['name'])}</b>?",
                                parse_mode="HTML", reply_markup=_share_keyboard(token))
                return
            await msg.chat.do("typing")
            res = await orch.share_workspace_to_chat(
                space, ws["project_slug"], chat, comment=comment,
                author=_display_name(msg), as_topic=as_topic,
            )
            await _reply_html(msg, ("✅ " if res.startswith("OK") else "❌ ") + res)
            return
        if sub == "list":
            rows = [r for r in await db.list_workspaces(msg.chat.id) if r["project_slug"]]
            if not rows:
                await msg.reply("Воркспейсов пока нет. /ws new <имя> или кнопка «📌 Воркспейсы».")
                return
            lines = ["<b>Воркспейсы:</b>"]
            for r in rows:
                lines.append(f"• 📌 <b>{_esc(r['name'])}</b> → <code>wiki/projects/{_esc(r['project_slug'] or '?')}.md</code>")
            await msg.reply("\n".join(lines), parse_mode="HTML")
            return
        if sub == "new":
            name = rest.strip()
            if not name:
                await msg.reply("Использование: /ws new <имя воркспейса>")
                return
            ok, text, _tid = await _create_workspace(space, msg.chat.id, name)
            await msg.reply(("✅ " if ok else "❌ ") + _esc(text), parse_mode="HTML")
            return
        tid = _thread_id(msg)
        if tid is None:
            await msg.reply(
                "Это общее пространство (без темы). Создай тему в чате или /ws new <имя>; "
                "/ws list — список воркспейсов."
            )
            return
        if args:  # /ws <новое имя> — переименовать/привязать
            slug = await _bind_workspace(space, msg.chat.id, tid, args)
            await msg.reply(f"📌 Воркспейс → <b>{_esc(args)}</b> (<code>wiki/projects/{_esc(slug)}.md</code>)",
                            parse_mode="HTML")
            return
        ws = await _workspace(msg, space)
        if not ws:
            name = await _topic_name(msg, tid)
            await msg.reply(
                f"Эта тема («{_esc(name)}») пока просто чат, не воркспейс.",
                parse_mode="HTML", reply_markup=_bind_kb(tid, name),
            )
            return
        token = uuid.uuid4().hex[:10]
        _pending_share[token] = (space.user_id, f"@ws:{ws['project_slug']}:0", "")
        _trim(_pending_share)
        pages = len(__import__("bot.personal", fromlist=["collect_workspace_pages"]).collect_workspace_pages(space, ws["project_slug"]))
        await msg.reply(
            f"📌 Воркспейс <b>{_esc(ws['name'])}</b> → <code>wiki/projects/{_esc(ws['project_slug'])}.md</code>\n"
            f"Связанных страниц: {pages}.\n"
            "/ws <имя> — переименовать · /show " + _esc(ws['project_slug']) + " — страница проекта\n"
            "/ws share <чат> [topic] [комментарий] — выгрузить весь воркспейс "
            "(topic = в отдельную тему группы с тем же именем).",
            parse_mode="HTML", reply_markup=_share_keyboard(token),
        )

    # ---- artifacts

    @router.message(Command("list", "artifacts", "files"))
    async def cmd_list(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        kind = (command.args or "").strip().lower() or None
        if kind and kind not in ARTIFACT_KINDS:
            await msg.reply("Виды: " + ", ".join(ARTIFACT_KINDS))
            return
        arts = space.list_artifacts(kind=kind, limit=40)
        _last_listing[space.user_id] = arts
        head = f"**Артефакты{(' · ' + kind) if kind else ''}** ({len(arts)}):\n\n"
        await _reply_html(msg, head + artifact_markdown_table(arts)
                          + "\n\n_/show N · /share N [чат] · /del N_")

    @router.message(Command("show"))
    async def cmd_show(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        ref = (command.args or "").strip()
        rel = _resolve_ref(space, space.user_id, ref) if ref else None
        if not rel:
            await msg.reply("Не нашёл. Использование: /show <№ из /list | имя | путь>")
            return
        abs_path = space.root / rel
        ext = abs_path.suffix.lower()
        if ext == ".md":
            text = space.read_artifact(rel)
            # Длинные страницы не заваливаем чанками: первые ~3500 символов + кнопки.
            preview = text if len(text) <= 3500 else text[:3500].rsplit("\n", 1)[0] + "\n\n_… (дальше — кнопками ниже)_"
            await _reply_html(msg, f"`{rel}`\n\n{preview}")
            await msg.answer("Как удобнее смотреть?", reply_markup=_view_kb(space.user_id, rel))
        else:
            try:
                if ext in _IMG_EXTS:
                    await msg.answer_photo(FSInputFile(abs_path), caption=rel)
                elif ext in _VID_EXTS:
                    await msg.answer_video(FSInputFile(abs_path), caption=rel)
                else:
                    await msg.answer_document(FSInputFile(abs_path), caption=rel)
            except Exception as exc:  # noqa: BLE001
                await msg.reply(f"Не удалось отправить файл: {exc}")

    def _view_kb(user_id: int, rel: str) -> InlineKeyboardMarkup:
        token = uuid.uuid4().hex[:10]
        _pending_view[token] = (user_id, rel)
        _trim(_pending_view)
        row = [InlineKeyboardButton(text="📄 .md файл", callback_data=f"pfile:{token}")]
        if settings.telegraph_enabled:
            row.insert(0, InlineKeyboardButton(text="📰 Открыть статьёй", callback_data=f"ptg:{token}"))
        return InlineKeyboardMarkup(inline_keyboard=[row])

    @router.callback_query(F.data.startswith("pfile:"))
    async def cb_send_file(cb: CallbackQuery) -> None:
        token = (cb.data or "").split(":", 1)[1]
        entry = _pending_view.get(token)
        if not entry or not cb.from_user or cb.from_user.id != entry[0]:
            await cb.answer("⌛ устарело")
            return
        space = orch.personal_space(entry[0])
        path = space.root / entry[1]
        await cb.answer()
        try:
            await cb.message.answer_document(FSInputFile(path), caption=entry[1])  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            if cb.message:
                await cb.message.answer(f"Не удалось отправить файл: {_esc(str(exc))}", parse_mode="HTML")

    @router.callback_query(F.data.startswith("ptg:"))
    async def cb_telegraph(cb: CallbackQuery) -> None:
        token = (cb.data or "").split(":", 1)[1]
        entry = _pending_view.get(token)
        if not entry or not cb.from_user or cb.from_user.id != entry[0]:
            await cb.answer("⌛ устарело")
            return
        if not settings.telegraph_enabled:
            await cb.answer("Telegraph выключен", show_alert=True)
            return
        space = orch.personal_space(entry[0])
        rel = entry[1]
        await cb.answer("публикую…")
        try:
            md = space.read_artifact(rel)
            url = await TelegraphClient(db, author_name=space.display_name).publish_markdown(
                md, title=page_title(md, rel.rsplit("/", 1)[-1][:-3]), author=space.display_name,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("telegraph publish failed: %s", exc)
            if cb.message:
                await cb.message.answer(f"❌ Telegraph: {_esc(str(exc))}", parse_mode="HTML")
            return
        if cb.message:
            await cb.message.answer(
                f"📰 {url}\n<i>Страница публична по ссылке (без индексации).</i>",
                parse_mode="HTML", disable_web_page_preview=False,
            )

    @router.message(Command("del", "delete", "rm"))
    async def cmd_del(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        ref = (command.args or "").strip()
        rel = _resolve_ref(space, space.user_id, ref) if ref else None
        if not rel:
            await msg.reply("Не нашёл. Использование: /del <№ из /list | имя | путь>")
            return
        token = uuid.uuid4().hex[:10]
        _pending_delete[token] = (space.user_id, rel)
        _trim(_pending_delete)
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="🗑 Удалить", callback_data=f"pdel:{token}:y"),
            InlineKeyboardButton(text="Отмена", callback_data=f"pdel:{token}:n"),
        ]])
        await msg.reply(f"Удалить <code>{_esc(rel)}</code>?", parse_mode="HTML", reply_markup=kb)

    @router.message(Command("search"))
    async def cmd_search(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        q = (command.args or "").strip()
        if not q:
            await msg.reply("Использование: /search <строка>")
            return
        hits = space.wiki.search(q, max_hits=12)
        if not hits:
            await msg.reply("Ничего не найдено.")
            return
        out = "\n".join(f"• <code>{_esc(h['path'])}:{h['line']}</code> — {_esc(h['text'][:120])}"
                        for h in hits)
        await msg.reply(out[:4000], parse_mode="HTML")

    @router.message(Command("note"))
    async def cmd_note(msg: Message, command: CommandObject) -> None:
        """Быстрая заметка без агента."""
        space = await _guard(msg)
        if not space:
            return
        text = (command.args or "").strip()
        if not text:
            await msg.reply("Использование: /note <текст>")
            return
        ws = await _workspace(msg, space)
        now = datetime.now(timezone.utc)
        date = now.strftime("%Y-%m-%d")
        title = text.splitlines()[0][:60]
        rel = f"wiki/notes/{date}-{slugify(title)}-{now.strftime('%H%M%S')}.md"
        inbox_rel = space.append_inbox(who=_display_name(msg) + (f" · 📌 {ws['name']}" if ws else ""),
                                       text=f"/note {text}", ts=now, message_id=msg.message_id)
        project_line = f"project: \"[[{ws['project_slug']}]]\"\n" if ws else ""
        body = (
            "---\ntype: note\n"
            f"title: \"{title.replace(chr(34), chr(39))}\"\n"
            f"created: {date}\nupdated: {date}\ntags: [quick]\n{project_line}---\n\n"
            f"# {title}\n\n{text}\n\n_Источник: [[{inbox_rel[:-3]}#msg-{msg.message_id}]]_\n"
        )
        space.wiki.write_file(rel, body)
        try:
            space.wiki.append_file("index.md", f"- [[{rel.rsplit('/', 1)[-1][:-3]}]] — {title}")
            if ws:
                space.wiki.append_file(f"wiki/projects/{ws['project_slug']}.md",
                                       f"- **{date}** — заметка [[{rel.rsplit('/', 1)[-1][:-3]}]]: {title}")
        except Exception:  # noqa: BLE001
            pass
        space.log(f"note | {rel}")
        await msg.reply(f"📝 Записал: <code>{rel}</code>", parse_mode="HTML")

    @router.message(Command("ask"))
    async def cmd_ask(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        q = (command.args or msg.caption or "").strip()
        if q.startswith("/ask"):
            q = q[4:].strip()
        media = await _download_ask_media(bot, msg, space.wiki)
        if msg.reply_to_message:
            r = msg.reply_to_message
            quoted = (r.text or r.caption or "").strip()
            if quoted and not r.forum_topic_created:
                q = f"{q}\n\nКонтекст (реплай):\n{quoted}".strip()
            media += await _download_ask_media(bot, r, space.wiki)
        if media:
            q += "\n\n[ПРИКРЕПЛЕНО — вызови describe_media для каждого:]\n" + "\n".join(f"- {m}" for m in media)
        if not q:
            await msg.reply("Использование: /ask <вопрос>")
            return
        ws = await _workspace(msg, space)
        if ws:
            q = f"[Воркспейс: {ws['name']} → wiki/projects/{ws['project_slug']}.md]\n{q}"
        await _react(bot, msg, "👀")
        stop = asyncio.Event()
        typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop, _thread_id(msg)))
        try:
            answer = await orch.query(q, chat_cfg=space.cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("private /ask failed: %s", exc)
            await _react(bot, msg, "❌")
            await msg.reply("⚠️ Ошибка LLM, попробуй ещё раз.")
            return
        finally:
            stop.set()
            typing.cancel()
        await _send_agent_answer(msg, space, answer, title_hint=q.splitlines()[0],
                                 ws_name=ws["name"] if ws else None)

    @router.message(Command("share"))
    async def cmd_share(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        ref, chat, comment = _split_share_args(command.args or "")
        chats = settings.get_chats()
        # Вариант 1: /share реплаем на ответ бота → сохранить ответ как research и пошарить.
        r = msg.reply_to_message
        if r is not None and r.from_user and r.from_user.id == bot.id and not r.forum_topic_created \
                and not ref.isdigit() and not (ref and space.resolve_artifact(ref)):
            text = (r.text or r.caption or "").strip()
            if not text:
                await msg.reply("В этом сообщении нет текста для шаринга.")
                return
            if ref and ref.lstrip("#") in {c.name for c in chats}:
                chat, comment = ref.lstrip("#"), " ".join(filter(None, [chat or "", comment])).strip()
            title = text.splitlines()[0].lstrip("#* ").strip()[:60] or "Ответ бота"
            rel = space.save_research(title, text, query="(из диалога)")
        else:
            if not ref:
                await msg.reply(
                    "Использование: /share <№ из /list | имя | путь> [чат] [комментарий]\n"
                    "или /share реплаем на мой ответ."
                )
                return
            rel = _resolve_ref(space, space.user_id, ref)
            if not rel:
                await msg.reply(f"Артефакт «{ref}» не найден. Смотри /list.")
                return
        if chat is None:
            if len(chats) == 1:
                chat = chats[0].name
            else:
                token = uuid.uuid4().hex[:10]
                _pending_share[token] = (space.user_id, f"@file:{rel}", comment)
                _trim(_pending_share)
                kb = _share_keyboard(token)
                await msg.reply(f"Куда отправить <code>{_esc(rel)}</code>?",
                                parse_mode="HTML", reply_markup=kb)
                return
        await msg.chat.do("typing")
        res = await orch.share_to_chat(space, rel, chat, comment=comment, author=_display_name(msg))
        await _reply_html(msg, ("✅ " if res.startswith("OK") else "❌ ") + res)

    # ---- voice / image

    @router.message(Command("say"))
    async def cmd_say(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        text = (command.args or "").strip()
        r = msg.reply_to_message
        if not text and r is not None and not r.forum_topic_created:
            text = (r.text or r.caption or "").strip()
        if not text:
            await msg.reply("Использование: /say <текст> — или реплаем на сообщение.")
            return
        if not get_media_ai().tts_enabled:
            await msg.reply("🔇 Синтез речи выключен (нужен MINIMAX_API_KEY / TTS_PROVIDER).")
            return
        await _send_voice(msg, space, text)

    @router.message(Command("voice"))
    async def cmd_voice(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        args = (command.args or "").strip().split()
        if not args:
            await msg.reply(
                f"🗣 Голосовые ответы: <b>{await _voice_mode(space.user_id)}</b>, голос "
                f"<code>{_esc(await _voice_id(space.user_id))}</code>.\n"
                "/voice on|off|auto — режим (auto = отвечать голосом на голосовые)\n"
                "/voice set <voice_id> — выбрать голос (/voices — список)",
                parse_mode="HTML",
            )
            return
        if args[0].lower() in ("on", "off", "auto"):
            await db.set_state(f"voice_mode_{space.user_id}", args[0].lower())
            await msg.reply(f"🗣 Голосовые ответы: <b>{args[0].lower()}</b>", parse_mode="HTML")
            return
        if args[0].lower() == "set" and len(args) > 1:
            await db.set_state(f"voice_id_{space.user_id}", args[1])
            await msg.reply(f"🔊 Голос: <code>{_esc(args[1])}</code>", parse_mode="HTML")
            return
        await msg.reply("Использование: /voice on|off|auto | /voice set <voice_id>")

    @router.message(Command("voices"))
    async def cmd_voices(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        prefix = (command.args or "Russian").strip()
        try:
            voices = await get_media_ai().list_voices(prefix)
        except MediaAIError as exc:
            await msg.reply(f"❌ {_esc(str(exc))}", parse_mode="HTML")
            return
        if not voices:
            await msg.reply(f"Голосов с префиксом «{_esc(prefix)}» не найдено. Попробуй /voices English.",
                            parse_mode="HTML")
            return
        lines = [f"<b>Голоса ({_esc(prefix)}):</b>"]
        for v in voices[:25]:
            lines.append(f"• <code>{_esc(v['voice_id'])}</code> — {_esc(v['description'][:90])}")
        lines.append("\nВыбрать: /voice set <voice_id>")
        await msg.reply("\n".join(lines)[:4000], parse_mode="HTML")

    @router.message(Command("imagine", "draw", "img"))
    async def cmd_imagine(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        prompt, aspect = _parse_imagine_args(command.args or "")
        if not prompt:
            await msg.reply("Использование: /imagine <описание> [1:1|16:9|9:16|4:3|3:4|3:2|2:3|21:9]")
            return
        ai = get_media_ai()
        if not ai.image_enabled:
            await msg.reply("🎨 Генерация картинок выключена (нужен MINIMAX_API_KEY / IMAGE_PROVIDER).")
            return
        await _react(bot, msg, "👀")
        stop = asyncio.Event()
        typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop, _thread_id(msg), "upload_photo"))
        try:
            images = await ai.generate_images(prompt, aspect_ratio=aspect, n=1)
        except MediaAIError as exc:
            await msg.reply(f"❌ {_esc(str(exc))}", parse_mode="HTML")
            return
        finally:
            stop.set()
            typing.cancel()
        now = datetime.now(timezone.utc)
        rel = space.wiki.save_asset(images[0], f"gen_{now.strftime('%H%M%S')}.jpg", now.strftime("%Y-%m-%d"))
        ws = await _workspace(msg, space)
        space.append_inbox(who=_display_name(msg) + (f" · 📌 {ws['name']}" if ws else ""),
                           text=f"/imagine {prompt}", ts=now, message_id=msg.message_id,
                           media_type="generated", media_path=rel)
        await msg.answer_photo(FSInputFile(space.root / rel), caption=f"🎨 {prompt[:200]}\n{rel}")

    # ------------------------------------------------------------- callbacks

    @router.callback_query(F.data.startswith("ans:"))
    async def cb_answer(cb: CallbackQuery) -> None:
        await handle_answer_callback(cb, db, orch)

    @router.callback_query(F.data.startswith("pshr:"))
    async def cb_share(cb: CallbackQuery) -> None:
        parts = (cb.data or "").split(":", 2)
        if len(parts) != 3:
            await cb.answer("bad data")
            return
        token, chat = parts[1], parts[2]
        entry = _pending_share.get(token)
        if not entry:
            await cb.answer("⌛ устарело", show_alert=False)
            return
        user_id, payload, title = entry
        if not cb.from_user or cb.from_user.id != user_id:
            await cb.answer("не твоя кнопка")
            return
        space = orch.personal_space(user_id)
        author = cb.from_user.full_name or str(user_id)
        await cb.answer("отправляю…")
        if payload.startswith("@ws:"):
            _, slug, flag = payload.split(":", 2)
            res = await orch.share_workspace_to_chat(space, slug, chat, comment=title, author=author,
                                                     as_topic=flag == "1")
        else:
            if payload.startswith("@file:"):
                rel, comment = payload[6:], title
            else:
                rel = space.save_research(title or "Ответ бота", payload, query="(из диалога)")
                comment = ""
            res = await orch.share_to_chat(space, rel, chat, comment=comment, author=author)
        _pending_share.pop(token, None)
        try:
            await cb.message.edit_text(  # type: ignore[union-attr]
                md_to_tg_html(("✅ " if res.startswith("OK") else "❌ ") + res)[:4000],
                parse_mode="HTML",
            )
        except Exception:  # noqa: BLE001
            if cb.message:
                await cb.message.answer(md_to_tg_html(res)[:4000], parse_mode="HTML")

    @router.callback_query(F.data.startswith("pdel:"))
    async def cb_delete(cb: CallbackQuery) -> None:
        parts = (cb.data or "").split(":", 2)
        if len(parts) != 3:
            await cb.answer("bad data")
            return
        token, yes = parts[1], parts[2] == "y"
        entry = _pending_delete.pop(token, None)
        if not entry:
            await cb.answer("⌛ устарело")
            return
        user_id, rel = entry
        if not cb.from_user or cb.from_user.id != user_id:
            await cb.answer("не твоя кнопка")
            return
        if not yes:
            await cb.answer("отменено")
            try:
                await cb.message.edit_text("Отменено.")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            return
        space = orch.personal_space(user_id)
        try:
            space.delete_artifact(rel)
            text = f"🗑 Удалено: <code>{_esc(rel)}</code>"
        except Exception as exc:  # noqa: BLE001
            text = f"❌ {_esc(str(exc))}"
        await cb.answer("ok")
        try:
            await cb.message.edit_text(text, parse_mode="HTML")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------- menu buttons

    def _voice_kb() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="🔊 on", callback_data="pvm:on"),
            InlineKeyboardButton(text="🎤 auto", callback_data="pvm:auto"),
            InlineKeyboardButton(text="🔇 off", callback_data="pvm:off"),
        ]])

    async def _show_workspaces(msg: Message) -> None:
        rows = [r for r in await db.list_workspaces(msg.chat.id) if r["project_slug"]]
        lines = ["<b>Воркспейсы</b> — темы в этом чате, привязанные к проектам."]
        if rows:
            for r in rows:
                lines.append(f"• 📌 <b>{_esc(r['name'])}</b> → <code>wiki/projects/{_esc(r['project_slug'] or '?')}.md</code>")
        else:
            lines.append("Пока нет. Создай тему в чате или нажми кнопку ниже.")
        lines.append("\nВ теме: /ws — статус, /ws share &lt;чат&gt; — выгрузить в группу.")
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="➕ Создать воркспейс", callback_data="pwsnew"),
        ]])
        await msg.reply("\n".join(lines), parse_mode="HTML", reply_markup=kb)

    async def _show_admin_panel(msg: Message) -> None:
        if not is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        reqs = await _access_requests()
        dyn = await db.get_private_users()
        lines = ["<b>🛡 Админ-панель</b>", "",
                 f"🧠 Модель: <b>{_esc(orch.model_info())}</b>", "",
                 "<b>Доступ к личному режиму:</b>"]
        for uid in settings.telegram_admins:
            lines.append(f"• <code>{uid}</code> — admin (.env)")
        for uid in settings.telegram_private_users:
            if uid not in settings.telegram_admins:
                lines.append(f"• <code>{uid}</code> — .env")
        for uid in dyn:
            lines.append(f"• <code>{uid}</code> — через /allow")
        kb_rows: list[list[InlineKeyboardButton]] = []
        if reqs:
            lines += ["", f"<b>Ожидают подтверждения ({len(reqs)}):</b>"]
            for r in reqs[-8:]:
                who = _esc(r.get("name") or "") + (f" (@{_esc(r['username'])})" if r.get("username") else "")
                lines.append(f"• {who} · <code>{r['id']}</code> · {r.get('ts', '')}")
                kb_rows.append([
                    InlineKeyboardButton(text=f"✅ {str(r.get('name') or r['id'])[:20]}", callback_data=f"pacc:{r['id']}:y"),
                    InlineKeyboardButton(text="❌", callback_data=f"pacc:{r['id']}:n"),
                ])
        else:
            lines += ["", "Запросов на доступ нет. Новый пользователь напишет боту — тебе придёт карточка с кнопками."]
        for uid in dyn[:6]:
            kb_rows.append([InlineKeyboardButton(text=f"🚫 отозвать {uid}", callback_data=f"pdeny:{uid}")])
        prov_row = [
            InlineKeyboardButton(text=f"🧠 {p['key']}", callback_data=f"pmodel:{p['key']}")
            for p in orch.client.available_providers() if p["has_key"] and p["key"] != "ollama"
        ][:4]
        if prov_row:
            kb_rows.append(prov_row)
        lines += ["", "Команды: /allow &lt;id&gt; · /deny &lt;id&gt; · /users · /model &lt;provider&gt; [model]"]
        await msg.reply("\n".join(lines)[:4000], parse_mode="HTML",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows) if kb_rows else None)

    @router.message(F.text.in_(MENU_BUTTONS))
    async def on_menu_button(msg: Message) -> None:
        space = await _guard(msg)
        if not space:
            return
        t = (msg.text or "").strip()
        if t == BTN_WS:
            await _show_workspaces(msg)
        elif t == BTN_ART:
            arts = space.list_artifacts(limit=40)
            _last_listing[space.user_id] = arts
            await _reply_html(msg, f"**Артефакты** ({len(arts)}):\n\n" + artifact_markdown_table(arts)
                              + "\n\n_/show N · /share N [чат] · /del N · /list notes|ideas|research|files_")
        elif t == BTN_PROJ:
            await cmd_projects(msg)
        elif t == BTN_SPACE:
            await cmd_space(msg)
        elif t == BTN_VOICE:
            await msg.reply(
                f"🗣 Голосовые ответы: <b>{await _voice_mode(space.user_id)}</b>, голос "
                f"<code>{_esc(await _voice_id(space.user_id))}</code>\n"
                "auto — голосом на голосовые. /voices — выбрать голос, /say — озвучить текст.",
                parse_mode="HTML", reply_markup=_voice_kb(),
            )
        elif t == BTN_HELP:
            await msg.reply(HELP_TEXT, parse_mode="HTML", reply_markup=_menu_kb(is_admin(space.user_id)))
        elif t == BTN_ADMIN:
            await _show_admin_panel(msg)

    @router.callback_query(F.data == "pwsnew")
    async def cb_ws_new(cb: CallbackQuery) -> None:
        if not cb.from_user:
            return
        _awaiting[cb.from_user.id] = "ws_name"
        await cb.answer()
        if cb.message:
            await cb.message.answer("Как назвать воркспейс? Напиши имя одним сообщением (или /cancel).")

    @router.message(Command("cancel"))
    async def cmd_cancel(msg: Message) -> None:
        if msg.from_user and _awaiting.pop(msg.from_user.id, None):
            await msg.reply("Ок, отменил.")
        else:
            await msg.reply("Нечего отменять.")

    @router.callback_query(F.data.startswith("pwsbind:"))
    async def cb_ws_bind(cb: CallbackQuery) -> None:
        if not cb.from_user or not cb.message:
            return
        tid = int((cb.data or "").split(":", 1)[1])
        space = orch.personal_space(cb.from_user.id)
        row = await db.get_workspace(cb.message.chat.id, tid)
        name = (row["name"] if row else None) or f"тема {tid}"
        slug = await _bind_workspace(space, cb.message.chat.id, tid, name)
        await cb.answer("готово")
        try:
            await cb.message.edit_text(
                f"📌 Воркспейс <b>{_esc(name)}</b> → <code>wiki/projects/{_esc(slug)}.md</code>.\n"
                "Пиши сюда всё по этому проекту — я буду вести его страницу. "
                "/ws share &lt;чат&gt; — выгрузить команде.",
                parse_mode="HTML",
            )
        except Exception:  # noqa: BLE001
            pass

    @router.callback_query(F.data.startswith("pwsname:"))
    async def cb_ws_name(cb: CallbackQuery) -> None:
        if not cb.from_user or not cb.message:
            return
        tid = int((cb.data or "").split(":", 1)[1])
        _awaiting[cb.from_user.id] = f"ws_bind:{tid}"
        await cb.answer()
        await cb.message.answer("Как назвать воркспейс? Напиши имя одним сообщением (или /cancel).")

    @router.callback_query(F.data.startswith("pwsno:"))
    async def cb_ws_no(cb: CallbackQuery) -> None:
        await cb.answer("ок, просто чат")
        if cb.message:
            try:
                await cb.message.edit_text("💬 Ок, это просто чат. Передумаешь — /ws <имя> в этой теме.")
            except Exception:  # noqa: BLE001
                pass

    @router.callback_query(F.data.startswith("pvm:"))
    async def cb_voice_mode(cb: CallbackQuery) -> None:
        if not cb.from_user:
            return
        mode = (cb.data or "").split(":", 1)[1]
        if mode not in ("on", "off", "auto"):
            await cb.answer("bad")
            return
        await db.set_state(f"voice_mode_{cb.from_user.id}", mode)
        await cb.answer(f"голос: {mode}")
        try:
            await cb.message.edit_text(f"🗣 Голосовые ответы: <b>{mode}</b>", parse_mode="HTML",  # type: ignore[union-attr]
                                       reply_markup=_voice_kb())
        except Exception:  # noqa: BLE001
            pass

    @router.callback_query(F.data.startswith("pacc:"))
    async def cb_access(cb: CallbackQuery) -> None:
        if not cb.from_user or not is_admin(cb.from_user.id):
            await cb.answer("только для админов", show_alert=True)
            return
        parts = (cb.data or "").split(":")
        if len(parts) != 3 or not parts[1].lstrip("-").isdigit():
            await cb.answer("bad data")
            return
        uid, yes = int(parts[1]), parts[2] == "y"
        items = [i for i in await _access_requests() if int(i.get("id", 0)) != uid]
        await _set_access_requests(items)
        if yes:
            await db.add_private_user(uid)
            _denied_notified.discard(uid)
            try:
                await bot.send_message(uid, "✅ Доступ к личному режиму открыт. Напиши /start.",
                                       reply_markup=_menu_kb(False))
            except Exception as exc:  # noqa: BLE001
                log.info("notify user %s failed: %s", uid, exc)
            text = f"✅ <code>{uid}</code> — доступ открыт"
        else:
            text = f"❌ <code>{uid}</code> — отклонено"
        await cb.answer("ok")
        try:
            await cb.message.edit_text(text, parse_mode="HTML")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    @router.callback_query(F.data.startswith("pdeny:"))
    async def cb_deny(cb: CallbackQuery) -> None:
        if not cb.from_user or not is_admin(cb.from_user.id):
            await cb.answer("только для админов", show_alert=True)
            return
        uid = int((cb.data or "").split(":", 1)[1])
        await db.remove_private_user(uid)
        await cb.answer(f"отозван {uid}")
        if cb.message:
            await cb.message.answer(f"🚫 <code>{uid}</code> — доступ отозван", parse_mode="HTML")

    @router.callback_query(F.data.startswith("pmodel:"))
    async def cb_model(cb: CallbackQuery) -> None:
        if not cb.from_user or not is_admin(cb.from_user.id):
            await cb.answer("только для админов", show_alert=True)
            return
        provider = (cb.data or "").split(":", 1)[1]
        try:
            desc = await orch.set_model(provider)
        except Exception as exc:  # noqa: BLE001
            await cb.answer(str(exc)[:180], show_alert=True)
            return
        await cb.answer("ok")
        if cb.message:
            await cb.message.answer(f"✅ Модель: <b>{_esc(desc)}</b>\n\n" + format_model_list(orch),
                                    parse_mode="HTML")

    # ------------------------------------------------------------- catch-all

    @router.message()
    async def on_message(msg: Message) -> None:
        # Служебные сообщения от ботов (в т.ч. «тема создана» от нас самих при
        # create_forum_topic) — не пользовательский ввод и не запрос доступа.
        if msg.from_user and msg.from_user.is_bot:
            return
        space = await _guard(msg)
        if not space:
            return
        # Ожидали имя воркспейса после кнопки «➕ Создать».
        pending = _awaiting.get(space.user_id)
        if pending and msg.text and not msg.text.startswith("/"):
            _awaiting.pop(space.user_id, None)
            if pending == "ws_name":
                ok, text, _tid = await _create_workspace(space, msg.chat.id, msg.text)
                await msg.reply(("✅ " if ok else "❌ ") + _esc(text), parse_mode="HTML")
                return
            if pending.startswith("ws_bind:"):
                tid = int(pending.split(":", 1)[1])
                slug = await _bind_workspace(space, msg.chat.id, tid, msg.text)
                await msg.reply(f"📌 Тема → воркспейс <b>{_esc(msg.text.strip()[:128])}</b> "
                                f"(<code>wiki/projects/{_esc(slug)}.md</code>)", parse_mode="HTML")
                return
        # Сервисные сообщения тем: создание / переименование → воркспейс.
        if msg.forum_topic_created and msg.message_thread_id:
            tid = msg.message_thread_id
            name = (msg.forum_topic_created.name or f"тема {tid}")[:128]
            if await db.get_workspace(msg.chat.id, tid) is None:
                await db.upsert_workspace(msg.chat.id, tid, name, None)
                if not _looks_auto_topic(name, "") and name != INBOX_TOPIC_NAME:
                    try:
                        await msg.answer(
                            f"🆕 Тема «{_esc(name)}». Сделать её воркспейсом (проектом)?",
                            parse_mode="HTML", reply_markup=_bind_kb(tid, name),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug("bind hint failed: %s", exc)
            return
        if msg.forum_topic_edited and msg.message_thread_id:
            name = msg.forum_topic_edited.name
            if name:
                await db.upsert_workspace(msg.chat.id, msg.message_thread_id, name, None)
            return
        if msg.forum_topic_closed or msg.forum_topic_reopened or msg.general_forum_topic_hidden \
                or msg.general_forum_topic_unhidden:
            return
        # Ответ на вопрос агента (ask_user) — реплаем на сообщение с id=N.
        qid = question_id_from_reply(msg)
        if qid > 0 and msg.text:
            if await orch.deliver_answer(qid, msg.text):
                await msg.reply("✅ ответ передан агенту")
                return
        async with _lock(space.user_id):
            await _process_message(msg, space)

    async def _process_message(msg: Message, space: PersonalSpace) -> None:
        ts = msg.date or datetime.now(timezone.utc)
        who = _display_name(msg)
        text = (msg.text or msg.caption or "").strip()
        media_type: str | None = None
        media_path: str | None = None
        transcript: str | None = None
        extracted: str | None = None
        intake_block: str | None = None
        attached: list[str] = []
        ws = await _workspace(msg, space)
        ws_name = ws["name"] if ws else None

        has_media = bool(msg.photo or msg.video or msg.document or msg.audio
                         or msg.voice or msg.video_note or msg.animation or msg.sticker)
        spoke = bool(msg.voice or msg.video_note or msg.audio)
        if has_media:
            await _react(bot, msg, "👀")
            media_type, media_path = await _download_media(
                bot, msg, settings.vault_path, space.cfg.name,
            )
            if media_type and not media_path:
                # Telegram Bot API не отдаёт ботам файлы > 20 МБ (getFile). Не зовём агента —
                # иначе он найдёт в raw/assets старый файл и разберёт его повторно.
                size = 0
                for attr in ("document", "video", "audio", "animation"):
                    obj = getattr(msg, attr, None)
                    if obj is not None and getattr(obj, "file_size", None):
                        size = int(obj.file_size)
                        break
                big = size > 20 * 1024 * 1024
                await _react(bot, msg, "❌")
                await msg.reply(
                    "⚠️ Не смог скачать вложение" + (f" ({size // 1024 // 1024} МБ)" if size else "") + ".\n"
                    + ("Telegram не отдаёт ботам файлы больше <b>20 МБ</b>. Варианты:\n"
                       "• пришли <b>прямую ссылку</b> на файл (облако/сервер) — скачаю сам, лимит 512 МБ;\n"
                       "• для экспорта чата выгрузи его <b>без медиа</b> (только текст/JSON) — он весит мало;\n"
                       "• или разбей архив на части до 20 МБ."
                       if big else "Попробуй ещё раз или пришли прямую ссылку на файл."),
                    parse_mode="HTML",
                )
                return
            if media_path:
                abs_path = space.root / media_path
                ext = abs_path.suffix.lower()
                stop = asyncio.Event()
                typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop, _thread_id(msg),
                                                          "upload_document" if media_type == "document" else "typing"))
                try:
                    if media_type in {"voice", "audio"} or ext in AUDIO_EXTENSIONS:
                        transcript = await extract_text_from_file(abs_path)
                    elif media_type in {"video_note", "video"} or ext in VIDEO_EXTENSIONS:
                        transcript = await transcribe_video(abs_path)
                        attached.append(media_path)
                    else:
                        # документы, архивы, фото → единый приём: текст/OCR/описание + sidecar
                        intake = await process_attachment(space.wiki, media_path, media_type)
                        attached.append(media_path)
                        intake_block = intake.agent_block()
                        if intake.archive is not None and intake.archive.dates:
                            await _offer_archive_ingest(msg, space, intake)
                except Exception as exc:  # noqa: BLE001
                    log.warning("private: media parse failed %s: %s", media_path, exc)
                finally:
                    stop.set()
                    typing.cancel()
                if transcript and transcript.startswith("("):
                    log.warning("private: STT placeholder: %s", transcript)
                    await msg.reply(f"⚠️ Не смог распознать речь: {_esc(transcript)}", parse_mode="HTML")
                    transcript = None

        # Ссылки на файлы в тексте (архив/док из облака) — скачиваем сами (обход лимита 20 МБ).
        if text and not has_media:
            for url in FILE_URL_RE.findall(text)[:2]:
                stop = asyncio.Event()
                typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop, _thread_id(msg), "upload_document"))
                try:
                    rel = await download_url_to_vault(space.wiki, url)
                    intake = await process_attachment(space.wiki, rel)
                    attached.append(rel)
                    media_type, media_path = "url_file", rel
                    intake_block = (intake_block + "\n\n" if intake_block else "") + intake.agent_block()
                    if intake.archive is not None and intake.archive.dates:
                        await _offer_archive_ingest(msg, space, intake)
                except Exception as exc:  # noqa: BLE001
                    log.warning("url download failed %s: %s", url, exc)
                    await msg.reply(f"⚠️ Не удалось скачать {_esc(url[:80])}: {_esc(str(exc)[:200])}", parse_mode="HTML")
                finally:
                    stop.set()
                    typing.cancel()

        # Текст для агента и для журнала.
        parts: list[str] = []
        if text:
            parts.append(text)
        if transcript:
            parts.append(f"🎤 (голосовое, расшифровка): {transcript.strip()}")
        if extracted and not extracted.startswith("("):
            parts.append(f"📄 (содержимое файла {media_path}):\n{extracted.strip()[:6000]}")
        if intake_block:
            parts.append(intake_block)
        full_text = "\n\n".join(parts).strip()
        if not full_text and not attached:
            await msg.reply("Пустое сообщение — нечего обрабатывать.")
            return

        # Журнал: БД + raw/inbox.
        try:
            await db.save_message(
                chat_id=msg.chat.id, message_id=msg.message_id, topic_id=_thread_id(msg),
                user_id=msg.from_user.id if msg.from_user else None,
                username=msg.from_user.username if msg.from_user else None,
                full_name=who, text=full_text or None,
                media_type=media_type, media_path=media_path,
                reply_to_message_id=msg.reply_to_message.message_id if msg.reply_to_message else None,
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("private: save_message failed: %s", exc)
        space.append_inbox(who=who + (f" · 📌 {ws_name}" if ws_name else ""), text=full_text, ts=ts,
                           message_id=msg.message_id, media_type=media_type, media_path=media_path)

        if transcript:
            await msg.reply(f"🎤 <i>{_esc(transcript.strip()[:3000])}</i>", parse_mode="HTML")

        # Контекст: реплай пользователя на сообщение бота (не на корень темы).
        agent_text = full_text
        r = msg.reply_to_message
        if r is not None and r.from_user and r.from_user.id == bot.id and not r.forum_topic_created \
                and r.message_id != msg.message_thread_id:
            prev = (r.text or r.caption or "").strip()
            if prev:
                agent_text = f"[Мой предыдущий ответ]:\n{prev[:1500]}\n\n[Реплика пользователя]:\n{full_text}"

        history = await _history(msg)
        if not has_media:
            await _react(bot, msg, "👀")
        stop = asyncio.Event()
        typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop, _thread_id(msg)))
        try:
            answer = await orch.personal_turn(
                space, agent_text or "(без текста — см. вложение)",
                message_id=msg.message_id, attached_media=attached, history=history,
                workspace=ws, progress=_draft_progress(msg),
                workspace_fn=_workspace_tool_fn(msg, space),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("personal_turn failed")
            await _react(bot, msg, "❌")
            await msg.reply(f"⚠️ Ошибка агента: {type(exc).__name__}. Попробуй ещё раз.")
            return
        finally:
            stop.set()
            typing.cancel()
        mode = await _voice_mode(space.user_id)
        voice = mode == "on" or (mode == "auto" and spoke)
        title_hint = (text or transcript or "Ответ").splitlines()[0]
        await _send_agent_answer(msg, space, answer, title_hint=title_hint, ws_name=ws_name, voice=voice)

    return router
