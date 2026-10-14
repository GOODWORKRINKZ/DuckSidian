"""Личный режим (DM): персональное пространство сотрудника.

Роутер ловит ВСЕ сообщения из приватных чатов (включается в Dispatcher первым,
чтобы групповой commands-роутер со своим chat_guard их не проглатывал).

Что умеет:
- доступ по allowlist (админы + TELEGRAM_PRIVATE_USERS + /allow);
- любое сообщение (текст / голос / видео-кружок / фото / документ) →
  скачивание в личный vault, STT/парсинг, запись в raw/inbox и БД,
  затем агент (запись мысли / поиск фактов / управление артефактами);
- /ask /search /note /projects /list /show /del /share /log /space /help;
- кнопки под ответами агента: 📤 поделиться результатом с групповым чатом.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from html import escape as _esc

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from ..config import settings
from ..db import DB
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
from ..tg_format import md_to_tg_html, split_tg_chunks
from .admin_shared import (
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

HELP_TEXT = (
    "<b>DuckSidian — личный режим</b>\n\n"
    "Просто пиши или говори голосом — я веду твоё личное пространство:\n"
    "• мысли, идеи, статусы по проектам → записываю в заметки/проекты;\n"
    "• «посмотри / проверь / найди / сравни …» → ищу в твоих данных, в wiki "
    "команды и в интернете, собираю подборку с источниками;\n"
    "• «покажи / удали / отправь в чат …» → управляю артефактами.\n\n"
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
    "/space — что лежит в пространстве\n"
    "/log — последние операции\n"
    "/chats — куда можно шарить\n"
    "/model — модель LLM <i>(admin)</i>\n"
    "/allow /deny /users — доступ к личному режиму <i>(admin)</i>\n"
)

# user_id → последний показанный список артефактов (для /show 3, /share 2 …)
_last_listing: dict[int, list[Artifact]] = {}
# token → (user_id, text_markdown, title) — ответы агента, ожидающие кнопки «поделиться»
_pending_share: dict[str, tuple[int, str, str]] = {}
# token → (user_id, rel_path) — подтверждение удаления
_pending_delete: dict[str, tuple[int, str]] = {}
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


async def _typing_loop(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, "typing")
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


def setup(db: DB, orch: Orchestrator, bot: Bot) -> Router:

    async def _guard(msg: Message) -> PersonalSpace | None:
        uid = msg.from_user.id if msg.from_user else None
        if not await is_private_allowed(db, uid):
            if uid not in _denied_notified:
                _denied_notified.add(uid or 0)
                await msg.reply(
                    "⛔ Личный режим не включён для тебя.\n"
                    f"Твой id: <code>{uid}</code> — попроси админа выполнить "
                    f"<code>/allow {uid}</code>.",
                    parse_mode="HTML",
                )
            return None
        return orch.personal_space(uid, _display_name(msg))  # type: ignore[arg-type]

    async def _save_bot_reply(space: PersonalSpace, chat_id: int, text: str,
                              sent_ids: list[int]) -> None:
        now = datetime.now(timezone.utc)
        mid = sent_ids[0] if sent_ids else -int(now.timestamp() * 1000)
        try:
            await db.save_message(
                chat_id=chat_id, message_id=mid, topic_id=None,
                user_id=bot.id, username="bot", full_name="DuckSidian",
                text=text, media_type=None, media_path=None,
                reply_to_message_id=None, ts=now,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("private: save bot reply failed: %s", exc)
        space.append_inbox(who="🤖 DuckSidian", text=text, ts=now, message_id=mid)

    async def _history(chat_id: int, exclude_message_id: int) -> list[dict[str, str]]:
        rows = await db.fetch_last_messages(chat_id, settings.personal_history_size + 1)
        out: list[dict[str, str]] = []
        for r in rows:
            if r["message_id"] == exclude_message_id:
                continue
            text = (r["text"] or "").strip()
            if not text:
                continue
            role = "assistant" if r["user_id"] == bot.id else "user"
            out.append({"role": role, "content": text[:2000]})
        return out[-settings.personal_history_size:]

    async def _send_agent_answer(msg: Message, space: PersonalSpace, answer: str,
                                 *, title_hint: str) -> None:
        display, has_images = _strip_images_block(answer)
        sent_ids = await _reply_html(msg, display or "(пусто)")
        if has_images:
            await _attach_media(msg, answer, space.wiki.root)
        await _save_bot_reply(space, msg.chat.id, display, sent_ids)
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
        await msg.reply(HELP_TEXT, parse_mode="HTML")

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
        text = (
            f"<b>Твоё пространство</b> <code>users/{space.user_id}/</code>\n"
            f"📁 проекты: {c['projects']}  📝 заметки: {c['notes']}  💡 идеи: {c['ideas']}\n"
            f"🔎 подборки: {c['research']}  ❓ queries: {c['queries']}  📎 файлы: {c['files']}\n"
            f"🧠 модель: {_esc(orch.model_info())}"
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
            await msg.reply("Проектов пока нет. Напиши про проект — я заведу страницу.")
            return
        emoji = {"active": "🟢", "paused": "🟡", "done": "✅", "archive": "📁"}
        lines = ["<b>Твои проекты:</b>"]
        for it in items:
            upd = f" <i>{it['updated']}</i>" if it["updated"] else ""
            lines.append(f"{emoji.get(it['status'], '⚪️')} <b>{_esc(it['name'])}</b>{upd}\n"
                         f" {_esc(it['summary'])}")
        await msg.reply("\n\n".join(lines)[:4000], parse_mode="HTML")

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
            await _reply_html(msg, f"`{rel}`\n\n{text[:12000]}")
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
        now = datetime.now(timezone.utc)
        date = now.strftime("%Y-%m-%d")
        title = text.splitlines()[0][:60]
        rel = f"wiki/notes/{date}-{slugify(title)}-{now.strftime('%H%M%S')}.md"
        inbox_rel = space.append_inbox(who=_display_name(msg), text=f"/note {text}",
                                       ts=now, message_id=msg.message_id)
        body = (
            "---\ntype: note\n"
            f"title: \"{title.replace(chr(34), chr(39))}\"\n"
            f"created: {date}\nupdated: {date}\ntags: [quick]\n---\n\n"
            f"# {title}\n\n{text}\n\n_Источник: [[{inbox_rel[:-3]}#msg-{msg.message_id}]]_\n"
        )
        space.wiki.write_file(rel, body)
        try:
            space.wiki.append_file("index.md", f"- [[{rel.rsplit('/', 1)[-1][:-3]}]] — {title}")
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
            if quoted:
                q = f"{q}\n\nКонтекст (реплай):\n{quoted}".strip()
            media += await _download_ask_media(bot, r, space.wiki)
        if media:
            q += "\n\n[ПРИКРЕПЛЕНО — вызови describe_media для каждого:]\n" + "\n".join(f"- {m}" for m in media)
        if not q:
            await msg.reply("Использование: /ask <вопрос>")
            return
        await _react(bot, msg, "👀")
        stop = asyncio.Event()
        typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop))
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
        await _send_agent_answer(msg, space, answer, title_hint=q.splitlines()[0])

    @router.message(Command("share"))
    async def cmd_share(msg: Message, command: CommandObject) -> None:
        space = await _guard(msg)
        if not space:
            return
        ref, chat, comment = _split_share_args(command.args or "")
        chats = settings.get_chats()
        # Вариант 1: /share реплаем на ответ бота → сохранить ответ как research и пошарить.
        r = msg.reply_to_message
        if r is not None and r.from_user and r.from_user.id == bot.id and not ref.isdigit() \
                and not (ref and space.resolve_artifact(ref)):
            text = (r.text or r.caption or "").strip()
            if not text:
                await msg.reply("В этом сообщении нет текста для шаринга.")
                return
            # ref здесь может быть именем чата
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
        if payload.startswith("@file:"):
            rel, comment = payload[6:], title
        else:
            rel = space.save_research(title or "Ответ бота", payload, query="(из диалога)")
            comment = ""
        await cb.answer("отправляю…")
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

    # ------------------------------------------------------------- catch-all

    @router.message()
    async def on_message(msg: Message) -> None:
        space = await _guard(msg)
        if not space:
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
        attached: list[str] = []

        has_media = bool(msg.photo or msg.video or msg.document or msg.audio
                         or msg.voice or msg.video_note or msg.animation or msg.sticker)
        if has_media:
            await _react(bot, msg, "👀")
            media_type, media_path = await _download_media(
                bot, msg, settings.vault_path, space.cfg.name,
            )
            if media_path:
                abs_path = space.root / media_path
                ext = abs_path.suffix.lower()
                try:
                    if media_type in {"voice", "audio"} or ext in AUDIO_EXTENSIONS:
                        await msg.chat.do("record_voice")
                        transcript = await extract_text_from_file(abs_path)
                    elif media_type in {"video_note", "video"} or ext in VIDEO_EXTENSIONS:
                        await msg.chat.do("record_voice")
                        transcript = await transcribe_video(abs_path)
                        attached.append(media_path)
                    elif media_type == "document":
                        extracted = await extract_text_from_file(abs_path)
                        attached.append(media_path)
                    else:  # photo / animation / sticker → агент сам посмотрит через describe_media
                        attached.append(media_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning("private: media parse failed %s: %s", media_path, exc)
                if transcript and transcript.startswith("("):
                    # Заглушка вида "(ошибка STT: ...)" — покажем пользователю, но не как текст.
                    log.warning("private: STT placeholder: %s", transcript)
                    await msg.reply(f"⚠️ Не смог распознать речь: {_esc(transcript)}", parse_mode="HTML")
                    transcript = None

        # Текст для агента и для журнала.
        parts: list[str] = []
        if text:
            parts.append(text)
        if transcript:
            parts.append(f"🎤 (голосовое, расшифровка): {transcript.strip()}")
        if extracted and not extracted.startswith("("):
            parts.append(f"📄 (содержимое файла {media_path}):\n{extracted.strip()[:6000]}")
        full_text = "\n\n".join(parts).strip()
        if not full_text and not attached:
            await msg.reply("Пустое сообщение — нечего обрабатывать.")
            return

        # Журнал: БД + raw/inbox.
        try:
            await db.save_message(
                chat_id=msg.chat.id, message_id=msg.message_id, topic_id=None,
                user_id=msg.from_user.id if msg.from_user else None,
                username=msg.from_user.username if msg.from_user else None,
                full_name=who, text=full_text or None,
                media_type=media_type, media_path=media_path,
                reply_to_message_id=msg.reply_to_message.message_id if msg.reply_to_message else None,
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("private: save_message failed: %s", exc)
        space.append_inbox(who=who, text=full_text, ts=ts, message_id=msg.message_id,
                           media_type=media_type, media_path=media_path)

        if transcript:
            # Показать расшифровку сразу — пользователь видит, что бот понял.
            await msg.reply(f"🎤 <i>{_esc(transcript.strip()[:3000])}</i>", parse_mode="HTML")

        # Контекст: реплай пользователя на сообщение бота.
        agent_text = full_text
        r = msg.reply_to_message
        if r is not None and r.from_user and r.from_user.id == bot.id:
            prev = (r.text or r.caption or "").strip()
            if prev:
                agent_text = f"[Мой предыдущий ответ]:\n{prev[:1500]}\n\n[Реплика пользователя]:\n{full_text}"

        history = await _history(msg.chat.id, msg.message_id)
        if not has_media:
            await _react(bot, msg, "👀")
        stop = asyncio.Event()
        typing = asyncio.create_task(_typing_loop(bot, msg.chat.id, stop))
        try:
            answer = await orch.personal_turn(
                space, agent_text or "(без текста — см. вложение)",
                message_id=msg.message_id, attached_media=attached, history=history,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("personal_turn failed")
            await _react(bot, msg, "❌")
            await msg.reply(f"⚠️ Ошибка агента: {type(exc).__name__}. Попробуй ещё раз.")
            return
        finally:
            stop.set()
            typing.cancel()
        title_hint = (text or transcript or "Ответ").splitlines()[0]
        await _send_agent_answer(msg, space, answer, title_hint=title_hint)

    return router
