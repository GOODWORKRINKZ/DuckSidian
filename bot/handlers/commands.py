"""Команды бота: /ask /ingest /lint /summary /note /search /log /pause /resume /help /chats."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from ..config import ChatConfig, settings
from ..db import DB
from ..orchestrator import Orchestrator, _chat_wiki
from ..tg_format import md_to_tg_html, split_tg_chunks
from ..topic_manager import ensure_bot_topic, reset_bot_topic
from ..wiki import Wiki

log = logging.getLogger(__name__)

router = Router(name="commands")


def _is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in settings.telegram_admins


def _chat_for_msg(msg: Message) -> ChatConfig | None:
    """Найти ChatConfig для чата, из которого пришло сообщение.

    Если чат не в списке отслеживаемых — вернёт None. Админские команды
    в таком случае падают на первый чат из конфига.
    """
    for cfg in settings.get_chats():
        if cfg.chat_id == msg.chat.id:
            return cfg
    return None


def _wiki_for_msg(msg: Message) -> Wiki:
    """Per-chat Wiki для сообщения. Fallback — первый чат из конфига."""
    cfg = _chat_for_msg(msg) or settings.get_chats()[0]
    return _chat_wiki(cfg)


async def _reply_html(msg: Message, markdown_text: str) -> None:
    """Конвертировать markdown-ответ агента в Telegram HTML и отправить чанками."""
    html_text = md_to_tg_html(markdown_text or "(пусто)")
    chunks = split_tg_chunks(html_text)
    for i, chunk in enumerate(chunks):
        if i == 0:
            await msg.reply(chunk, parse_mode="HTML")
        else:
            await msg.answer(chunk, parse_mode="HTML")


HELP_TEXT = (
    "<b>DuckSidian — команды</b>\n\n"
    "/ask &lt;вопрос&gt; — спросить wiki\n"
    "/search &lt;строка&gt; — поиск по vault\n"
    "/summary day|week — сводка\n"
    "/log — последние записи log.md\n"
    "/note &lt;текст&gt; — ручная заметка в raw/notes/\n"
    "/chats — список отслеживаемых чатов\n"
    "/ingest [today|N|&lt;YYYY-MM-DD&gt;] [&lt;chat&gt;] — внеплановый ingest <i>(admin)</i>\n"
    "/lint — health-check <i>(admin)</i>\n"
    "/topic — создать/пересоздать служебный топик <i>(admin)</i>\n"
    "/pause /resume — авто-ingest <i>(admin)</i>\n"
)


def setup(db: DB, wiki: Wiki, orch: Orchestrator) -> Router:

    @router.message(Command("help", "start"))
    async def cmd_help(msg: Message) -> None:
        log.info(
            "CMD chat_id=%s title=%r thread=%s user=%s",
            msg.chat.id, msg.chat.title, msg.message_thread_id,
            msg.from_user.username if msg.from_user else None,
        )
        # При /start в группе — попробовать создать бот-топик
        if msg.chat.type in ("group", "supergroup"):
            await ensure_bot_topic(msg.bot, db, msg.chat.id)  # type: ignore[arg-type]
        await msg.reply(HELP_TEXT, parse_mode="HTML")

    @router.message(Command("topic"))
    async def cmd_topic(msg: Message) -> None:
        """Пересоздать служебный топик бота."""
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        await reset_bot_topic(db, msg.chat.id)
        tid = await ensure_bot_topic(msg.bot, db, msg.chat.id)  # type: ignore[arg-type]
        if tid:
            await msg.reply(f"✅ Топик создан: thread_id=<code>{tid}</code>", parse_mode="HTML")
        else:
            await msg.reply("❌ Не удалось создать топик. Проверь права бота (Manage Topics).")

    @router.message(Command("chats"))
    async def cmd_chats(msg: Message) -> None:
        chats = settings.get_chats()
        lines = ["<b>Отслеживаемые чаты:</b>"]
        for c in chats:
            lines.append(
                f"• <code>{c.name}</code> — chat_id <code>{c.chat_id}</code>"
                + (f", topic <code>{c.topic_id}</code>" if c.topic_id else "")
            )
        await msg.reply("\n".join(lines), parse_mode="HTML")

    @router.message(Command("ask"))
    async def cmd_ask(msg: Message, command: CommandObject) -> None:
        q = (command.args or "").strip()
        if not q:
            await msg.reply("Использование: /ask <вопрос>")
            return
        await msg.chat.do("typing")
        try:
            answer = await orch.query(q)
            await _reply_html(msg, answer or "(пусто)")
        except Exception as exc:  # noqa: BLE001
            log.error("query failed: %s", exc)
            await msg.reply("⚠️ Ошибка при обращении к DeepSeek, попробуй ещё раз.")

    @router.message(Command("search"))
    async def cmd_search(msg: Message, command: CommandObject) -> None:
        q = (command.args or "").strip()
        if not q:
            await msg.reply("Использование: /search <строка>")
            return
        chat_wiki = _wiki_for_msg(msg)
        hits = chat_wiki.search(q, max_hits=10)
        if not hits:
            await msg.reply("Ничего не найдено.")
            return
        out = "\n".join(
            f"• <code>{h['path']}:{h['line']}</code> — {h['text'][:120]}"
            for h in hits
        )
        await msg.reply(out[:4000], parse_mode="HTML")

    @router.message(Command("summary"))
    async def cmd_summary(msg: Message, command: CommandObject) -> None:
        period = (command.args or "day").strip()
        await msg.chat.do("typing")
        out = await orch.summary(period)
        await _reply_html(msg, out or "(пусто)")

    @router.message(Command("note"))
    async def cmd_note(msg: Message, command: CommandObject) -> None:
        text = (command.args or "").strip()
        if not text:
            await msg.reply("Использование: /note <текст>")
            return
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        author = (
            msg.from_user.full_name if msg.from_user else "anon"
        ) if msg.from_user else "anon"
        slug = "manual"
        rel = f"raw/notes/{ts}-{slug}.md"

        # 1) Пишем копию в per-chat raw/notes/ (для видимости в Obsidian)
        chat_wiki = _wiki_for_msg(msg)
        chat_wiki.write_file(
            rel,
            f"# Note {ts}\n\n_by {author}_\n\n{text}\n",
        )

        # 2) Сохраняем как сообщение в DB — чтобы ingest_for_date / auto_ingest подхватил.
        # Синтетический message_id: отрицательный timestamp в миллисекундах —
        # гарантированно не пересекается с настоящими TG-id (всегда положительными).
        synthetic_id = -int(now.timestamp() * 1000)
        user = msg.from_user
        text_for_db = f"✍️ /note: {text}"
        try:
            await db.save_message(
                chat_id=msg.chat.id,
                message_id=synthetic_id,
                topic_id=msg.message_thread_id,
                user_id=user.id if user else None,
                username=user.username if user else None,
                full_name=user.full_name if user else None,
                text=text_for_db,
                media_type=None,
                media_path=f"{rel}",
                reply_to_message_id=None,
                ts=now,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("/note db.save_message failed: %s", exc)
            await msg.reply(
                f"⚠️ Нота записана в <code>{rel}</code>, но в DB не сохранилась: {exc}",
                parse_mode="HTML",
            )
            return
        await msg.reply(
            f"✍️ Записал в <code>{rel}</code> и в DB — войдёт в ближайший daily-batch.",
            parse_mode="HTML",
        )

    @router.message(Command("log"))
    async def cmd_log(msg: Message) -> None:
        chat_wiki = _wiki_for_msg(msg)
        try:
            content = chat_wiki.read_file("log.md")
        except Exception as exc:  # noqa: BLE001
            await msg.reply(f"log.md недоступен: {exc}")
            return
        lines = [ln for ln in content.splitlines() if ln.startswith("## [")]
        tail = "\n".join(lines[-10:]) or "(лог пуст)"
        from html import escape as _esc
        await msg.reply(f"<pre>{_esc(tail)}</pre>", parse_mode="HTML")

    @router.message(Command("ingest"))
    async def cmd_ingest(msg: Message, command: CommandObject) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        args = (command.args or "today").strip().split()
        arg = args[0]
        # Опциональный слаг чата: /ingest today sales
        chat_name_filter = args[1] if len(args) > 1 else None
        chats = settings.get_chats()
        if chat_name_filter:
            chats = [c for c in chats if c.name == chat_name_filter]
            if not chats:
                await msg.reply(f"Чат '{chat_name_filter}' не найден. /chats — список.")
                return
        await msg.reply(f"⏳ ingest: {arg} {'('+ chat_name_filter +')' if chat_name_filter else '(все чаты)'}…")
        results = []
        for chat_cfg in chats:
            if arg == "today":
                out = await orch.ingest_today(chat_cfg)
            elif arg.isdigit():
                out = await orch.ingest_last_n(int(arg), chat_cfg)
            else:
                try:
                    datetime.strptime(arg, "%Y-%m-%d")
                except ValueError:
                    await msg.reply("Использование: /ingest [today|N|YYYY-MM-DD] [chat_name]")
                    return
                out = await orch.ingest_for_date(arg, chat_cfg)
            results.append(f"[{chat_cfg.name}] {out}")
        await _reply_html(msg, "✅ " + chr(10).join(results))

    @router.message(Command("ingestfiles"))
    async def cmd_ingest_files(msg: Message, command: CommandObject) -> None:
        """Инжест уже готовых raw/daily/*.md (не из БД).

        /ingestfiles           — следующие 5 необработанных дней
        /ingestfiles 10        — следующие 10 дней
        /ingestfiles 2025-05-01 — конкретная дата
        """
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        args = (command.args or "").strip().split()
        arg = args[0] if args else ""
        chat_name_filter = args[1] if len(args) > 1 else None
        chats = settings.get_chats()
        if chat_name_filter:
            chats = [c for c in chats if c.name == chat_name_filter]
            if not chats:
                await msg.reply(f"Чат '{chat_name_filter}' не найден.")
                return
        await msg.reply("⏳ ingest-files запущен…")
        results = []
        for chat_cfg in chats:
            if not arg or arg.isdigit():
                limit = int(arg) if arg.isdigit() else 5
                out = await orch.ingest_raw_pending(chat_cfg, limit=limit)
            else:
                try:
                    datetime.strptime(arg, "%Y-%m-%d")
                except ValueError:
                    await msg.reply("Использование: /ingest-files [N|YYYY-MM-DD] [chat_name]")
                    return
                out = await orch.ingest_raw_file(arg, chat_cfg)
            results.append(f"[{chat_cfg.name}] {out}")
        await _reply_html(msg, "✅ " + chr(10).join(results))

    @router.message(Command("lint"))
    async def cmd_lint(msg: Message) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        await msg.reply("⏳ lint…")
        out = await orch.lint()
        await _reply_html(msg, "🩺 " + out)

    @router.message(Command("pause"))
    async def cmd_pause(msg: Message) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        await db.set_state("paused", "1")
        await msg.reply("⏸ авто-ingest приостановлен")

    @router.message(Command("resume"))
    async def cmd_resume(msg: Message) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        await db.set_state("paused", "0")
        await msg.reply("▶️ авто-ingest возобновлён")

    return router
