"""Команды бота: /ask /ingest /lint /summary /note /search /log /pause /resume /help."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from ..config import settings
from ..db import DB
from ..orchestrator import Orchestrator
from ..wiki import Wiki

log = logging.getLogger(__name__)

router = Router(name="commands")


def _is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in settings.telegram_admins


HELP_TEXT = (
    "*DuckSidian — команды*\n\n"
    "/ask <вопрос> — спросить wiki\n"
    "/search <строка> — поиск по vault\n"
    "/summary day|week — сводка\n"
    "/log — последние записи log.md\n"
    "/note <текст> — ручная заметка в raw/notes/\n"
    "/ingest [today|N|<YYYY-MM-DD>] — внеплановый ingest *(admin)*\n"
    "/lint — health-check *(admin)*\n"
    "/pause /resume — авто-ingest *(admin)*\n"
)


def setup(db: DB, wiki: Wiki, orch: Orchestrator) -> Router:

    @router.message(Command("help", "start"))
    async def cmd_help(msg: Message) -> None:
        await msg.reply(HELP_TEXT, parse_mode="Markdown")

    @router.message(Command("ask"))
    async def cmd_ask(msg: Message, command: CommandObject) -> None:
        q = (command.args or "").strip()
        if not q:
            await msg.reply("Использование: /ask <вопрос>")
            return
        await msg.chat.do("typing")
        answer = await orch.query(q)
        await msg.reply(answer[:4000] or "(пусто)")

    @router.message(Command("search"))
    async def cmd_search(msg: Message, command: CommandObject) -> None:
        q = (command.args or "").strip()
        if not q:
            await msg.reply("Использование: /search <строка>")
            return
        hits = wiki.search(q, max_hits=10)
        if not hits:
            await msg.reply("Ничего не найдено.")
            return
        out = "\n".join(
            f"• `{h['path']}:{h['line']}` — {h['text'][:120]}"
            for h in hits
        )
        await msg.reply(out[:4000], parse_mode="Markdown")

    @router.message(Command("summary"))
    async def cmd_summary(msg: Message, command: CommandObject) -> None:
        period = (command.args or "day").strip()
        await msg.chat.do("typing")
        out = await orch.summary(period)
        await msg.reply(out[:4000] or "(пусто)")

    @router.message(Command("note"))
    async def cmd_note(msg: Message, command: CommandObject) -> None:
        text = (command.args or "").strip()
        if not text:
            await msg.reply("Использование: /note <текст>")
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        author = (
            msg.from_user.full_name if msg.from_user else "anon"
        ) if msg.from_user else "anon"
        slug = "manual"
        rel = f"raw/notes/{ts}-{slug}.md"
        wiki.write_file(
            rel,
            f"# Note {ts}\n\n_by {author}_\n\n{text}\n",
        )
        await msg.reply(f"Записал в `{rel}`", parse_mode="Markdown")

    @router.message(Command("log"))
    async def cmd_log(msg: Message) -> None:
        try:
            content = wiki.read_file("log.md")
        except Exception as exc:  # noqa: BLE001
            await msg.reply(f"log.md недоступен: {exc}")
            return
        lines = [ln for ln in content.splitlines() if ln.startswith("## [")]
        tail = "\n".join(lines[-10:]) or "(лог пуст)"
        await msg.reply(f"```\n{tail}\n```", parse_mode="Markdown")

    @router.message(Command("ingest"))
    async def cmd_ingest(msg: Message, command: CommandObject) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        arg = (command.args or "today").strip()
        await msg.reply(f"⏳ ingest: {arg}…")
        if arg == "today":
            out = await orch.ingest_today()
        elif arg.isdigit():
            out = await orch.ingest_last_n(int(arg))
        else:
            try:
                datetime.strptime(arg, "%Y-%m-%d")
            except ValueError:
                await msg.reply("Использование: /ingest [today|N|YYYY-MM-DD]")
                return
            out = await orch.ingest_for_date(arg)
        await msg.reply(f"✅ {out[:3500]}")

    @router.message(Command("lint"))
    async def cmd_lint(msg: Message) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        await msg.reply("⏳ lint…")
        out = await orch.lint()
        await msg.reply(f"🩺 {out[:3800]}")

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
