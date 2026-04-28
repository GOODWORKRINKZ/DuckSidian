"""Высокоуровневые операции: ingest / query / lint / summary.

Поддержка нескольких чатов: каждый чат имеет свою поддиректорию vault'а
(vault/<chat_name>/) и изолированный Wiki-экземпляр.

ask_user pipeline: агент → bot posts в forum-топик → ждёт ответ → агенту.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from .agent.deepseek import DeepSeekClient
from .agent.loop import run_agent
from .agent.prompts import (
    INGEST_SYSTEM,
    LINT_SYSTEM,
    QUERY_SYSTEM,
    SUMMARY_SYSTEM,
)
from .agent.tools import ToolExecutor
from .config import ChatConfig, settings
from .db import DB
from .git_sync import commit_and_push
from .topic_manager import ensure_bot_topic
from .wiki import Wiki

log = logging.getLogger(__name__)


def _chat_wiki(chat_cfg: ChatConfig) -> Wiki:
    """Вернуть Wiki для конкретного чата (vault/<name>/)."""
    root = settings.vault_path / chat_cfg.name
    root.mkdir(parents=True, exist_ok=True)
    # Если AGENTS.md отсутствует — скопируем из глобального vault-template
    if not (root / "AGENTS.md").exists():
        tmpl = settings.vault_path / "AGENTS.md"
        if tmpl.exists():
            shutil.copy2(tmpl, root / "AGENTS.md")
        # Базовые директории
        for subdir in (
            "raw/daily", "raw/notes", "raw/assets",
            "wiki/entities", "wiki/concepts", "wiki/sources",
            "wiki/daily", "wiki/queries",
        ):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        for fname in ("index.md", "log.md"):
            p = root / fname
            if not p.exists():
                p.write_text(f"# {fname}\n", encoding="utf-8")
    return Wiki(root)


class Orchestrator:
    def __init__(self, bot: Bot, db: DB, wiki: Wiki):
        self.bot = bot
        self.db = db
        self.wiki = wiki  # глобальный wiki (для /ask, /search, /lint, /summary)
        self.client = DeepSeekClient()
        self._waiters: dict[int, asyncio.Future[str]] = {}

    async def aclose(self) -> None:
        await self.client.aclose()

    # --- ask_user pipeline ---

    async def _bot_topic_id(self, chat_cfg: ChatConfig | None) -> int | None:
        """Получить thread_id бот-топика (или None если не форум)."""
        chat_id = (chat_cfg.chat_id if chat_cfg else None) or settings.telegram_chat_id
        return await ensure_bot_topic(self.bot, self.db, chat_id)

    async def _ask_user(
        self, question: str, options: list[str] | None,
        chat_cfg: ChatConfig | None = None,
    ) -> str:
        topic_id = await self._bot_topic_id(chat_cfg)
        chat_id = (chat_cfg.chat_id if chat_cfg else None) or settings.telegram_chat_id

        qid = await self.db.add_question(
            batch_id=None,
            question=question,
            options=json.dumps(options, ensure_ascii=False) if options else None,
        )
        kb = None
        if options:
            buttons = [
                [
                    InlineKeyboardButton(
                        text=opt[:60],
                        callback_data=f"ans:{qid}:{i}",
                    )
                ]
                for i, opt in enumerate(options[:8])
            ]
            kb = InlineKeyboardMarkup(inline_keyboard=buttons)

        text = (
            f"❓ *Вопрос от агента*\n\n{question}\n\n"
            f"_id={qid}. Ответьте reply'ем на это сообщение или кнопкой._"
        )
        sent = await self.bot.send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=topic_id,
            reply_markup=kb,
            parse_mode="Markdown",
        )
        await self.db.attach_question_post(qid, sent.message_id)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._waiters[qid] = fut
        try:
            return await asyncio.wait_for(fut, timeout=60 * 30)
        except asyncio.TimeoutError:
            return "(timeout: команда не ответила за 30 минут, делай по умолчанию)"
        finally:
            self._waiters.pop(qid, None)

    async def deliver_answer(self, qid: int, answer: str) -> bool:
        await self.db.answer_question(qid, answer)
        fut = self._waiters.get(qid)
        if fut and not fut.done():
            fut.set_result(answer)
            return True
        return False

    def _make_executor(self, wiki: Wiki, chat_cfg: ChatConfig | None = None) -> ToolExecutor:
        async def ask_user_cb(question: str, options: list[str] | None) -> str:
            return await self._ask_user(question, options, chat_cfg)
        return ToolExecutor(wiki, ask_user_cb, deepseek_client=self.client)

    # --- INGEST ---

    async def ingest_for_date(
        self, date_iso: str, chat_cfg: ChatConfig | None = None
    ) -> str:
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)

        rows = await self.db.fetch_messages_for_date(cfg.chat_id, date_iso)
        msgs = [dict(r) for r in rows]
        if cfg.topic_id is not None:
            msgs = [m for m in msgs if m.get("topic_id") != cfg.topic_id]
        if not msgs:
            return f"Нет сообщений за {date_iso} в чате [{cfg.name}]."

        rel = wiki.write_daily_raw(date_iso, msgs)
        batch_id = await self.db.create_batch(date_iso)
        await self.db.mark_processed([m["id"] for m in msgs], batch_id)

        executor = self._make_executor(wiki, cfg)
        user_prompt = (
            f"Сделай ingest нового дневного батча: `{rel}`. "
            f"Дата: {date_iso}. Чат: {cfg.name}. В батче {len(msgs)} сообщений. "
            f"Следуй процедуре ingest из AGENTS.md."
        )
        try:
            run = await run_agent(
                client=self.client,
                executor=executor,
                system_prompt=INGEST_SYSTEM,
                user_prompt=user_prompt,
                max_steps=30,
            )
            status = "ok" if run.finished else "partial"
            await self.db.finish_batch(
                batch_id, status, len(msgs), notes=run.summary[:500],
            )
            self._append_log(
                wiki,
                f"ingest | {date_iso} | {len(msgs)} сообщений | "
                f"{run.steps} шагов | {status}",
            )
            if settings.git_autocommit:
                commit_and_push(f"ingest {cfg.name} {date_iso}: {len(msgs)} msgs")
            return run.summary or "Ingest завершён."
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest failed")
            await self.db.finish_batch(batch_id, "error", len(msgs), str(exc))
            return f"Ошибка ingest: {exc}"

    async def ingest_today(self, chat_cfg: ChatConfig | None = None) -> str:
        today = datetime.now(timezone.utc).date().isoformat()
        return await self.ingest_for_date(today, chat_cfg)

    async def ingest_last_n(
        self, n: int, chat_cfg: ChatConfig | None = None
    ) -> str:
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)

        rows = await self.db.fetch_last_messages(cfg.chat_id, n)
        msgs = [dict(r) for r in rows]
        if not msgs:
            return "Нет сообщений в БД."
        date_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d_adhoc-%H%M%S")
        rel = wiki.write_daily_raw(date_iso, msgs)
        batch_id = await self.db.create_batch(date_iso)
        executor = self._make_executor(wiki, cfg)
        user_prompt = (
            f"Внеплановый ingest. Файл: `{rel}` ({len(msgs)} сообщений). "
            f"Чат: {cfg.name}. Следуй процедуре ingest из AGENTS.md."
        )
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=INGEST_SYSTEM,
            user_prompt=user_prompt,
            max_steps=30,
        )
        await self.db.finish_batch(
            batch_id, "ok" if run.finished else "partial", len(msgs)
        )
        self._append_log(wiki, f"ingest-adhoc | last {n} | {run.steps} шагов")
        if settings.git_autocommit:
            commit_and_push(f"ingest adhoc {cfg.name}: last {n} msgs")
        return run.summary or "Ingest завершён."

    # --- QUERY ---

    async def query(self, question: str, chat_cfg: ChatConfig | None = None) -> str:
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        executor = self._make_executor(wiki, cfg)
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=QUERY_SYSTEM,
            user_prompt=question,
            max_steps=15,
        )
        return run.summary or "Не нашёл ответа."

    # --- LINT ---

    async def lint(self, chat_cfg: ChatConfig | None = None) -> str:
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        executor = self._make_executor(wiki, cfg)
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=LINT_SYSTEM,
            user_prompt="Сделай health-check вики. Верни отчёт.",
            max_steps=20,
        )
        self._append_log(wiki, f"lint | {run.steps} шагов")
        if settings.git_autocommit:
            commit_and_push(f"lint {cfg.name}")
        return run.summary or "Lint завершён."

    # --- SUMMARY ---

    async def summary(
        self, period: str, chat_cfg: ChatConfig | None = None
    ) -> str:
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        executor = self._make_executor(wiki, cfg)
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=SUMMARY_SYSTEM,
            user_prompt=f"Сделай сводку за период: {period}.",
            max_steps=10,
        )
        return run.summary or "Сводка не сгенерирована."

    # --- helpers ---

    def _append_log(self, wiki: Wiki, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            wiki.append_file("log.md", f"## [{ts}] {line}")
        except Exception as exc:  # noqa: BLE001
            log.warning("append log failed: %s", exc)
