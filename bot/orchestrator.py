"""Высокоуровневые операции: ingest / query / lint / summary.

Здесь же реализован ask_user pipeline: агент задаёт вопрос → бот постит в
forum-топик с inline-кнопками → ждёт ответ из БД → возвращает агенту.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
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
from .config import settings
from .db import DB
from .git_sync import commit_and_push
from .wiki import Wiki

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, bot: Bot, db: DB, wiki: Wiki):
        self.bot = bot
        self.db = db
        self.wiki = wiki
        self.client = DeepSeekClient()
        self._waiters: dict[int, asyncio.Future[str]] = {}

    async def aclose(self) -> None:
        await self.client.aclose()

    # --- ask_user pipeline ---

    async def _ask_user(
        self, question: str, options: list[str] | None
    ) -> str:
        topic_id = settings.telegram_topic_id
        chat_id = settings.telegram_chat_id

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
            return await asyncio.wait_for(fut, timeout=60 * 30)  # 30 минут
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

    def _make_executor(self) -> ToolExecutor:
        return ToolExecutor(self.wiki, self._ask_user)

    # --- INGEST ---

    async def ingest_for_date(self, date_iso: str) -> str:
        """Сформировать дневной батч и запустить агентский ingest."""
        rows = await self.db.fetch_messages_for_date(
            settings.telegram_chat_id, date_iso
        )
        msgs = [dict(r) for r in rows]
        # Не тащим в raw сообщения из служебного forum-топика бота.
        if settings.telegram_topic_id is not None:
            msgs = [
                m for m in msgs
                if m.get("topic_id") != settings.telegram_topic_id
            ]
        if not msgs:
            return f"Нет сообщений за {date_iso}."

        rel = self.wiki.write_daily_raw(date_iso, msgs)
        batch_id = await self.db.create_batch(date_iso)
        await self.db.mark_processed([m["id"] for m in msgs], batch_id)

        executor = self._make_executor()
        user_prompt = (
            f"Сделай ingest нового дневного батча: `{rel}`. "
            f"Дата: {date_iso}. В батче {len(msgs)} сообщений. "
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
                batch_id, status, len(msgs),
                notes=run.summary[:500],
            )
            self._append_log(
                f"ingest | {date_iso} | {len(msgs)} сообщений | "
                f"{run.steps} шагов | {status}"
            )
            if settings.git_autocommit:
                commit_and_push(f"ingest {date_iso}: {len(msgs)} msgs")
            return run.summary or "Ingest завершён."
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest failed")
            await self.db.finish_batch(batch_id, "error", len(msgs), str(exc))
            return f"Ошибка ingest: {exc}"

    async def ingest_today(self) -> str:
        today = datetime.now(timezone.utc).date().isoformat()
        return await self.ingest_for_date(today)

    async def ingest_last_n(self, n: int) -> str:
        """Внеплановый ingest последних N сообщений (помечаем 'adhoc')."""
        rows = await self.db.fetch_last_messages(settings.telegram_chat_id, n)
        msgs = [dict(r) for r in rows]
        if not msgs:
            return "Нет сообщений в БД."
        date_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d_adhoc-%H%M%S")
        rel = self.wiki.write_daily_raw(date_iso, msgs)
        batch_id = await self.db.create_batch(date_iso)
        executor = self._make_executor()
        user_prompt = (
            f"Внеплановый ingest. Файл: `{rel}` ({len(msgs)} сообщений). "
            f"Следуй процедуре ingest из AGENTS.md."
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
        self._append_log(f"ingest-adhoc | last {n} | {run.steps} шагов")
        if settings.git_autocommit:
            commit_and_push(f"ingest adhoc: last {n} msgs")
        return run.summary or "Ingest завершён."

    # --- QUERY ---

    async def query(self, question: str) -> str:
        executor = self._make_executor()
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=QUERY_SYSTEM,
            user_prompt=question,
            max_steps=15,
        )
        return run.summary or "Не нашёл ответа."

    # --- LINT ---

    async def lint(self) -> str:
        executor = self._make_executor()
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=LINT_SYSTEM,
            user_prompt="Сделай health-check вики. Верни отчёт.",
            max_steps=20,
        )
        self._append_log(f"lint | {run.steps} шагов")
        if settings.git_autocommit:
            commit_and_push("lint pass")
        return run.summary or "Lint завершён."

    # --- SUMMARY ---

    async def summary(self, period: str) -> str:
        executor = self._make_executor()
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=SUMMARY_SYSTEM,
            user_prompt=f"Сделай сводку за период: {period}.",
            max_steps=10,
        )
        return run.summary or "Сводка не сгенерирована."

    # --- helpers ---

    def _append_log(self, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            self.wiki.append_file("log.md", f"## [{ts}] {line}")
        except Exception as exc:  # noqa: BLE001
            log.warning("append log failed: %s", exc)
