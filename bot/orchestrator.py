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
    REVISE_SYSTEM,
    SUMMARY_SYSTEM,
    TRIZ_SYSTEM,
)
from .agent.tools import ToolExecutor
from .config import ChatConfig, settings
from .db import DB
from .git_sync import commit_and_push
from .tg_format import md_to_tg_html
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
            "wiki/projects",
            "wiki/entities", "wiki/concepts", "wiki/sources",
            "wiki/daily", "wiki/queries",
        ):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        for fname in ("index.md", "log.md"):
            p = root / fname
            if not p.exists():
                p.write_text(f"# {fname}\n", encoding="utf-8")
    # Миграция: добавить wiki/projects/ в существующие волты и шаблон
    projects_dir = root / "wiki" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    tmpl_src = Path(__file__).resolve().parent.parent / "vault-template" / "wiki" / "projects" / "_template.md"
    tmpl_dst = projects_dir / "_template.md"
    if tmpl_src.exists() and not tmpl_dst.exists():
        try:
            shutil.copy2(tmpl_src, tmpl_dst)
        except Exception as exc:  # noqa: BLE001
            log.warning("copy _template.md failed: %s", exc)
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
            f"❓ <b>Вопрос от агента</b>\n\n{md_to_tg_html(question)}\n\n"
            f"<i>id={qid}. Ответьте reply'ем на это сообщение или кнопкой.</i>"
        )
        sent = await self.bot.send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=topic_id,
            reply_markup=kb,
            parse_mode="HTML",
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

    def _make_executor(self, wiki: Wiki, chat_cfg: ChatConfig | None = None,
                       file_cache: dict[str, str] | None = None) -> ToolExecutor:
        async def ask_user_cb(question: str, options: list[str] | None) -> str:
            return await self._ask_user(question, options, chat_cfg)
        return ToolExecutor(wiki, ask_user_cb, deepseek_client=self.client,
                            file_cache=file_cache)

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

    @staticmethod
    def _extract_media_refs(content: str, date_iso: str) -> list[str]:
        """Достать все wiki-ссылки на медиа дня: [[raw/assets/<date>/<file>]]."""
        import re
        pat = re.compile(
            r"\[\[(raw/assets/" + re.escape(date_iso) + r"/[^\]\r\n]+)\]\]"
        )
        seen: set[str] = set()
        out: list[str] = []
        for m in pat.finditer(content):
            rel = m.group(1)
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
        return out

    async def _preanalyze_media(
        self, refs: list[str], wiki: Wiki, executor: ToolExecutor
    ) -> dict[str, str]:
        """Получить описание каждого медиа-файла дня.

        Кэшируется на диске в `.cache/media/<flat>.txt`, чтобы не повторять
        дорогую vision/STT обработку при retry/повторных ingest. Возвращает
        отображение rel_path → краткое описание (≤600 символов).
        """
        cache_root = wiki.root / ".cache" / "media"
        cache_root.mkdir(parents=True, exist_ok=True)
        out: dict[str, str] = {}
        for rel in refs:
            flat = rel.replace("/", "__")
            cache_file = cache_root / f"{flat}.txt"
            if cache_file.exists():
                desc = cache_file.read_text(encoding="utf-8")
            else:
                try:
                    desc = await executor._describe_media(rel)
                except Exception as exc:  # noqa: BLE001
                    log.warning("preanalyze_media %s failed: %s", rel, exc)
                    desc = f"[ошибка анализа: {exc}]"
                # Сохраняем как есть; усечение делаем при встраивании.
                try:
                    cache_file.write_text(desc, encoding="utf-8")
                except OSError as exc:
                    log.warning("media cache write %s failed: %s", rel, exc)
            # Чистим возможный DSML-мусор и нормализуем переносы.
            desc = desc.strip()
            if len(desc) > 600:
                desc = desc[:600].rstrip() + "…"
            desc = desc.replace("\n", " ⏎ ")
            out[rel] = desc
        return out

    @staticmethod
    def _enrich_with_media(content: str, media_map: dict[str, str]) -> str:
        """Встроить описание медиа сразу после строки, ссылающейся на него.

        Karpathy: "LLMs can't natively read markdown with inline images in one
        pass — workaround: read the text first, then view referenced images
        separately to gain additional context." Делаем это заранее: в одном
        проходе агент уже видит и сообщение, и описание медиа, без вызова
        describe_media.
        """
        if not media_map:
            return content
        out_lines: list[str] = []
        for line in content.splitlines():
            out_lines.append(line)
            for rel, desc in media_map.items():
                if f"[[{rel}]]" in line:
                    out_lines.append(f"> 🧠 [МЕДИА {rel.rsplit('/', 1)[-1]}]: {desc}")
                    break
        return "\n".join(out_lines)

    @staticmethod
    def _split_raw_chunks(
        content: str, max_msgs: int = 8, max_chars: int = 3_000
    ) -> list[str]:
        """Разбить raw/daily/*.md на чанки по сообщениям.

        Сообщения в файле разделены строкой "---". Заголовок (до первого "---")
        повторяется в каждом чанке как контекст. Каждый чанк ≤ max_msgs
        сообщений и ≤ max_chars символов.
        """
        # Разбиваем по строке "---" (граница между сообщениями).
        parts = content.split("\n---\n")
        if len(parts) <= 2:
            return [content]
        header = parts[0]
        msgs = [p for p in parts[1:] if p.strip()]
        chunks: list[str] = []
        cur: list[str] = []
        cur_chars = 0
        for m in msgs:
            m_len = len(m) + 5  # +"\n---\n"
            if cur and (len(cur) >= max_msgs or cur_chars + m_len > max_chars):
                chunks.append(header + "\n---\n" + "\n---\n".join(cur))
                cur, cur_chars = [], 0
            cur.append(m)
            cur_chars += m_len
        if cur:
            chunks.append(header + "\n---\n" + "\n---\n".join(cur))
        return chunks

    async def ingest_raw_file(
        self, date_iso: str, chat_cfg: ChatConfig | None = None
    ) -> str:
        """Ingest целого дня в стиле Karpathy LLM-Wiki.

        Pipeline:
          1) Прескан медиа: для каждого `[[raw/assets/<date>/...]]` получаем
             описание (vision/STT/parse) ОДИН раз и кэшируем на диске.
          2) Обогащённый контент: описания медиа встраиваются inline сразу
             после ссылки на файл — агенту не нужно дёргать describe_media.
          3) Чанкинг по сообщениям; в каждом чанке агент обновляет
             wiki/entities, wiki/projects, wiki/sources, wiki/concepts,
             index.md. НИКАКОЙ дайджест-страницы wiki/daily/<date>.md
             не создаётся (Karpathy: wiki — это entity/concept/source pages,
             не пер-дневные саммари).
          4) Маркер «день обработан» = строка в log.md `ingest-day | <date>`.
        """
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        raw_file = wiki.root / "raw" / "daily" / f"{date_iso}.md"
        if not raw_file.exists():
            return f"Файл не найден: raw/daily/{date_iso}.md"
        content = raw_file.read_text(encoding="utf-8")
        n_msgs = content.count("\n---\n> **")
        rel = f"raw/daily/{date_iso}.md"

        # Один раз читаем «головной» контекст vault и встраиваем в каждый чанк.
        def _read_or(p: Path, cap: int) -> str:
            try:
                t = p.read_text(encoding="utf-8")
            except OSError:
                return "(нет файла)"
            return t if len(t) <= cap else t[:cap] + "\n...[truncated]"

        agents_md = _read_or(wiki.root / "AGENTS.md", 8_000)
        index_md = _read_or(wiki.root / "index.md", 6_000)
        template_md = _read_or(wiki.root / "wiki" / "projects" / "_template.md", 2_000)
        head_ctx = (
            "=== AGENTS.md ===\n" + agents_md
            + "\n\n=== index.md ===\n" + index_md
            + "\n\n=== wiki/projects/_template.md ===\n" + template_md
        )

        executor = self._make_executor(wiki, cfg)

        # 1+2: Прескан медиа и встраивание описаний в текст.
        media_refs = self._extract_media_refs(content, date_iso)
        log.info(
            "ingest_raw_file: %s — %d msgs, %d media refs to pre-analyze",
            date_iso, n_msgs, len(media_refs),
        )
        media_map = await self._preanalyze_media(media_refs, wiki, executor)
        enriched = self._enrich_with_media(content, media_map)

        # 3: Чанкуем обогащённый контент. Лимит чуть больше т.к. встроены описания.
        chunks = self._split_raw_chunks(enriched, max_msgs=8, max_chars=4_500)

        log.info(
            "ingest_raw_file: %s — split into %d chunk(s), %d media described",
            date_iso, len(chunks), len(media_map),
        )
        total_steps = 0
        try:
            for idx, chunk in enumerate(chunks, 1):
                user_prompt = (
                    f"Ingest чанка {idx}/{len(chunks)} файла `{rel}` "
                    f"(дата {date_iso}, чат {cfg.name}). "
                    f"Содержимое чанка приведено НИЖЕ — НЕ читай весь raw-файл, "
                    f"работай с этим текстом. ВСЕ медиа уже описаны inline "
                    f"(строки `> 🧠 [МЕДИА ...]:`) — НЕ вызывай describe_media. "
                    f"Обнови соответствующие страницы в wiki/entities/, "
                    f"wiki/projects/, wiki/sources/, wiki/concepts/, index.md "
                    f"по правилам AGENTS.md. НЕ создавай wiki/daily/{date_iso}.md "
                    f"— дневные саммари в архитектуре не нужны (маркер ставит "
                    f"оркестратор в log.md). В finish() дай короткую сводку: "
                    f"что нового добавил/обновил.\n\n"
                    f"--- КОНТЕКСТ VAULT (НЕ ЧИТАЙ ЭТИ ФАЙЛЫ ЧЕРЕЗ read_file) ---\n"
                    f"{head_ctx}\n"
                    f"--- КОНЕЦ КОНТЕКСТА ---\n\n"
                    f"--- НАЧАЛО ЧАНКА ---\n{chunk}\n--- КОНЕЦ ЧАНКА ---"
                )
                run = await run_agent(
                    client=self.client,
                    executor=executor,
                    system_prompt=INGEST_SYSTEM,
                    user_prompt=user_prompt,
                    max_steps=12,
                )
                total_steps += run.steps
                if not run.finished:
                    log.warning(
                        "ingest_raw_file: chunk %d/%d for %s did not finish",
                        idx, len(chunks), date_iso,
                    )
            # 4: Маркер обработки — запись в log.md.
            self._append_log(
                wiki,
                f"ingest-day | {date_iso} | {n_msgs} msgs | "
                f"{len(media_map)} media | {len(chunks)} chunks | "
                f"{total_steps} steps | ok",
            )
            if settings.git_autocommit:
                commit_and_push(f"ingest-day {cfg.name} {date_iso}")
            return (
                f"Ingest {date_iso} завершён: {n_msgs} сообщений, "
                f"{len(media_map)} медиа, {len(chunks)} чанк(ов), "
                f"{total_steps} шагов."
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest_raw_file failed")
            return f"Ошибка ingest: {exc}"


    async def ingest_raw_pending(
        self, chat_cfg: ChatConfig | None = None, limit: int = 5
    ) -> str:
        """Инжестировать первые N не-обработанных raw/daily/*.md файлов.

        «Обработанный» определяется по двум критериям (любой):
          * в log.md есть строка `ingest-day | <date> |` (новый маркер);
          * существует wiki/daily/<date>.md (legacy: ранее писали дайджесты).
        """
        import re
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        raw_dir = wiki.root / "raw" / "daily"
        wiki_dir = wiki.root / "wiki" / "daily"
        log_file = wiki.root / "log.md"
        if not raw_dir.exists():
            return "Нет raw/daily/ директории."
        log_text = ""
        if log_file.exists():
            try:
                log_text = log_file.read_text(encoding="utf-8")
            except OSError:
                log_text = ""
        # Маркеры обработки в log.md: поддерживаем три исторических формата:
        #   1) `ingest-day | YYYY-MM-DD |`        — текущий
        #   2) `ingest-file | YYYY-MM-DD |`       — промежуточный
        #   3) `## [YYYY-MM-DD HH:MM] ingest |`   — самый старый (заголовок-дата)
        done_in_log: set[str] = set(
            re.findall(r"ingest-day \| (\d{4}-\d{2}-\d{2}) \|", log_text)
        )
        done_in_log.update(
            re.findall(r"ingest-file \| (\d{4}-\d{2}-\d{2}) \|", log_text)
        )
        done_in_log.update(
            re.findall(r"^## \[(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}\] ingest \|", log_text, re.MULTILINE)
        )
        all_dates = sorted(p.stem for p in raw_dir.glob("*.md"))
        pending = [
            d for d in all_dates
            if d not in done_in_log and not (wiki_dir / f"{d}.md").exists()
        ]
        if not pending:
            return "Все raw/daily/*.md уже обработаны (см. log.md)."
        batch = pending[:limit]
        results: list[str] = []
        for date_iso in batch:
            res = ""
            for attempt in range(1, 4):
                log.info("ingest_raw_pending: %s (attempt %d/3)", date_iso, attempt)
                res = await self.ingest_raw_file(date_iso, cfg)
                if not res.startswith("Ошибка ingest"):
                    break
                log.warning(
                    "ingest_raw_pending: %s attempt %d failed: %s",
                    date_iso, attempt, res[:200],
                )
                await asyncio.sleep(2 ** attempt)  # 2s, 4s, 8s
            results.append(f"**{date_iso}**: {res[:120]}")
        remaining = len(pending) - len(batch)
        summary = "\n".join(results)
        if remaining:
            summary += f"\n\n_Ещё {remaining} дней в очереди. Запусти /ingest-files снова._"
        return summary

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

    async def revise_query(
        self,
        question: str,
        original_answer: str,
        correction: str,
        chat_cfg: ChatConfig | None = None,
    ) -> str:
        """Перегенерировать ответ /ask с учётом правок пользователя.

        Правки = ground truth, перевешивают wiki/git.
        """
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        executor = self._make_executor(wiki, cfg)
        user_prompt = (
            f"ИСХОДНЫЙ ВОПРОС:\n{question}\n\n"
            f"ЧЕРНОВОЙ ОТВЕТ (есть ошибки):\n{original_answer}\n\n"
            f"ПРАВКИ ПОЛЬЗОВАТЕЛЯ (ИСТОЧНИК ПРАВДЫ):\n{correction}\n\n"
            "Перепиши ответ так, чтобы факты соответствовали правкам."
        )
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=REVISE_SYSTEM,
            user_prompt=user_prompt,
            max_steps=8,
        )
        return run.summary or original_answer

    # --- LINT ---

    # Допустимые scope'ы для /lint. Сопоставление: scope → относительная папка.
    LINT_SCOPES: dict[str, str] = {
        # concepts/*
        "audio-voice": "wiki/concepts/audio-voice",
        "chassis": "wiki/concepts/chassis",
        "drive": "wiki/concepts/drive",
        "electronics": "wiki/concepts/electronics",
        "manufacturing": "wiki/concepts/manufacturing",
        "military": "wiki/concepts/military",
        "misc": "wiki/concepts/misc",
        "platforms": "wiki/concepts/platforms",
        "sensors": "wiki/concepts/sensors",
        "software": "wiki/concepts/software",
        # entities/*
        "companies": "wiki/entities/companies",
        "orgs": "wiki/entities/orgs",
        "people": "wiki/entities/people",
        # flat
        "projects": "wiki/projects",
        "sources": "wiki/sources",
    }

    async def _lint_scope(
        self,
        wiki: Wiki,
        cfg: ChatConfig,
        scope: str,
        rel_dir: str,
    ) -> str:
        """Прогон lint-агента по одному скоупу. Возвращает короткий отчёт."""
        executor = self._make_executor(wiki, cfg)
        prompt = (
            f"Скоуп: `{rel_dir}/`.\n"
            f"Сделай health-check ТОЛЬКО файлов из этой папки. "
            f"Верни короткий отчёт по правилам из system-prompt."
        )
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=LINT_SYSTEM,
            user_prompt=prompt,
            max_steps=25,
        )
        body = (run.summary or "").strip() or "(пусто)"
        return f"### {scope} ({run.steps} шагов)\n{body}"

    async def lint(
        self,
        chat_cfg: ChatConfig | None = None,
        scope: str | None = None,
    ) -> str:
        """Health-check вики.

        - `scope=None` или `"all"` — последовательно по всем доменам.
        - `scope="<domain>"` — только указанный домен (см. `LINT_SCOPES`).
        """
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)

        scope_norm = (scope or "all").strip().lower()
        if scope_norm in ("", "all"):
            targets = list(self.LINT_SCOPES.items())
        elif scope_norm in self.LINT_SCOPES:
            targets = [(scope_norm, self.LINT_SCOPES[scope_norm])]
        else:
            available = ", ".join(sorted(self.LINT_SCOPES))
            return (
                f"⚠️ Неизвестный scope `{scope_norm}`.\n"
                f"Доступны: {available}, или `all`."
            )

        total_steps = 0
        parts: list[str] = []
        for name, rel_dir in targets:
            try:
                report = await self._lint_scope(wiki, cfg, name, rel_dir)
                parts.append(report)
            except Exception as exc:  # noqa: BLE001
                log.exception("lint scope %s failed", name)
                parts.append(f"### {name}\n❌ упал: {type(exc).__name__}: {exc}")
            # Грубая оценка шагов для лога: парсим '(N шагов)' из отчёта.
            try:
                last = parts[-1].splitlines()[0]
                if "шагов)" in last:
                    n = int(last.split("(")[1].split(" ")[0])
                    total_steps += n
            except Exception:  # noqa: BLE001
                pass

        self._append_log(wiki, f"lint scope={scope_norm} | ~{total_steps} шагов")
        if settings.git_autocommit:
            commit_and_push(f"lint {cfg.name} scope={scope_norm}")
        return "\n\n".join(parts) or "Lint завершён."

    # --- TRIZ ---

    async def triz(
        self, problem: str, chat_cfg: ChatConfig | None = None
    ) -> str:
        """ТРИЗ-разбор: один прямой запрос к модели, без агент-цикла и vault.

        TRIZ — методология применяется к формулировке проблемы. Поиск по вики
        только сбивает модель и жжёт шаги, поэтому выключен.
        """
        _ = chat_cfg  # сохраняем сигнатуру, но не используем
        if not problem.strip():
            return ("Дай формулировку проблемы текстом после `/triz`, "
                    "либо ответом на сообщение с описанием.")
        messages = [
            {"role": "system", "content": TRIZ_SYSTEM},
            {"role": "user", "content": problem.strip()},
        ]
        try:
            resp = await self.client.chat(messages=messages, tools=None)
            content = resp["choices"][0]["message"].get("content") or ""
        except Exception as exc:  # noqa: BLE001
            log.error("triz: model call failed: %s", exc)
            return f"ТРИЗ-разбор не удался: {exc}"
        return content.strip() or "ТРИЗ-разбор не получился (пустой ответ)."


    # --- PROJECTS ---

    def list_projects(self, chat_cfg: ChatConfig | None = None) -> list[dict[str, Any]]:
        """Список страниц wiki/projects/ с парсингом frontmatter.

        Возвращает [{name, status, updated, sources, summary}], отсортированный
        по updated убыв. Шаблон `_template.md` исключается.
        """
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        pdir = wiki.root / "wiki" / "projects"
        out: list[dict[str, Any]] = []
        if not pdir.exists():
            return out
        for p in sorted(pdir.glob("*.md")):
            if p.name.startswith("_"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            fm: dict[str, str] = {}
            if text.startswith("---\n"):
                end = text.find("\n---", 4)
                if end > 0:
                    for line in text[4:end].splitlines():
                        if ":" in line:
                            k, _, v = line.partition(":")
                            fm[k.strip()] = v.strip().strip('"').strip("'")
            # первая непустая строка после заголовка — однострочное описание
            summary = ""
            body = text[text.find("\n---", 4) + 4 :] if text.startswith("---\n") else text
            for line in body.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                summary = s[:120]
                break
            out.append({
                "name": p.stem,
                "status": fm.get("status", "?"),
                "updated": fm.get("updated", ""),
                "sources": fm.get("sources", "0"),
                "summary": summary,
            })
        out.sort(key=lambda x: x["updated"], reverse=True)
        return out

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
