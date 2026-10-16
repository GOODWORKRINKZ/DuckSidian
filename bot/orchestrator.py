"""Высокоуровневые операции: ingest / query / lint / summary.

Поддержка нескольких чатов: каждый чат имеет свою поддиректорию vault'а
(vault/<chat_name>/) и изолированный Wiki-экземпляр.

ask_user pipeline: агент → bot posts в forum-топик → ждёт ответ → агенту.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataclasses import dataclass

from aiogram import Bot
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from .agent.deepseek import LLMClient
from .agent.loop import run_agent
from .agent.prompts import (
    ENTITY_BUILD_SYSTEM,
    ENTITY_MERGE_SYSTEM,
    INGEST_SYSTEM,
    LINT_SYSTEM,
    PERSONAL_SYSTEM,
    QUERY_SYSTEM,
    REVISE_SYSTEM,
    SUMMARY_SYSTEM,
    TRIZ_SYSTEM,
)
from .agent.loop import MEDIA_TOOLS, UNCONFIRMED_MEDIA_NOTE
from .agent.tools import PERSONAL_WRITE_TOOLS, ToolExecutor, personal_tool_schemas
from .agent.wikitools import sections
from .skills import catalog, load_skills
from .config import ChatConfig, settings
from .db import DB
from .git_sync import commit_and_push
from .media_refs import find_images_marker, find_tail_marker, strip_images_block
from .personal import PersonalSpace, is_user_cfg, share_artifact, share_workspace
from .tg_format import md_to_tg_html, split_tg_chunks
from .topic_manager import ensure_bot_topic
from .wiki import Wiki, WikiPathError

log = logging.getLogger(__name__)

UNCONFIRMED_WRITE_NOTE = "⚠️ в vault ничего не записано"


_WRITE_VERB = {"write_file": "создан/перезаписан", "delete_file": "удалён", "move_file": "перенесён"}


def writes_footer(writes: dict[str, str], limit: int = 8) -> str:
    """Сводка реально изменённых файлов — генерирует код по логу инструментов, а не
    модель, поэтому «записал» в ответе всегда можно сверить."""
    paths = [p for p in writes if p.endswith(".md") and not p.startswith("raw/")]
    if not paths:
        return ""
    items = []
    for p in paths[:limit]:
        verb = _WRITE_VERB.get(writes[p], "")
        items.append(f"`{p}`" + (f" ({verb})" if verb else ""))
    more = f" и ещё {len(paths) - limit}" if len(paths) > limit else ""
    return "✍️ Изменено: " + ", ".join(items) + more


def add_footer(answer: str, footer: str) -> str:
    """Добавить строку до блоков IMAGES:/VOICE: (их хвост бот отрезает перед отправкой)."""
    if not footer:
        return answer
    m = find_tail_marker(answer)
    if not m:
        return f"{answer.rstrip()}\n\n{footer}"
    return f"{answer[:m.start()].rstrip()}\n\n{footer}\n\n{answer[m.start():].lstrip()}"


def mark_unconfirmed_media(answer: str) -> str:
    """Ответ показывает/описывает картинку, которую агент не скачивал и не смотрел:
    блок IMAGES (с выдуманными путями) убираем, в конце — явная пометка."""
    text, _ = strip_images_block(answer)
    return f"{text}\n\n{UNCONFIRMED_MEDIA_NOTE}"


def mark_unconfirmed_write(answer: str) -> str:
    """Пометить ответ, утверждающий запись, которой не было. Пометка — до блоков
    IMAGES:/VOICE: (их хвост бот отрезает от текста перед отправкой)."""
    m = find_tail_marker(answer)
    if not m:
        return f"{answer.rstrip()}\n\n{UNCONFIRMED_WRITE_NOTE}"
    return f"{answer[:m.start()].rstrip()}\n\n{UNCONFIRMED_WRITE_NOTE}\n\n{answer[m.start():].lstrip()}"
tokens_log = logging.getLogger("bot.tokens")


def _chat_wiki(chat_cfg: ChatConfig) -> Wiki:
    """Вернуть Wiki для конкретного чата (vault/<name>/).

    Для личных пространств (users/<id>) — делегируем PersonalSpace, у него
    другой шаблон и набор директорий.
    """
    if is_user_cfg(chat_cfg):
        return PersonalSpace(chat_cfg.chat_id).wiki
    if chat_cfg.name.startswith("shared/"):
        return PersonalSpace.shared(chat_cfg.name[len("shared/"):]).wiki
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


@dataclass
class _RunStat:
    op: str
    ts: str
    steps: int
    tokens_prompt: int
    tokens_completion: int
    tokens_cached: int

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion


class Orchestrator:
    def __init__(self, bot: Bot, db: DB, wiki: Wiki):
        self.bot = bot
        self.db = db
        self.wiki = wiki  # глобальный wiki (для /ask, /search, /lint, /summary)
        self.client = LLMClient()
        self._waiters: dict[int, asyncio.Future[str]] = {}
        # Открытый вопрос ask_choice в личке: (chat_id, thread_id) → qid. Пока он есть,
        # обычное текстовое сообщение в этой теме — ответ на вопрос, а не новый запрос.
        self._private_pending: dict[tuple[int, int | None], int] = {}
        # Темы, где идущий ход уже задавал вопросы (интерактивный диалог): там реплики во время
        # хода идут в него, а не в очередь. Чистит обработчик в конце хода.
        self._asked_in: set[tuple[int, int | None]] = set()
        # Личные пространства (DM): user_id → PersonalSpace
        self._spaces: dict[int, PersonalSpace] = {}
        # Накапливаем статистику токенов с момента старта бота
        self._run_stats: list[_RunStat] = []
        # Per-file locks to prevent duplicate concurrent media analysis
        self._media_locks: dict[str, asyncio.Lock] = {}

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
            return await asyncio.wait_for(fut, timeout=60 * 60)
        except asyncio.TimeoutError:
            return "(timeout: команда не ответила за 60 минут, делай по умолчанию)"
        finally:
            self._waiters.pop(qid, None)

    async def ask_private(
        self, question: str, options: list[str] | None, chat_id: int, thread_id: int | None,
        timeout: float = 15 * 60,
    ) -> str:
        """Вопрос агента пользователю в его личный чат/тему: кнопки-варианты + можно
        ответить текстом. Ждём до timeout, потом агент решает сам."""
        qid = await self.db.add_question(
            batch_id=None, question=question,
            options=json.dumps(options, ensure_ascii=False) if options else None,
        )
        kb = None
        if options:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text=opt[:60], callback_data=f"ans:{qid}:{i}")]
                for i, opt in enumerate(options[:8])
            ])
        text = (
            f"❓ {md_to_tg_html(question)}\n\n"
            f"<i>Выбери вариант или просто напиши свой ответ. id={qid}.</i>"
        )
        sent = await self.bot.send_message(
            chat_id=chat_id, text=text, message_thread_id=thread_id,
            reply_markup=kb, parse_mode="HTML",
        )
        await self.db.attach_question_post(qid, sent.message_id)
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._waiters[qid] = fut
        key = (chat_id, thread_id)
        self._private_pending[key] = qid
        self._asked_in.add(key)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return (f"(нет ответа за {int(timeout // 60)} мин — реши сам, явно пометь это как "
                    "допущение и спроси в итоговом ответе)")
        finally:
            self._waiters.pop(qid, None)
            if self._private_pending.get(key) == qid:
                self._private_pending.pop(key, None)

    def pending_private_question(self, chat_id: int, thread_id: int | None) -> int | None:
        return self._private_pending.get((chat_id, thread_id))

    def turn_is_interactive(self, chat_id: int, thread_id: int | None) -> bool:
        return (chat_id, thread_id) in self._asked_in

    def end_interactive(self, chat_id: int, thread_id: int | None) -> None:
        self._asked_in.discard((chat_id, thread_id))

    async def deliver_answer(self, qid: int, answer: str) -> bool:
        await self.db.answer_question(qid, answer)
        fut = self._waiters.get(qid)
        if fut and not fut.done():
            fut.set_result(answer)
            return True
        return False

    def _make_executor(self, wiki: Wiki, chat_cfg: ChatConfig | None = None,
                       file_cache: dict[str, str] | None = None,
                       team_wikis: dict[str, Wiki] | None = None,
                       share_fn: Any = None, workspace_fn: Any = None,
                       choice_fn: Any = None) -> ToolExecutor:
        async def ask_user_cb(question: str, options: list[str] | None) -> str:
            return await self._ask_user(question, options, chat_cfg)
        return ToolExecutor(wiki, ask_user_cb, deepseek_client=self.client,
                            file_cache=file_cache, db=self.db,
                            team_wikis=team_wikis, share_fn=share_fn,
                            workspace_fn=workspace_fn, choice_fn=choice_fn)

    # --- LLM model switching ---

    async def restore_model(self) -> None:
        """Восстановить выбранный через /model провайдер после рестарта."""
        raw = await self.db.get_state("llm_profile", "")
        if not raw:
            return
        try:
            prof = json.loads(raw)
            self.client.switch(prof["provider"], prof.get("model"))
            log.info("LLM profile restored: %s", self.client.describe())
        except Exception as exc:  # noqa: BLE001
            log.warning("restore_model failed (%s), using .env defaults", exc)

    async def set_model(self, provider: str, model: str | None = None) -> str:
        desc = self.client.switch(provider, model)
        await self.db.set_state(
            "llm_profile",
            json.dumps({"provider": self.client.provider.key, "model": self.client.model}),
        )
        return desc

    def model_info(self) -> str:
        return self.client.describe()

    def _record_run(self, op: str, run: Any) -> None:
        """Сохранить статистику токенов одного запуска агента."""
        from .agent.loop import AgentRun
        if not isinstance(run, AgentRun):
            return
        stat = _RunStat(
            op=op,
            ts=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            steps=run.steps,
            tokens_prompt=run.tokens_prompt,
            tokens_completion=run.tokens_completion,
            tokens_cached=run.tokens_cached,
        )
        self._run_stats.append(stat)
        # Держим последние 50 запусков
        if len(self._run_stats) > 50:
            self._run_stats = self._run_stats[-50:]
        # Цены — из реестра провайдера (config.LLM_PROVIDERS), $/1M токенов.
        run_cost = self.client.cost_usd(
            stat.tokens_prompt, stat.tokens_completion, stat.tokens_cached,
        )
        total_p = sum(s.tokens_prompt for s in self._run_stats)
        total_c = sum(s.tokens_completion for s in self._run_stats)
        total_ca = sum(s.tokens_cached for s in self._run_stats)
        session_cost = self.client.cost_usd(total_p, total_c, total_ca)
        prev = self._run_stats[-2] if len(self._run_stats) >= 2 else None
        delta = f" delta={stat.tokens_total - prev.tokens_total:+d}" if prev else ""
        tokens_log.info(
            "run op=%s steps=%d prompt=%d compl=%d cached=%d total=%d%s "
            "run_cost=$%.5f session_cost=$%.5f session_runs=%d",
            op, stat.steps,
            stat.tokens_prompt, stat.tokens_completion, stat.tokens_cached,
            stat.tokens_total, delta,
            run_cost, session_cost, len(self._run_stats),
        )

    def get_token_stats(self) -> dict:
        """Суммарная статистика токенов с момента старта."""
        total_p = sum(s.tokens_prompt for s in self._run_stats)
        total_c = sum(s.tokens_completion for s in self._run_stats)
        total_ca = sum(s.tokens_cached for s in self._run_stats)
        return {
            "runs": len(self._run_stats),
            "total_prompt": total_p,
            "total_completion": total_c,
            "total_tokens": total_p + total_c,
            "total_cached": total_ca,
            "cost_usd": self.client.cost_usd(total_p, total_c, total_ca),
            "model": self.client.describe(),
            "recent": self._run_stats[-10:],
        }

    # --- PERSONAL MODE (DM) ---

    def personal_space(self, user_id: int, display_name: str | None = None) -> PersonalSpace:
        sp = self._spaces.get(user_id)
        if sp is None:
            sp = PersonalSpace(user_id, display_name)
            self._spaces[user_id] = sp
        elif display_name:
            sp.display_name = display_name
        return sp

    def _team_wikis(self) -> dict[str, Wiki]:
        return {c.name: _chat_wiki(c) for c in settings.get_chats()}

    async def personal_turn(
        self,
        space: PersonalSpace,
        text: str,
        *,
        message_id: int | None = None,
        attached_media: list[str] | None = None,
        history: list[dict[str, str]] | None = None,
        workspace: dict[str, Any] | None = None,
        progress: Any = None,
        workspace_fn: Any = None,
        author_name: str | None = None,
        author_id: int | None = None,
        chat_id: int | None = None,
        thread_id: int | None = None,
        interject: Any = None,
    ) -> str:
        """Один ход диалога в личном режиме: агент классифицирует сообщение
        (запись / поиск / управление артефактами / разговор) и действует.
        author_* — кто написал реплику (в общем vault у space user_id=0);
        chat_id — личка-источник (для якоря raw/inbox в общем vault).
        Возвращает текст ответа пользователю (markdown, возможно с блоком IMAGES:)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        chats = settings.get_chats()
        chats_desc = ", ".join(f"`{c.name}`" for c in chats) or "(нет)"

        async def share_fn(path: str, chat: str, comment: str) -> str:
            return await self.share_to_chat(space, path, chat, comment=comment)

        choice_fn = None
        if chat_id is not None:
            async def choice_fn(question: str, options: list[str] | None) -> str:
                return await self.ask_private(question, options, chat_id, thread_id)

        executor = self._make_executor(
            space.wiki, space.cfg, team_wikis=self._team_wikis(), share_fn=share_fn,
            workspace_fn=workspace_fn, choice_fn=choice_fn,
        )
        skills = load_skills(space.wiki.root)
        skills_block = (f"СКИЛЫ (use_skill(name), когда запрос подходит):\n{catalog(skills)}\n"
                        if skills else "")

        def _read_or(rel: str, cap: int) -> str:
            try:
                t = space.wiki.read_file(rel)
            except Exception:  # noqa: BLE001
                return "(нет файла)"
            return t if len(t) <= cap else t[:cap] + "\n...[truncated]"

        head_ctx = (
            "=== AGENTS.md ===\n" + _read_or("AGENTS.md", 6_000)
            + "\n\n=== index.md ===\n" + _read_or("index.md", 4_000)
        )
        media_block = ""
        if attached_media:
            media_block = (
                "\n\n[ПРИКРЕПЛЕНО — путь в личном vault; при необходимости describe_media:]\n"
                + "\n".join(f"- {m}" for m in attached_media)
            )
        ws_block = ""
        pinned_block = ""
        if workspace:
            slug = workspace.get("project_slug") or ""
            ws_block = (
                f"Текущий воркспейс: «{workspace.get('name')}»"
                + (f" → проект wiki/projects/{slug}.md" if slug else "")
                + ". Всё ниже относится к нему, если не сказано иное.\n"
            )
            pinned = self._project_pinned(space, slug)
            if pinned:
                pinned_block = (
                    f"\n\n=== ЗАКРЕПЛЕНО: wiki/projects/{slug.rsplit('/', 1)[-1]}.md "
                    "(Цель / Текущий статус / Решения — источник правды о проекте) ===\n"
                    f"{pinned}"
                )
        author_name = author_name or space.display_name
        author_id = author_id if author_id is not None else space.user_id
        if space.is_shared:
            who_block = (
                f"Сегодня: {today}. Совместный воркспейс shared/{space.shared_slug} — общий vault "
                "нескольких участников.\n"
                f"Автор этой реплики: {author_name} (user_id={author_id}). Реплики других "
                "участников в истории диалога помечены «[Имя]: …» — не приписывай их автору "
                "этой реплики.\n"
                f"ПРАВИЛО АВТОРСТВА: каждую запись в wiki/projects (и в заметках) помечай автором — "
                f"в конце строки «— {author_name}» (для чужих реплик из истории — их имя).\n"
            )
        else:
            who_block = f"Сегодня: {today}. Владелец: {author_name} (user_id={author_id}).\n"
        anchor = space.msg_anchor(message_id, chat_id if chat_id is not None else author_id)
        user_prompt = (
            f"{who_block}"
            f"Групповые чаты для search_team_wiki/share_to_chat: {chats_desc}.\n"
            f"Бюджет шагов: {settings.personal_agent_max_steps}.\n"
            f"{ws_block}"
            f"{skills_block}"
            f"Источник этой реплики: [[raw/inbox/{today}#{anchor}]].\n\n"
            f"--- КОНТЕКСТ VAULT (не читай эти файлы через read_file) ---\n{head_ctx}{pinned_block}\n"
            f"--- КОНЕЦ КОНТЕКСТА ---\n\n"
            f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{text.strip()}{media_block}"
        )
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=PERSONAL_SYSTEM,
            user_prompt=user_prompt,
            max_steps=settings.personal_agent_max_steps,
            tools=personal_tool_schemas(),
            history=history,
            force_finish_hint=(
                "СТОП. Шаги кончились, инструментов больше нет. Ответь пользователю "
                "обычным текстом: что успел сделать/найти и что осталось."
            ),
            on_step=progress,
            interject=interject,
            write_tools=PERSONAL_WRITE_TOOLS,
            media_tools=MEDIA_TOOLS,
        )
        self._record_run(f"personal/{space.user_id}", run)
        if settings.git_autocommit:
            commit_and_push(f"personal {space.user_id}: turn")
        answer = run.summary or "Не понял, переформулируй."
        if run.write_nudged or run.unconfirmed_write:
            log.warning(
                "personal: answer claimed a write without write tools — user=%s (id=%s) "
                "space=%s recovered=%s tools=%s head=%r",
                author_name, author_id, space.cfg.name, not run.unconfirmed_write,
                run.tools_called, answer[:200],
            )
        if run.media_nudged or run.unconfirmed_media:
            log.warning(
                "personal: answer showed/described media it never fetched — user=%s (id=%s) "
                "space=%s recovered=%s tools=%s head=%r",
                author_name, author_id, space.cfg.name, not run.unconfirmed_media,
                run.tools_called, answer[:200],
            )
        if run.unconfirmed_media:
            answer = mark_unconfirmed_media(answer)
        if run.unconfirmed_write:
            answer = mark_unconfirmed_write(answer)
        else:
            answer = add_footer(answer, writes_footer(run.writes))
        return answer

    @staticmethod
    def _project_pinned(space: Any, slug: str, cap: int = 4_000) -> str:
        """Цель / Текущий статус / Решения страницы проекта воркспейса — в каждый ход,
        чтобы ключевые факты не вымывались из окна истории диалога."""
        name = (slug or "").rsplit("/", 1)[-1]
        if not name:
            return ""
        try:
            text = space.wiki.read_file(f"wiki/projects/{name}.md", max_bytes=2_000_000)
        except Exception:  # noqa: BLE001
            return ""
        return sections(text, ["Цель", "Текущий статус", "Статус", "Решения"], cap)

    async def summarize_for_voice(self, text: str, max_chars: int | None = None) -> str:
        """Сжать ответ в короткий устный пересказ для TTS: 1–3 фразы, суть, без
        markdown, ссылок, путей и перечислений. Короткий текст возвращается как есть."""
        from .media_ai import _md_for_tts  # noqa: PLC0415
        limit = max_chars or settings.voice_reply_max_chars
        plain = _md_for_tts(text)
        if len(plain) <= limit:
            return plain
        sys_prompt = (
            "Ты озвучиваешь ответ ассистента голосом. Перескажи СУТЬ ответа в 1–3 коротких "
            f"разговорных фразах (до {limit} символов), по-русски, от первого лица, как сказал бы "
            "человек вслух. Без markdown, без ссылок, без путей к файлам, без списков и цифр "
            "в скобках. Если в ответе был вопрос пользователю — обязательно задай его. "
            "Верни только текст для озвучки."
        )
        try:
            resp = await self.client.chat(
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": plain[:6000]}],
                tools=None, temperature=0.3,
            )
            out = (resp["choices"][0]["message"].get("content") or "").strip()
            from .agent.loop import _sanitize_summary  # noqa: PLC0415
            out = _md_for_tts(_sanitize_summary(out))
            if out:
                return out[: limit * 2]
        except Exception as exc:  # noqa: BLE001
            log.warning("summarize_for_voice failed: %s", exc)
        cut = plain[:limit]
        return cut.rsplit(". ", 1)[0] + "." if ". " in cut else cut

    async def share_to_chat(
        self,
        space: PersonalSpace,
        rel: str,
        chat_name: str,
        *,
        comment: str = "",
        author: str | None = None,
    ) -> str:
        """Поделиться артефактом личного vault с групповым чатом.

        1) Копия в vault группы (raw/notes/ + raw/assets/shared/).
        2) Запись в БД как сообщение группы — попадёт в ближайший ingest.
        3) Анонс в Telegram (бот-топик группы) + файл, если бинарник.
        Возвращает человекочитаемый результат (для агента и для /share).
        """
        target = settings.get_chat_by_name(chat_name.strip().lstrip("#"))
        if target is None:
            avail = ", ".join(c.name for c in settings.get_chats())
            return f"ERROR: чат '{chat_name}' не найден. Доступны: {avail}"
        resolved = space.resolve_artifact(rel)
        if not resolved:
            return f"ERROR: артефакт '{rel}' не найден в личном vault"
        author = author or space.display_name
        try:
            res = share_artifact(space, resolved, target, author=author, comment=comment)
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("share_artifact failed")
            return f"ERROR: {exc}"

        now = datetime.now(timezone.utc)
        # 2) БД: синтетическое сообщение для ingest (как /note).
        synthetic_id = -int(now.timestamp() * 1000)
        db_text = f"📤 shared by {author}: {res.title}\n\n{res.body_md[:3000]}"
        try:
            await self.db.save_message(
                chat_id=target.chat_id, message_id=synthetic_id, topic_id=None,
                user_id=space.user_id, username=None, full_name=author,
                text=db_text, media_type="shared", media_path=res.dst_note,
                reply_to_message_id=None, ts=now,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("share: db.save_message failed: %s", exc)

        # 3) Telegram-анонс в бот-топик группы.
        posted_id: int | None = None
        try:
            topic_id = await ensure_bot_topic(self.bot, self.db, target.chat_id)
            header = f"📤 **{author}** поделился: **{res.title}**\n`{res.dst_note}`\n\n"
            html = md_to_tg_html(header + res.body_md[:3500], tg_chat_id=target.chat_id)
            chunks = split_tg_chunks(html)
            for chunk in chunks[:3]:
                sent = await self.bot.send_message(
                    chat_id=target.chat_id, text=chunk,
                    message_thread_id=topic_id, parse_mode="HTML",
                )
                posted_id = posted_id or sent.message_id
            if res.is_binary and res.copied_assets:
                from aiogram.types import FSInputFile  # noqa: PLC0415
                f_abs = _chat_wiki(target).root / res.copied_assets[0]
                if f_abs.is_file() and f_abs.stat().st_size <= 50 * 1024 * 1024:
                    ext = f_abs.suffix.lower()
                    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                        await self.bot.send_photo(target.chat_id, FSInputFile(f_abs),
                                                  message_thread_id=topic_id)
                    else:
                        await self.bot.send_document(target.chat_id, FSInputFile(f_abs),
                                                     message_thread_id=topic_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("share: telegram post failed: %s", exc)

        try:
            await self.db.add_share(space.user_id, resolved, target.chat_id,
                                    res.dst_note, posted_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("share: db.add_share failed: %s", exc)
        self._append_log(_chat_wiki(target),
                         f"share | from users/{space.user_id}/{resolved} → {res.dst_note}")
        if settings.git_autocommit:
            commit_and_push(f"share {target.name}: {res.dst_note}")
        assets = f", файлов: {len(res.copied_assets)}" if res.copied_assets else ""
        posted = " и анонсировано в Telegram" if posted_id else ""
        return (
            f"OK: «{res.title}» → чат {target.name}: `{res.dst_note}`{assets}{posted}. "
            f"Попадёт в wiki чата при ближайшем ingest."
        )

    async def share_workspace_to_chat(
        self,
        space: PersonalSpace,
        slug: str,
        chat_name: str,
        *,
        comment: str = "",
        author: str | None = None,
        as_topic: bool = False,
    ) -> str:
        """Поделиться целым воркспейсом (проект + связанные заметки/подборки/файлы).

        Каждая страница ложится отдельной заметкой в raw/notes/ группы и отдельным
        сообщением в БД (для ingest). Анонс один: в бот-топик, либо (as_topic) в
        отдельную тему группы с именем воркспейса — «вкладка в вкладку».
        """
        target = settings.get_chat_by_name(chat_name.strip().lstrip("#"))
        if target is None:
            avail = ", ".join(c.name for c in settings.get_chats())
            return f"ERROR: чат '{chat_name}' не найден. Доступны: {avail}"
        author = author or space.display_name
        try:
            res = share_workspace(space, slug, target, author=author, comment=comment)
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("share_workspace failed")
            return f"ERROR: {exc}"

        now = datetime.now(timezone.utc)
        base_id = -int(now.timestamp() * 1000)
        for i, page in enumerate(res.pages):
            try:
                await self.db.save_message(
                    chat_id=target.chat_id, message_id=base_id - i, topic_id=None,
                    user_id=space.user_id, username=None, full_name=author,
                    text=f"📤 workspace «{res.title}» / {page.title} (shared by {author})\n\n{page.body_md[:3000]}",
                    media_type="shared", media_path=page.dst_note,
                    reply_to_message_id=None, ts=now,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("share_workspace: db.save_message failed: %s", exc)

        posted_id: int | None = None
        thread_id: int | None = None
        try:
            if as_topic:
                key = f"ws_topic_{target.chat_id}_{slug}"
                stored = await self.db.get_state(key, "")
                if stored:
                    thread_id = int(stored)
                else:
                    topic = await self.bot.create_forum_topic(chat_id=target.chat_id, name=res.title[:128])
                    thread_id = topic.message_thread_id
                    await self.db.set_state(key, str(thread_id))
                    await self.db.save_topic(target.chat_id, thread_id, res.title[:128])
            else:
                thread_id = await ensure_bot_topic(self.bot, self.db, target.chat_id)
            html = md_to_tg_html(res.summary_md, tg_chat_id=target.chat_id)
            for chunk in split_tg_chunks(html)[:3]:
                sent = await self.bot.send_message(
                    chat_id=target.chat_id, text=chunk,
                    message_thread_id=thread_id, parse_mode="HTML",
                )
                posted_id = posted_id or sent.message_id
        except Exception as exc:  # noqa: BLE001
            log.warning("share_workspace: telegram post failed: %s", exc)

        try:
            await self.db.add_share(space.user_id, f"workspace:{slug}", target.chat_id,
                                    res.pages[0].dst_note if res.pages else "", posted_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("share_workspace: db.add_share failed: %s", exc)
        self._append_log(_chat_wiki(target),
                         f"share-workspace | users/{space.user_id}:{slug} | {len(res.pages)} pages | "
                         f"{len(res.copied_assets)} assets")
        if settings.git_autocommit:
            commit_and_push(f"share-workspace {target.name}: {slug}")
        where = (f"в тему «{res.title}»" if as_topic and thread_id else "в бот-топик") if posted_id else "(анонс не отправлен)"
        return (
            f"OK: воркспейс «{res.title}» → чат {target.name}: {len(res.pages)} страниц(ы), "
            f"{len(res.copied_assets)} файл(ов), анонс {where}. Попадёт в wiki чата при ближайшем ingest."
        )

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

        topics = await self.db.get_topics(cfg.chat_id)
        rel = wiki.write_daily_raw(date_iso, msgs, topics=topics)
        batch_id = await self.db.create_batch(date_iso)
        await self.db.mark_processed([m["id"] for m in msgs], batch_id)

        executor = self._make_executor(wiki, cfg)

        # Предварительный анализ медиа — кэшируем в .cache/media/ чтобы
        # агент не запускал STT/vision повторно при следующих ingest.
        content = wiki.read_file(rel)
        media_refs = self._extract_media_refs(content, date_iso)
        if media_refs:
            log.info("ingest_for_date: pre-analyzing %d media refs for %s", len(media_refs), date_iso)
            await self._preanalyze_media(media_refs, wiki, executor)

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
                max_steps=50,
            )
            self._record_run(f"ingest/{date_iso}", run)
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
        topics = await self.db.get_topics(cfg.chat_id)
        rel = wiki.write_daily_raw(date_iso, msgs, topics=topics)
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
            max_steps=50,
        )
        self._record_run(f"ingest-adhoc/{cfg.name}", run)
        await self.db.finish_batch(
            batch_id, "ok" if run.finished else "partial", len(msgs)
        )
        self._append_log(wiki, f"ingest-adhoc | last {n} | {run.steps} шагов")
        if settings.git_autocommit:
            commit_and_push(f"ingest adhoc {cfg.name}: last {n} msgs")
        return run.summary or "Ingest завершён."

    @staticmethod
    def _extract_media_refs(content: str, date_iso: str) -> list[str]:
        """Достать все wiki-ссылки на медиа дня: [[raw/assets/<date>/<file>]].

        Поддерживает как bare [[...]], так и `[[...]]` (backtick-обёртка в raw daily).
        """
        import re
        pat = re.compile(
            r"`?\[\[(raw/assets/" + re.escape(date_iso) + r"/[^\]\r\n]+)\]\]`?"
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
            if rel not in self._media_locks:
                self._media_locks[rel] = asyncio.Lock()
            async with self._media_locks[rel]:
                if cache_file.exists():
                    log.info("preanalyze_media cache hit: %s", rel)
                    desc = cache_file.read_text(encoding="utf-8")
                else:
                    log.info("preanalyze_media start: %s", rel)
                    try:
                        desc = await asyncio.wait_for(
                            executor._describe_media(rel), timeout=300.0
                        )
                    except asyncio.TimeoutError:
                        log.warning("preanalyze_media timeout (>300s): %s", rel)
                        desc = "[таймаут анализа медиа]"
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
                # head_ctx (AGENTS.md + index.md + template) шлём только в первый чанк —
                # в остальных агент уже знает структуру vault'а из кэша.
                ctx_section = (
                    f"--- КОНТЕКСТ VAULT (НЕ ЧИТАЙ ЭТИ ФАЙЛЫ ЧЕРЕЗ read_file) ---\n"
                    f"{head_ctx}\n"
                    f"--- КОНЕЦ КОНТЕКСТА ---\n\n"
                ) if idx == 1 else ""
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
                    f"{ctx_section}"
                    f"--- НАЧАЛО ЧАНКА ---\n{chunk}\n--- КОНЕЦ ЧАНКА ---"
                )
                run = await run_agent(
                    client=self.client,
                    executor=executor,
                    system_prompt=INGEST_SYSTEM,
                    user_prompt=user_prompt,
                    max_steps=12,
                )
                self._record_run(f"ingest-raw/{date_iso}/chunk{idx}", run)
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
        #   1) `ingest-day | YYYY-MM-DD |`               — legacy (только дата)
        #   1b) `ingest-day | YYYY-MM-DD_adhoc-XXXX |`   — текущий (полный стем)
        #   2) `ingest-file | YYYY-MM-DD |`              — промежуточный
        #   3) `## [YYYY-MM-DD HH:MM] ingest |`          — самый старый (заголовок-дата)
        done_in_log: set[str] = set(
            re.findall(r"ingest-day \| ([\w.-]+) \|", log_text)
        )
        done_in_log.update(
            re.findall(r"ingest-file \| ([\w.-]+) \|", log_text)
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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        user_prompt = f"Сегодня: {today}.\n\n{question}"
        run = await run_agent(
            client=self.client,
            executor=executor,
            system_prompt=QUERY_SYSTEM,
            user_prompt=user_prompt,
            max_steps=50,
        )
        self._record_run("query", run)
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
        self._record_run("revise-query", run)
        return run.summary or original_answer

    # --- ENTITY BUILD ---

    async def build_entity(
        self,
        entity_name: str,
        chat_cfg: ChatConfig | None = None,
    ) -> str:
        """Собрать/обновить wiki-страницу сущности по всем сырым данным.

        Агент сам ищет упоминания через search_wiki + grep_file, определяет тип
        (person/project/concept/...) и создаёт или дополняет страницу.
        """
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        executor = self._make_executor(wiki, cfg)
        today = datetime.now(timezone.utc).date().isoformat()
        user_prompt = (
            f"Построй или обнови wiki-страницу сущности **{entity_name}**.\n"
            f"Дата сегодня: {today}. Чат: {cfg.name}.\n"
            f"Следуй алгоритму из system-prompt: сначала search_wiki, затем "
            f"grep по raw-файлам, затем создай/обнови страницу."
        )
        try:
            run = await run_agent(
                client=self.client,
                executor=executor,
                system_prompt=ENTITY_BUILD_SYSTEM,
                user_prompt=user_prompt,
                max_steps=20,
            )
            self._record_run(f"build-entity/{entity_name}", run)
            status = "ok" if run.finished else "partial"
            self._append_log(
                wiki,
                f"build-entity | {entity_name} | {run.steps} шагов | {status}",
            )
            if settings.git_autocommit:
                commit_and_push(f"build-entity {cfg.name}: {entity_name}")
            return run.summary or "Готово."
        except Exception as exc:  # noqa: BLE001
            log.exception("build_entity failed for %r", entity_name)
            return f"Ошибка build-entity: {exc}"

    # --- ENTITY MERGE ---

    async def merge_entities(
        self,
        canonical: str,
        alias: str,
        chat_cfg: ChatConfig | None = None,
    ) -> str:
        """Слить страницу псевдонима в каноническую.

        canonical — главное имя/путь, alias — дубль/псевдоним.
        Агент читает обе страницы, переносит уникальные факты в каноническую,
        заменяет страницу псевдонима редиректом и обновляет wikilinks.
        """
        cfg = chat_cfg or settings.get_chats()[0]
        wiki = _chat_wiki(cfg)
        executor = self._make_executor(wiki, cfg)
        today = datetime.now(timezone.utc).date().isoformat()
        user_prompt = (
            f"Слей две сущности в одну:\n"
            f"- **Каноническая** (сохраняем): `{canonical}`\n"
            f"- **Псевдоним/дубль** (растворяем): `{alias}`\n\n"
            f"Дата сегодня: {today}. Чат: {cfg.name}.\n"
            f"Следуй алгоритму из system-prompt."
        )
        try:
            run = await run_agent(
                client=self.client,
                executor=executor,
                system_prompt=ENTITY_MERGE_SYSTEM,
                user_prompt=user_prompt,
                max_steps=50,
            )
            self._record_run(f"merge-entity/{alias}→{canonical}", run)
            status = "ok" if run.finished else "partial"
            self._append_log(
                wiki,
                f"merge-entity | {alias} → {canonical} | {run.steps} шагов | {status}",
            )
            if settings.git_autocommit:
                commit_and_push(f"merge-entity {cfg.name}: {alias} → {canonical}")
            return run.summary or "Готово."
        except Exception as exc:  # noqa: BLE001
            log.exception("merge_entities failed: %r → %r", alias, canonical)
            return f"Ошибка merge-entity: {exc}"

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
        self._record_run(f"lint/{scope}", run)
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
        self._record_run("summary", run)
        return run.summary or "Сводка не сгенерирована."

    # --- helpers ---

    def _append_log(self, wiki: Wiki, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            wiki.append_file("log.md", f"## [{ts}] {line}")
        except Exception as exc:  # noqa: BLE001
            log.warning("append log failed: %s", exc)
