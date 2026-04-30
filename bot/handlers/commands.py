"""Команды бота: /ask /ingest /lint /summary /note /search /log /pause /resume /help /chats."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)

from ..config import ChatConfig, settings
from ..db import DB
from ..orchestrator import Orchestrator, _chat_wiki
from ..tg_format import md_to_tg_html, split_tg_chunks
from ..topic_manager import ensure_bot_topic, reset_bot_topic
from ..wiki import Wiki

log = logging.getLogger(__name__)

router = Router(name="commands")


# Кэш ответов /ask, ожидающих кнопки "Сохранить".
# Ключ — короткий id (пихается в callback_data, лимит 64 байта),
# значение — (question, answer_markdown, chat_id). Хранится в памяти процесса:
# после рестарта бота кнопки протухнут — это ок, они одноразовые.
_pending_saves: dict[str, tuple[str, str, int]] = {}
_PENDING_SAVES_MAX = 200


# Кэш ответов /ask, ожидающих пользовательских правок (кнопка ✏️ Поправить).
# Ключ — тот же save_id, что и в _pending_saves, чтобы при нажатии «Поправить»
# можно было отдать пользователю исходный (q, answer) и попросить написать
# правки реплаем. После того как мы прислали бот-сообщение с просьбой,
# его message_id записывается в _fix_prompt_to_save_id, чтобы reply-handler
# нашёл нужную запись.
_pending_fixes: dict[str, tuple[str, str, int]] = {}
_fix_prompt_to_save_id: dict[int, str] = {}
_PENDING_FIXES_MAX = 200


def _trim_pending_fixes() -> None:
    if len(_pending_fixes) <= _PENDING_FIXES_MAX:
        return
    for key in list(_pending_fixes.keys())[: len(_pending_fixes) - _PENDING_FIXES_MAX]:
        _pending_fixes.pop(key, None)
    # Подчистим обратный индекс от ссылок на удалённые save_id.
    stale = [mid for mid, sid in _fix_prompt_to_save_id.items() if sid not in _pending_fixes]
    for mid in stale:
        _fix_prompt_to_save_id.pop(mid, None)


def _trim_pending_saves() -> None:
    """Грубая защита от утечки: держим не больше _PENDING_SAVES_MAX записей."""
    if len(_pending_saves) <= _PENDING_SAVES_MAX:
        return
    # Удаляем самые старые (dict сохраняет порядок вставки).
    for key in list(_pending_saves.keys())[: len(_pending_saves) - _PENDING_SAVES_MAX]:
        _pending_saves.pop(key, None)


def _slugify_query(text: str, max_len: int = 40) -> str:
    """Сделать slug для имени файла из вопроса. Кириллицу сохраняем."""
    s = (text or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    if not s:
        s = "query"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


# Расширения, которые Telegram умеет показывать как фото в media group.
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _extract_image_refs(answer: str, wiki_root, max_n: int = 10) -> list:
    """Найти картинки в финальном блоке IMAGES: ответа агента.

    Формат, который агент обязан использовать (см. QUERY_SYSTEM):

        ...текст ответа...

        IMAGES:
        raw/assets/2026-04-28/photo_1.jpg
        raw/assets/2026-04-28/photo_2.png

    Любые `raw/assets/...` упоминания вне этого блока игнорируются —
    они часто оказываются левыми (агент копирует из wiki-источников).
    Файл должен реально существовать в vault'е чата; jpg/jpeg/png/webp.
    """
    if not answer:
        return []
    # Берём всё после последнего "IMAGES:" (case-insensitive), до конца.
    m = re.search(r"(?im)^\s*IMAGES\s*:\s*$", answer)
    if not m:
        return []
    tail = answer[m.end():]
    rels: list[str] = []
    seen: set[str] = set()
    line_re = re.compile(r"raw/assets/[^\s\)\]\}\"'<>|]+", re.UNICODE)
    for raw_line in tail.splitlines():
        line = raw_line.strip().lstrip("-*•").strip().strip("`").strip()
        if not line:
            # пустая строка завершает блок IMAGES
            if rels:
                break
            continue
        # Если строка явно не путь — выходим (блок закончился).
        lm = line_re.search(line)
        if not lm:
            break
        rel = lm.group(0).rstrip(".,;:!?")
        ext = rel.lower().rsplit(".", 1)
        if len(ext) != 2 or "." + ext[1] not in _IMG_EXTS:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        try:
            target = (wiki_root / rel).resolve()
            target.relative_to(wiki_root)  # safety
        except Exception:  # noqa: BLE001
            continue
        if not target.is_file():
            continue
        rels.append(str(target))
        if len(rels) >= max_n:
            break
    return rels


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
    """Отправить markdown-ответ агента в Telegram чанками.

    Устойчив к:
    - удалённому исходному сообщению (msg.reply → падает в msg.answer);
    - кратковременным сетевым проблемам (ретрай 3х).
    Если всё равно не отправилось — логируем весь текст (чтобы не терять
    результат долгих агентных прогонов).
    """
    import asyncio
    from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError

    html_text = md_to_tg_html(markdown_text or "(пусто)", tg_chat_id=msg.chat.id)
    chunks = split_tg_chunks(html_text)

    async def _send_chunk(chunk: str, use_reply: bool) -> bool:
        backoffs = [0, 2, 5]
        for i, delay in enumerate(backoffs):
            if delay:
                await asyncio.sleep(delay)
            try:
                if use_reply:
                    await msg.reply(chunk, parse_mode="HTML")
                else:
                    await msg.answer(chunk, parse_mode="HTML")
                return True
            except TelegramBadRequest as exc:
                # Исходное сообщение удалили — переключаемся на answer.
                if use_reply and "reply" in str(exc).lower():
                    log.warning("reply target gone, falling back to answer: %s", exc)
                    use_reply = False
                    continue
                log.error("send chunk failed (BadRequest): %s", exc)
                return False
            except TelegramNetworkError as exc:
                log.warning("send chunk net err (try %d/%d): %s", i + 1, len(backoffs), exc)
        return False

    use_reply = True
    for chunk in chunks:
        ok = await _send_chunk(chunk, use_reply=use_reply)
        if not ok:
            log.error("chunk delivery failed; full agent answer follows:\n%s", markdown_text)
            break
        # После первого успеха или фолбэка — остальные чанки простыми answer.
        use_reply = False


HELP_TEXT = (
    "<b>DuckSidian — команды</b>\n\n"
    "/ask &lt;вопрос&gt; — спросить wiki\n"
    "/triz [проблема] — ТРИЗ-разбор и идеи по обсуждениям\n"
    "/projects — список проектов в вики\n"
    "/search &lt;строка&gt; — поиск по vault\n"
    "/summary day|week — сводка\n"
    "/log — последние записи log.md\n"
    "/note &lt;текст&gt; — ручная заметка в raw/notes/\n"
    "/chats — список отслеживаемых чатов\n"
    "/build &lt;имя&gt; [&lt;chat&gt;] — собрать/обновить страницу сущности из всех raw-данных <i>(admin)</i>\n"
    "/merge &lt;канонич&gt; | &lt;псевдоним&gt; [&lt;chat&gt;] — слить дубли в одну страницу <i>(admin)</i>\n"
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
            # Отрезаем технический блок IMAGES: — пользователю он не нужен
            # (картинки прикрепим отдельно через _extract_image_refs).
            display = answer or ""
            _img_marker = re.search(r"(?im)^\s*IMAGES\s*:\s*$", display)
            if _img_marker:
                display = display[: _img_marker.start()].rstrip()
            await _reply_html(msg, display or "(пусто)")
            # Прикрепить картинки из ответа (если агент сослался на raw/assets/*.jpg|png|...)
            try:
                chat_wiki = _wiki_for_msg(msg)
                img_paths = _extract_image_refs(answer or "", chat_wiki.root)
                if img_paths:
                    if len(img_paths) == 1:
                        await msg.answer_photo(FSInputFile(img_paths[0]))
                    else:
                        media = [InputMediaPhoto(media=FSInputFile(p)) for p in img_paths[:10]]
                        await msg.answer_media_group(media)
            except Exception as exc:  # noqa: BLE001
                log.warning("ask: attach images failed: %s: %r", type(exc).__name__, exc)
            # Кнопка "Сохранить" — кладём ответ в in-memory кэш под коротким id
            # и шлём отдельное сообщение с inline-клавиатурой. callback_data
            # ограничено 64 байтами, поэтому нельзя пихать туда сам текст.
            if answer:
                save_id = uuid.uuid4().hex[:12]
                _pending_saves[save_id] = (q, answer, msg.chat.id)
                _pending_fixes[save_id] = (q, answer, msg.chat.id)
                _trim_pending_saves()
                _trim_pending_fixes()
                kb = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="💾 Сохранить",
                        callback_data=f"qsave:{save_id}",
                    ),
                    InlineKeyboardButton(
                        text="✏️ Поправить",
                        callback_data=f"qfix:{save_id}",
                    ),
                ]])
                try:
                    await msg.answer(
                        "Сохранить как заметку или поправить ответ?",
                        reply_markup=kb,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("ask: send save-button failed: %s", exc)
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

    @router.message(Command("triz"))
    async def cmd_triz(msg: Message, command: CommandObject) -> None:
        problem = (command.args or "").strip()
        # Если /triz сделан reply'ем — добавим к проблеме автора и текст того сообщения.
        if msg.reply_to_message:
            r = msg.reply_to_message
            quoted = (r.text or r.caption or "").strip()
            if quoted:
                author = (
                    r.from_user.full_name if r.from_user else "anon"
                ) if r.from_user else "anon"
                quoted_block = f"[{author}]: {quoted}"
                problem = f"{problem}\n\n{quoted_block}".strip() if problem else quoted_block
        cfg = _chat_for_msg(msg)
        status_msg = await msg.reply("🧠 ТРИЗ-разбор запущен, жди минуту…")
        try:
            out = await orch.triz(problem, cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("triz failed: %s", exc)
            await msg.reply("⚠️ Ошибка ТРИЗ-агента.")
            return
        await _reply_html(msg, out or "(пусто)")
        # Удаляем служебное «запущен…» — разбор уже в чате.
        try:
            await status_msg.delete()
        except Exception as exc:  # noqa: BLE001
            log.debug("triz: failed to delete status message: %s", exc)

    @router.message(Command("projects"))
    async def cmd_projects(msg: Message) -> None:
        cfg = _chat_for_msg(msg) or settings.get_chats()[0]
        items = orch.list_projects(cfg)
        if not items:
            await msg.reply(
                "Пока ни одного проекта не опознано. "
                "Запусти /ingest — агент заведёт страницы в wiki/projects/."
            )
            return
        emoji = {"active": "🟢", "paused": "🟡", "done": "✅", "archive": "📁"}
        lines = [f"<b>Проекты [{cfg.name}]:</b>"]
        for it in items:
            mark = emoji.get(it["status"], "⚪️")
            upd = f" <i>upd {it['updated']}</i>" if it["updated"] else ""
            src = f" · src={it['sources']}" if it["sources"] not in ("", "0") else ""
            from html import escape as _esc
            summary = _esc(it["summary"]) if it["summary"] else ""
            lines.append(
                f"{mark} <b>{_esc(it['name'])}</b>{upd}{src}\n {summary}"
            )
        text = "\n\n".join(lines)
        await msg.reply(text[:4000], parse_mode="HTML")

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

    @router.message(Command("build"))
    async def cmd_build(msg: Message, command: CommandObject) -> None:
        """/build <имя> [<chat>] — собрать/обновить страницу сущности из всех raw-данных."""
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        args = (command.args or "").strip()
        if not args:
            await msg.reply(
                "Использование: /build &lt;имя сущности&gt; [chat_name]\n"
                "Пример: /build БиБа\n"
                "Пример: /build «Мастер дрон» main",
                parse_mode="HTML",
            )
            return
        # Разбираем: последний токен — необязательный chat_name (без пробелов)
        parts = args.rsplit(None, 1)
        chats = settings.get_chats()
        chat_name_filter: str | None = None
        entity_name = args
        if len(parts) == 2:
            maybe_chat = parts[1]
            matched = [c for c in chats if c.name == maybe_chat]
            if matched:
                chat_name_filter = maybe_chat
                entity_name = parts[0]
                chats = matched
        if chat_name_filter is None:
            # берём чат сообщения или первый
            cfg = _chat_for_msg(msg) or chats[0]
            chats = [cfg]
        status_msg = await msg.reply(
            f"⏳ build-entity <b>{entity_name}</b>…",
            parse_mode="HTML",
        )
        results = []
        for chat_cfg in chats:
            out = await orch.build_entity(entity_name, chat_cfg)
            results.append(f"[{chat_cfg.name}] {out}")
        await _reply_html(msg, "✅ " + "\n".join(results))
        try:
            await status_msg.delete()
        except Exception as exc:  # noqa: BLE001
            log.debug("build: failed to delete status message: %s", exc)

    @router.message(Command("merge"))
    async def cmd_merge(msg: Message, command: CommandObject) -> None:
        """/merge <каноническая> | <псевдоним> [<chat>]

        Слить дубль/псевдоним в каноническую страницу.
        Разделитель между именами — символ |.
        Пример: /merge Мастер-Дрон | Денчик
        Пример: /merge SIRIUSROVER | Сириус-Ровер main
        """
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        args = (command.args or "").strip()
        if "|" not in args:
            await msg.reply(
                "Использование: /merge &lt;каноническая&gt; | &lt;псевдоним&gt; [chat_name]\n"
                "Пример: /merge Мастер-Дрон | Денчик\n"
                "Разделитель — символ <code>|</code>",
                parse_mode="HTML",
            )
            return
        raw_left, raw_right = args.split("|", 1)
        # Последний токен правой части — необязательный chat_name
        right_parts = raw_right.strip().rsplit(None, 1)
        chats = settings.get_chats()
        chat_name_filter: str | None = None
        alias = raw_right.strip()
        if len(right_parts) == 2:
            maybe_chat = right_parts[1]
            matched = [c for c in chats if c.name == maybe_chat]
            if matched:
                chat_name_filter = maybe_chat
                alias = right_parts[0].strip()
                chats = matched
        canonical = raw_left.strip()
        if not canonical or not alias:
            await msg.reply("Укажи оба имени: /merge &lt;каноническая&gt; | &lt;псевдоним&gt;", parse_mode="HTML")
            return
        if chat_name_filter is None:
            cfg = _chat_for_msg(msg) or chats[0]
            chats = [cfg]
        status_msg = await msg.reply(
            f"⏳ merge-entity: <b>{alias}</b> → <b>{canonical}</b>…",
            parse_mode="HTML",
        )
        results = []
        for chat_cfg in chats:
            out = await orch.merge_entities(canonical, alias, chat_cfg)
            results.append(f"[{chat_cfg.name}] {out}")
        await _reply_html(msg, "✅ " + "\n".join(results))
        try:
            await status_msg.delete()
        except Exception as exc:  # noqa: BLE001
            log.debug("merge: failed to delete status message: %s", exc)

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
        status_msg = await msg.reply("⏳ ingest-files запущен…")
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
        try:
            await status_msg.delete()
        except Exception as exc:  # noqa: BLE001
            log.debug("ingest-files: failed to delete status message: %s", exc)

    @router.message(Command("lint"))
    async def cmd_lint(msg: Message, command: CommandObject) -> None:
        if not _is_admin(msg.from_user.id if msg.from_user else None):
            await msg.reply("⛔ только для админов")
            return
        scope = (command.args or "").strip() or None
        label = scope or "all"
        await msg.reply(f"⏳ lint scope={label}…")
        try:
            out = await orch.lint(scope=scope)
        except Exception as exc:  # noqa: BLE001
            log.exception("lint failed")
            await msg.reply(f"❌ lint упал: {type(exc).__name__}: {exc}")
            return
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

    @router.callback_query(F.data.startswith("qsave:"))
    async def cb_save_query(cb: CallbackQuery) -> None:
        """Сохранить ответ /ask в wiki/queries/<ts>-<slug>.md."""
        save_id = (cb.data or "").split(":", 1)[1] if cb.data else ""
        entry = _pending_saves.pop(save_id, None)
        if not entry:
            await cb.answer("⌛ устарело, спроси заново", show_alert=False)
            return
        question, answer, chat_id = entry
        # Per-chat wiki — сохраняем в тот же vault, к которому привязан чат.
        cfg: ChatConfig | None = None
        for c in settings.get_chats():
            if c.chat_id == chat_id:
                cfg = c
                break
        if cfg is None:
            cfg = settings.get_chats()[0]
        chat_wiki = _chat_wiki(cfg)

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify_query(question)
        rel = f"wiki/queries/{ts}-{slug}.md"
        author = (
            cb.from_user.full_name if cb.from_user else "anon"
        ) if cb.from_user else "anon"
        body = (
            f"---\n"
            f"type: query\n"
            f"created: {now.strftime('%Y-%m-%d')}\n"
            f"author: {author}\n"
            f"---\n\n"
            f"# {question}\n\n"
            f"{answer}\n"
        )
        try:
            chat_wiki.write_file(rel, body)
        except Exception as exc:  # noqa: BLE001
            log.error("qsave write failed: %s", exc)
            await cb.answer("❌ не удалось сохранить", show_alert=True)
            return
        # На всякий случай выкидываем эту же запись из fix-кэша,
        # чтобы потом не висели битые reply-промпты.
        _pending_fixes.pop(save_id, None)
        # Убираем кнопку, чтобы повторно не нажали.
        try:
            if cb.message:
                await cb.message.edit_text(f"💾 Сохранено: <code>{rel}</code>", parse_mode="HTML")
        except Exception as exc:  # noqa: BLE001
            log.warning("qsave edit_text failed: %s", exc)
        await cb.answer("Сохранено ✅")

    @router.callback_query(F.data.startswith("qfix:"))
    async def cb_fix_request(cb: CallbackQuery) -> None:
        """Пользователь нажал «✏️ Поправить» — просим написать правки реплаем."""
        save_id = (cb.data or "").split(":", 1)[1] if cb.data else ""
        if save_id not in _pending_fixes:
            await cb.answer("⌛ устарело, спроси заново", show_alert=False)
            return
        if not cb.message:
            await cb.answer("не могу ответить тут", show_alert=True)
            return
        try:
            prompt_msg = await cb.message.answer(
                "✏️ Окей, ответь <b>реплаем на это сообщение</b> — напиши, "
                "что в ответе не так и как правильно. Я перепишу с учётом твоих "
                "правок и сохраню в <code>wiki/queries/</code>.",
                parse_mode="HTML",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("qfix prompt send failed: %s", exc)
            await cb.answer("❌ не получилось", show_alert=True)
            return
        _fix_prompt_to_save_id[prompt_msg.message_id] = save_id
        # Убираем кнопки у исходного сообщения, чтобы не нажимали повторно.
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        await cb.answer("жду правки 👀")

    @router.message(F.reply_to_message)
    async def on_correction_reply(msg: Message) -> None:
        """Реплай на бот-сообщение «жду правки» — переписываем ответ и сохраняем."""
        if not msg.reply_to_message:
            return
        target_id = msg.reply_to_message.message_id
        save_id = _fix_prompt_to_save_id.pop(target_id, None)
        if not save_id:
            return  # это не наш промпт — отдаём дальше другим хендлерам
        entry = _pending_fixes.pop(save_id, None)
        if not entry:
            await msg.reply("⌛ устарело, спроси заново через /ask")
            return
        # Если этот же save_id ещё лежал в _pending_saves — тоже выкидываем,
        # черновой ответ больше неактуален.
        _pending_saves.pop(save_id, None)
        question, original_answer, chat_id = entry
        correction = (msg.text or msg.caption or "").strip()
        if not correction:
            await msg.reply("Пустые правки — нечего применять.")
            return

        await msg.chat.do("typing")
        try:
            cfg: ChatConfig | None = None
            for c in settings.get_chats():
                if c.chat_id == chat_id:
                    cfg = c
                    break
            revised = await orch.revise_query(
                question, original_answer, correction, chat_cfg=cfg
            )
        except Exception as exc:  # noqa: BLE001
            log.error("revise_query failed: %s", exc)
            await msg.reply("⚠️ Не получилось переписать ответ, попробуй ещё раз.")
            return

        # Отправим переписанный ответ пользователю.
        try:
            await _reply_html(msg, revised or "(пусто)")
        except Exception as exc:  # noqa: BLE001
            log.warning("revise: send failed: %s", exc)

        # И сразу сохраняем в wiki/queries как revised.
        cfg = cfg or settings.get_chats()[0]
        chat_wiki = _chat_wiki(cfg)
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify_query(question)
        rel = f"wiki/queries/{ts}-{slug}-revised.md"
        author = msg.from_user.full_name if msg.from_user else "anon"
        body = (
            f"---\n"
            f"type: query\n"
            f"revised: true\n"
            f"created: {now.strftime('%Y-%m-%d')}\n"
            f"author: {author}\n"
            f"original_chars: {len(original_answer)}\n"
            f"---\n\n"
            f"# {question}\n\n"
            f"## Ответ (с учётом правок пользователя)\n\n"
            f"{revised}\n\n"
            f"## Правки от пользователя\n\n"
            f"{correction}\n"
        )
        try:
            chat_wiki.write_file(rel, body)
        except Exception as exc:  # noqa: BLE001
            log.error("revise: write failed: %s", exc)
            await msg.reply("⚠️ Ответ переписан, но сохранить в wiki не удалось.")
            return
        await msg.reply(f"✅ Сохранено с правками: <code>{rel}</code>", parse_mode="HTML")

    return router
