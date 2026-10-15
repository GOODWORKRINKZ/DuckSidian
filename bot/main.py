"""Entrypoint: aiogram + scheduler + healthcheck."""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    Update,
)
from aiohttp import web

from .config import settings
from .db import DB
from .git_sync import ensure_repo
from .handlers import commands as cmd_router_mod
from .handlers import listener as listener_mod
from .handlers import private as private_mod
from .orchestrator import Orchestrator
from .scheduler import build_scheduler
from .topic_manager import ensure_bot_topic
from .wiki import Wiki

log = logging.getLogger(__name__)


def _ensure_vault_initialised(vault: Path) -> None:
    """Если vault пуст — копируем встроенный template (для случая, когда
    bootstrap.sh не отработал и пользователь монтирует пустую папку)."""
    if (vault / "AGENTS.md").exists():
        return
    template = Path(__file__).parent.parent / "vault-template"
    if not template.exists():
        log.warning("vault template missing at %s", template)
        return
    log.info("seeding vault from %s", template)
    for src in template.rglob("*"):
        rel = src.relative_to(template)
        dst = vault / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


async def _run_health_server() -> None:
    async def healthz(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def busy(_: web.Request) -> web.Response:
        """Сколько ходов агента идёт/ждёт в очереди — для перезапуска без обрыва
        (scripts/restart_when_idle.sh ждёт здесь 0)."""
        from .handlers.private import busy_count  # noqa: PLC0415
        return web.Response(text=str(busy_count()))

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/busy", busy)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    log.info("healthz listening on :8080")


PRIVATE_COMMANDS = [
    ("help", "что я умею в личке"),
    ("ws", "воркспейс (тема): /ws, /ws new <имя>, /ws list, /ws share <чат>"),
    ("ask", "вопрос к моему личному vault"),
    ("note", "быстрая заметка без агента"),
    ("list", "мои артефакты"),
    ("show", "показать артефакт (№ / имя / путь)"),
    ("share", "поделиться артефактом с чатом"),
    ("projects", "мои проекты"),
    ("search", "поиск по личному vault"),
    ("say", "озвучить текст голосом"),
    ("voice", "голосовые ответы: on / off / auto"),
    ("imagine", "сгенерировать картинку"),
    ("space", "состояние пространства и медиа-ИИ"),
    ("del", "удалить артефакт"),
    ("model", "LLM-модель (admin)"),
]
GROUP_COMMANDS = [
    ("ask", "спросить wiki команды"),
    ("triz", "ТРИЗ-разбор проблемы"),
    ("projects", "проекты в вики"),
    ("search", "поиск по vault"),
    ("summary", "сводка day|week"),
    ("note", "заметка в raw/notes"),
    ("help", "все команды"),
    ("ingest", "внеплановый ingest (admin)"),
    ("model", "LLM-модель (admin)"),
    ("allow", "доступ к личному режиму (admin)"),
]


async def _setup_bot_menu(bot: Bot) -> None:
    """Меню команд по скоупам (личка / группы) + проверка режима тем в личке."""
    try:
        await bot.set_my_commands(
            [BotCommand(command=c, description=d[:256]) for c, d in PRIVATE_COMMANDS],
            scope=BotCommandScopeAllPrivateChats(),
        )
        await bot.set_my_commands(
            [BotCommand(command=c, description=d[:256]) for c, d in GROUP_COMMANDS],
            scope=BotCommandScopeAllGroupChats(),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("set_my_commands failed: %s", exc)
    try:
        me = await bot.get_me()
        topics = getattr(me, "has_topics_enabled", None)
        log.info("bot @%s: topics in private chats = %s", me.username, topics)
        if topics is False:
            log.warning(
                "Темы в личке выключены: включи в @BotFather → Bot Settings → "
                "Threaded mode (Topics), иначе воркспейсы (/ws new) не заработают."
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("get_me failed: %s", exc)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Отдельный файл для токенов/расходов — data/tokens.log
    _tokens_log_path = settings.data_path / "tokens.log"
    _tokens_log_path.parent.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_tokens_log_path, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _fh.setLevel(logging.DEBUG)
    logging.getLogger("bot.tokens").addHandler(_fh)
    logging.getLogger("bot.tokens").setLevel(logging.DEBUG)
    logging.getLogger("bot.tokens").propagate = False  # не дублировать в основной лог

    _ensure_vault_initialised(settings.vault_path)
    # Сеять поддиректории для каждого зарегистрированного чата
    for chat_cfg in settings.get_chats():
        from .orchestrator import _chat_wiki  # noqa: PLC0415
        _chat_wiki(chat_cfg)  # вызов создаёт директории
    ensure_repo(settings.vault_path)

    db = DB(settings.data_path / "bot.sqlite3")
    await db.init()

    wiki = Wiki(settings.vault_path)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    orch = Orchestrator(bot, db, wiki)
    await orch.restore_model()
    await _setup_bot_menu(bot)

    # Инициализировать форум-топик бота для каждого чата
    for chat_cfg in settings.get_chats():
        await ensure_bot_topic(bot, db, chat_cfg.chat_id)

    dp = Dispatcher()

    @dp.update.outer_middleware()
    async def _log_all_updates(handler, event: Update, data: dict):
        log.info(
            "RAW_UPDATE id=%s type=%s",
            event.update_id,
            event.event_type,
        )
        return await handler(event, data)

    # Личный режим (DM) — первым: его фильтр F.chat.type == "private" отсекает
    # приватные сообщения до chat_guard группового роутера.
    dp.include_router(private_mod.setup(db, orch, bot))
    dp.include_router(cmd_router_mod.setup(db, wiki, orch))
    dp.include_router(listener_mod.setup(db, orch, bot))

    scheduler = build_scheduler(db, orch)
    scheduler.start()

    await _run_health_server()

    log.info("starting polling for chat_id=%s", settings.telegram_chat_id)
    log.info("LLM: %s", orch.model_info())
    try:
        # Retry-обёртка: при потере DNS / сети aiogram может вылететь с
        # TelegramNetworkError ещё до старта polling-цикла. Не падаем — ждём.
        backoff = 5
        while True:
            try:
                await dp.start_polling(bot)
                break  # штатное завершение
            except Exception as exc:  # noqa: BLE001
                log.error("polling crashed: %s — retry in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
    finally:
        scheduler.shutdown(wait=False)
        await orch.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
