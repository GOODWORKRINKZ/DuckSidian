"""Entrypoint: aiogram + scheduler + healthcheck."""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiohttp import web

from .config import settings
from .db import DB
from .git_sync import ensure_repo
from .handlers import commands as cmd_router_mod
from .handlers import listener as listener_mod
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

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    log.info("healthz listening on :8080")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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

    # Инициализировать форум-топик бота для каждого чата
    for chat_cfg in settings.get_chats():
        await ensure_bot_topic(bot, db, chat_cfg.chat_id)

    dp = Dispatcher()
    dp.include_router(cmd_router_mod.setup(db, wiki, orch))
    dp.include_router(listener_mod.setup(db, orch, bot))

    scheduler = build_scheduler(db, orch)
    scheduler.start()

    await _run_health_server()

    log.info("starting polling for chat_id=%s", settings.telegram_chat_id)
    try:
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown(wait=False)
        await orch.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
