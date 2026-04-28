"""APScheduler: запускает ingest по cron из настроек."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import settings
from .db import DB
from .orchestrator import Orchestrator

log = logging.getLogger(__name__)


def build_scheduler(db: DB, orch: Orchestrator) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.tz)

    async def job() -> None:
        if (await db.get_state("paused", "0")) == "1":
            log.info("ingest paused, skipping")
            return
        date_iso = datetime.now(timezone.utc).date().isoformat()
        for chat_cfg in settings.get_chats():
            log.info("scheduled ingest for %s / %s", chat_cfg.name, date_iso)
            try:
                out = await orch.ingest_for_date(date_iso, chat_cfg)
                log.info("ingest done [%s]: %s", chat_cfg.name, out[:200])
            except Exception:  # noqa: BLE001
                log.exception("scheduled ingest failed for %s", chat_cfg.name)

    parts = settings.ingest_cron.split()
    if len(parts) != 5:
        raise ValueError(f"bad INGEST_CRON: {settings.ingest_cron!r}")
    minute, hour, day, month, dow = parts
    trigger = CronTrigger(
        minute=minute, hour=hour, day=day, month=month, day_of_week=dow,
        timezone=settings.tz,
    )
    scheduler.add_job(job, trigger, id="daily_ingest", replace_existing=True)

    _auto_ingest_running = False

    async def auto_ingest_pending() -> None:
        """Каждые 20 минут: если есть необработанные raw/daily/*.md — инжестируем."""
        nonlocal _auto_ingest_running
        if _auto_ingest_running:
            log.info("auto_ingest_pending: already running, skip")
            return
        if (await db.get_state("paused", "0")) == "1":
            return
        from pathlib import Path
        for chat_cfg in settings.get_chats():
            raw_dir = settings.vault_path / chat_cfg.name / "raw" / "daily"
            wiki_dir = settings.vault_path / chat_cfg.name / "wiki" / "daily"
            if not raw_dir.exists():
                continue
            pending = [
                p.stem for p in sorted(raw_dir.glob("*.md"))
                if not (wiki_dir / p.name).exists()
            ]
            if not pending:
                continue
            log.info("auto_ingest_pending: %d дней в очереди для %s, старт",
                     len(pending), chat_cfg.name)
            _auto_ingest_running = True
            try:
                out = await orch.ingest_raw_pending(chat_cfg, limit=3)
                log.info("auto_ingest_pending done [%s]: %s", chat_cfg.name, out[:200])
            except Exception:  # noqa: BLE001
                log.exception("auto_ingest_pending failed for %s", chat_cfg.name)
            finally:
                _auto_ingest_running = False

    scheduler.add_job(
        auto_ingest_pending,
        "interval",
        minutes=20,
        id="auto_ingest_pending",
        replace_existing=True,
    )

    return scheduler
