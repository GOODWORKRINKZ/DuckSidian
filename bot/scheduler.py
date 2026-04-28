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
        log.info("scheduled ingest for %s", date_iso)
        try:
            out = await orch.ingest_for_date(date_iso)
            log.info("ingest done: %s", out[:200])
        except Exception:  # noqa: BLE001
            log.exception("scheduled ingest failed")

    parts = settings.ingest_cron.split()
    if len(parts) != 5:
        raise ValueError(f"bad INGEST_CRON: {settings.ingest_cron!r}")
    minute, hour, day, month, dow = parts
    trigger = CronTrigger(
        minute=minute, hour=hour, day=day, month=month, day_of_week=dow,
        timezone=settings.tz,
    )
    scheduler.add_job(job, trigger, id="daily_ingest", replace_existing=True)
    return scheduler
