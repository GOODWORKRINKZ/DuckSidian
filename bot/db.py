"""SQLite-хранилище входящих сообщений и состояния бота."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable

import aiosqlite


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    topic_id INTEGER,
    user_id INTEGER,
    username TEXT,
    full_name TEXT,
    text TEXT,
    reply_to_message_id INTEGER,
    ts TEXT NOT NULL,
    processed_batch_id INTEGER,
    UNIQUE(chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_unprocessed
    ON messages(chat_id, processed_batch_id, ts);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'started',
    msg_count INTEGER DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS pending_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER,
    question TEXT NOT NULL,
    options TEXT,
    posted_message_id INTEGER,
    answer TEXT,
    answered_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class DB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def conn(self) -> AsyncIterator[aiosqlite.Connection]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    # --- messages ---

    async def save_message(
        self,
        *,
        chat_id: int,
        message_id: int,
        topic_id: int | None,
        user_id: int | None,
        username: str | None,
        full_name: str | None,
        text: str | None,
        reply_to_message_id: int | None,
        ts: datetime,
    ) -> None:
        async with self.conn() as db:
            await db.execute(
                """INSERT OR IGNORE INTO messages
                (chat_id, message_id, topic_id, user_id, username, full_name,
                 text, reply_to_message_id, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chat_id,
                    message_id,
                    topic_id,
                    user_id,
                    username,
                    full_name,
                    text,
                    reply_to_message_id,
                    ts.astimezone(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

    async def fetch_messages_for_date(
        self, chat_id: int, date_iso: str
    ) -> list[aiosqlite.Row]:
        """Все сообщения целевого чата за сутки (UTC) date_iso = YYYY-MM-DD."""
        start = f"{date_iso}T00:00:00+00:00"
        end = f"{date_iso}T23:59:59.999999+00:00"
        async with self.conn() as db:
            async with db.execute(
                """SELECT * FROM messages
                   WHERE chat_id=? AND ts>=? AND ts<=?
                   ORDER BY ts ASC""",
                (chat_id, start, end),
            ) as cur:
                return list(await cur.fetchall())

    async def fetch_last_messages(
        self, chat_id: int, n: int
    ) -> list[aiosqlite.Row]:
        async with self.conn() as db:
            async with db.execute(
                """SELECT * FROM (
                       SELECT * FROM messages WHERE chat_id=?
                       ORDER BY ts DESC LIMIT ?
                   ) ORDER BY ts ASC""",
                (chat_id, n),
            ) as cur:
                return list(await cur.fetchall())

    async def mark_processed(
        self, message_ids: Iterable[int], batch_id: int
    ) -> None:
        ids = list(message_ids)
        if not ids:
            return
        async with self.conn() as db:
            await db.executemany(
                "UPDATE messages SET processed_batch_id=? WHERE id=?",
                [(batch_id, i) for i in ids],
            )
            await db.commit()

    # --- batches ---

    async def create_batch(self, date_iso: str) -> int:
        async with self.conn() as db:
            cur = await db.execute(
                """INSERT INTO batches (date, started_at, status)
                   VALUES (?, ?, 'started')""",
                (date_iso, datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()
            return cur.lastrowid or 0

    async def finish_batch(
        self, batch_id: int, status: str, msg_count: int, notes: str = ""
    ) -> None:
        async with self.conn() as db:
            await db.execute(
                """UPDATE batches SET finished_at=?, status=?,
                   msg_count=?, notes=? WHERE id=?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    msg_count,
                    notes,
                    batch_id,
                ),
            )
            await db.commit()

    # --- state ---

    async def get_state(self, key: str, default: str = "") -> str:
        async with self.conn() as db:
            async with db.execute(
                "SELECT value FROM state WHERE key=?", (key,)
            ) as cur:
                row = await cur.fetchone()
                return row["value"] if row else default

    async def set_state(self, key: str, value: str) -> None:
        async with self.conn() as db:
            await db.execute(
                """INSERT INTO state(key, value) VALUES(?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (key, value),
            )
            await db.commit()

    # --- pending questions ---

    async def add_question(
        self, batch_id: int | None, question: str, options: str | None
    ) -> int:
        async with self.conn() as db:
            cur = await db.execute(
                """INSERT INTO pending_questions
                   (batch_id, question, options, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    batch_id,
                    question,
                    options,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
            return cur.lastrowid or 0

    async def attach_question_post(
        self, question_id: int, posted_message_id: int
    ) -> None:
        async with self.conn() as db:
            await db.execute(
                "UPDATE pending_questions SET posted_message_id=? WHERE id=?",
                (posted_message_id, question_id),
            )
            await db.commit()

    async def answer_question(self, question_id: int, answer: str) -> None:
        async with self.conn() as db:
            await db.execute(
                """UPDATE pending_questions SET answer=?, answered_at=?
                   WHERE id=?""",
                (
                    answer,
                    datetime.now(timezone.utc).isoformat(),
                    question_id,
                ),
            )
            await db.commit()

    async def get_question(self, question_id: int) -> aiosqlite.Row | None:
        async with self.conn() as db:
            async with db.execute(
                "SELECT * FROM pending_questions WHERE id=?", (question_id,)
            ) as cur:
                return await cur.fetchone()
