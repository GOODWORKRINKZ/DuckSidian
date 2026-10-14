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
    media_type TEXT,
    media_path TEXT,
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

CREATE TABLE IF NOT EXISTS disputes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    context TEXT,
    options TEXT,
    answer TEXT,
    answered_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    chat_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (chat_id, topic_id)
);

CREATE TABLE IF NOT EXISTS workspaces (
    chat_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    project_slug TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (chat_id, topic_id)
);

CREATE TABLE IF NOT EXISTS shares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    src_path TEXT NOT NULL,
    target_chat_id INTEGER NOT NULL,
    dst_path TEXT NOT NULL,
    posted_message_id INTEGER,
    created_at TEXT NOT NULL
);
"""


class DB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA)
            # Migration: добавить медиа-колонки в существующие БД
            for col in ("media_type", "media_path"):
                try:
                    await db.execute(
                        f"ALTER TABLE messages ADD COLUMN {col} TEXT"
                    )
                    await db.commit()
                except Exception:  # noqa: BLE001
                    pass  # колонка уже существует
            # Migration: создать таблицу disputes если она не создана через SCHEMA
            # (для уже существующих БД без неё)
            await db.execute(
                """CREATE TABLE IF NOT EXISTS disputes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    context TEXT,
                    options TEXT,
                    answer TEXT,
                    answered_at TEXT,
                    created_at TEXT NOT NULL
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS topics (
                    chat_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (chat_id, topic_id)
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS workspaces (
                    chat_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    project_slug TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (chat_id, topic_id)
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS shares (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    src_path TEXT NOT NULL,
                    target_chat_id INTEGER NOT NULL,
                    dst_path TEXT NOT NULL,
                    posted_message_id INTEGER,
                    created_at TEXT NOT NULL
                )"""
            )
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
        media_type: str | None = None,
        media_path: str | None = None,
        reply_to_message_id: int | None,
        ts: datetime,
    ) -> None:
        async with self.conn() as db:
            await db.execute(
                """INSERT OR IGNORE INTO messages
                (chat_id, message_id, topic_id, user_id, username, full_name,
                 text, media_type, media_path, reply_to_message_id, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chat_id,
                    message_id,
                    topic_id,
                    user_id,
                    username,
                    full_name,
                    text,
                    media_type,
                    media_path,
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

    # --- topics ---

    async def save_topic(self, chat_id: int, topic_id: int, name: str) -> None:
        async with self.conn() as db:
            await db.execute(
                """INSERT INTO topics (chat_id, topic_id, name, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(chat_id, topic_id) DO UPDATE
                   SET name=excluded.name, updated_at=excluded.updated_at""",
                (chat_id, topic_id, name, datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()

    async def get_topics(self, chat_id: int) -> dict[int, str]:
        """Вернуть {topic_id: name} для чата."""
        async with self.conn() as db:
            async with db.execute(
                "SELECT topic_id, name FROM topics WHERE chat_id=?", (chat_id,)
            ) as cur:
                rows = await cur.fetchall()
        return {int(r["topic_id"]): r["name"] for r in rows}

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

    # --- disputes ---

    async def add_dispute(
        self, question: str, context: str | None, options: list[str] | None
    ) -> int:
        import json
        async with self.conn() as db:
            cur = await db.execute(
                """INSERT INTO disputes (question, context, options, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    question,
                    context,
                    json.dumps(options, ensure_ascii=False) if options else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
            return cur.lastrowid or 0

    async def get_oldest_dispute(self) -> aiosqlite.Row | None:
        """Вернуть самый старый неотвеченный спорный вопрос."""
        async with self.conn() as db:
            async with db.execute(
                """SELECT * FROM disputes WHERE answer IS NULL
                   ORDER BY created_at ASC LIMIT 1"""
            ) as cur:
                return await cur.fetchone()

    async def count_pending_disputes(self) -> int:
        async with self.conn() as db:
            async with db.execute(
                "SELECT COUNT(*) FROM disputes WHERE answer IS NULL"
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

    async def answer_dispute(self, dispute_id: int, answer: str) -> None:
        async with self.conn() as db:
            await db.execute(
                """UPDATE disputes SET answer=?, answered_at=? WHERE id=?""",
                (answer, datetime.now(timezone.utc).isoformat(), dispute_id),
            )
            await db.commit()

    # --- private mode: allowlist (хранится в state как JSON-список) ---

    _PRIVATE_USERS_KEY = "private_users"

    async def get_private_users(self) -> list[int]:
        import json
        raw = await self.get_state(self._PRIVATE_USERS_KEY, "[]")
        try:
            return [int(x) for x in json.loads(raw)]
        except Exception:  # noqa: BLE001
            return []

    async def add_private_user(self, user_id: int) -> bool:
        import json
        users = await self.get_private_users()
        if user_id in users:
            return False
        users.append(int(user_id))
        await self.set_state(self._PRIVATE_USERS_KEY, json.dumps(users))
        return True

    async def remove_private_user(self, user_id: int) -> bool:
        import json
        users = await self.get_private_users()
        if user_id not in users:
            return False
        users.remove(int(user_id))
        await self.set_state(self._PRIVATE_USERS_KEY, json.dumps(users))
        return True

    # --- shares ---

    async def add_share(
        self, user_id: int, src_path: str, target_chat_id: int,
        dst_path: str, posted_message_id: int | None = None,
    ) -> int:
        async with self.conn() as db:
            cur = await db.execute(
                """INSERT INTO shares
                   (user_id, src_path, target_chat_id, dst_path, posted_message_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, src_path, target_chat_id, dst_path, posted_message_id,
                 datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()
            return cur.lastrowid or 0

    async def fetch_shares(self, user_id: int, limit: int = 20) -> list[aiosqlite.Row]:
        async with self.conn() as db:
            async with db.execute(
                """SELECT * FROM shares WHERE user_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            ) as cur:
                return list(await cur.fetchall())

    # --- workspaces: topic в личке с ботом = воркспейс/проект ---

    async def upsert_workspace(
        self, chat_id: int, topic_id: int, name: str, project_slug: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self.conn() as db:
            await db.execute(
                """INSERT INTO workspaces (chat_id, topic_id, name, project_slug, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chat_id, topic_id) DO UPDATE SET
                     name=excluded.name,
                     project_slug=COALESCE(excluded.project_slug, workspaces.project_slug),
                     updated_at=excluded.updated_at""",
                (chat_id, topic_id, name, project_slug, now, now),
            )
            await db.commit()

    async def get_workspace(self, chat_id: int, topic_id: int) -> aiosqlite.Row | None:
        async with self.conn() as db:
            async with db.execute(
                "SELECT * FROM workspaces WHERE chat_id=? AND topic_id=?", (chat_id, topic_id)
            ) as cur:
                return await cur.fetchone()

    async def list_workspaces(self, chat_id: int) -> list[aiosqlite.Row]:
        async with self.conn() as db:
            async with db.execute(
                "SELECT * FROM workspaces WHERE chat_id=? ORDER BY updated_at DESC", (chat_id,)
            ) as cur:
                return list(await cur.fetchall())

    async def delete_workspace(self, chat_id: int, topic_id: int) -> None:
        async with self.conn() as db:
            await db.execute(
                "DELETE FROM workspaces WHERE chat_id=? AND topic_id=?", (chat_id, topic_id)
            )
            await db.commit()

    async def fetch_last_messages_in_topic(
        self, chat_id: int, topic_id: int | None, n: int
    ) -> list[aiosqlite.Row]:
        """Последние n сообщений чата в конкретном топике (None = вне топиков)."""
        cond = "topic_id IS NULL" if topic_id is None else "topic_id=?"
        params: tuple = (chat_id, n) if topic_id is None else (chat_id, topic_id, n)
        async with self.conn() as db:
            async with db.execute(
                f"""SELECT * FROM (
                       SELECT * FROM messages WHERE chat_id=? AND {cond}
                       ORDER BY ts DESC LIMIT ?
                   ) ORDER BY ts ASC""",
                params,
            ) as cur:
                return list(await cur.fetchall())


    async def find_user_id_by_username(self, username: str) -> int | None:
        """user_id по @username среди всех, кто писал боту (в т.ч. в группе)."""
        u = username.lstrip("@").strip().lower()
        if not u:
            return None
        async with self.conn() as db:
            async with db.execute(
                """SELECT user_id FROM messages WHERE lower(username)=? AND user_id IS NOT NULL
                   ORDER BY ts DESC LIMIT 1""",
                (u,),
            ) as cur:
                row = await cur.fetchone()
        return int(row["user_id"]) if row else None


    async def find_workspaces_by_slug(self, project_slug: str) -> list[aiosqlite.Row]:
        """Все темы (всех пользователей), привязанные к project_slug (для shared/<slug>)."""
        async with self.conn() as db:
            async with db.execute(
                "SELECT * FROM workspaces WHERE project_slug=? ORDER BY updated_at", (project_slug,)
            ) as cur:
                return list(await cur.fetchall())
