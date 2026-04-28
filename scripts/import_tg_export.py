#!/usr/bin/env python3
"""Импорт истории чата из Telegram Desktop JSON-экспорта в bot.sqlite3.

Использование:
    python scripts/import_tg_export.py <path/to/result.json> [--chat-id CHAT_ID] [--dry-run]

Telegram Desktop: Настройки группы → Экспортировать историю чата → JSON.
Файл обычно называется result.json внутри папки экспорта.

Опции:
    --chat-id   Принудительно задать chat_id (по умолчанию берётся из JSON или из .env)
    --dry-run   Только показать сколько сообщений будет импортировано, без записи в БД
    --db        Путь к bot.sqlite3 (по умолчанию bot/data/bot.sqlite3)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Добавляем корень проекта в путь чтобы можно было импортировать bot.*
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_ts(date_str: str) -> str:
    """Конвертировать дату из формата Telegram в ISO 8601 UTC."""
    # Telegram экспортирует в формате "2024-01-15T14:30:45" (без timezone, local time)
    # или "2024-01-15 14:30:45"
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt)
            # Предполагаем что экспорт в UTC (Telegram хранит в UTC)
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    raise ValueError(f"Не удалось распарсить дату: {date_str!r}")


def extract_text(text_field) -> str | None:
    """Telegram хранит text как строку или как список сегментов."""
    if isinstance(text_field, str):
        return text_field or None
    if isinstance(text_field, list):
        parts = []
        for seg in text_field:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict):
                parts.append(seg.get("text", ""))
        result = "".join(parts)
        return result or None
    return None


def detect_media_type(msg: dict) -> str | None:
    """Определить тип медиа из структуры сообщения."""
    mt = msg.get("media_type") or msg.get("type")
    if mt == "sticker":
        return "sticker"
    if msg.get("photo"):
        return "photo"
    if msg.get("file"):
        file_path = str(msg.get("file", ""))
        mime = msg.get("mime_type", "")
        if mime.startswith("video") or file_path.endswith((".mp4", ".mov", ".avi")):
            return "video"
        if mime.startswith("audio") or file_path.endswith((".mp3", ".ogg", ".m4a")):
            return "audio"
        return "document"
    if msg.get("voice"):
        return "voice"
    if msg.get("video_file"):
        return "video"
    return None


async def import_messages(
    json_path: Path,
    db_path: Path,
    override_chat_id: int | None,
    dry_run: bool,
) -> None:
    import aiosqlite

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Определить chat_id
    if override_chat_id:
        chat_id = override_chat_id
    else:
        # Telegram Desktop хранит id как положительное число, но в Bot API группы negative
        raw_id = data.get("id", 0)
        if raw_id > 0:
            # Супергруппа/канал — добавляем -100 префикс
            chat_id = int(f"-100{raw_id}")
        else:
            chat_id = raw_id

    chat_name = data.get("name", "unknown")
    messages = data.get("messages", [])

    # Фильтруем только обычные сообщения (не service)
    text_messages = [m for m in messages if m.get("type") == "message"]

    print(f"Чат: {chat_name!r}  (id={chat_id})")
    print(f"Всего сообщений в экспорте: {len(messages)}")
    print(f"Обычных message: {len(text_messages)}")

    if dry_run:
        # Показать первые 5
        for m in text_messages[:5]:
            print(f"  [{m.get('date')}] {m.get('from', '?')}: {str(extract_text(m.get('text', '')))[:80]}")
        print("--dry-run: запись в БД не производилась")
        return

    # Импорт в БД
    async with aiosqlite.connect(db_path) as db:
        inserted = 0
        skipped = 0
        for msg in text_messages:
            date_str = msg.get("date", "")
            if not date_str:
                continue
            try:
                ts = parse_ts(date_str)
            except ValueError as e:
                print(f"  WARN: {e}")
                continue

            message_id = int(msg.get("id", 0))
            user_id = msg.get("from_id")
            # from_id может быть "user123456789" или число
            if isinstance(user_id, str) and user_id.startswith("user"):
                try:
                    user_id = int(user_id[4:])
                except ValueError:
                    user_id = None
            elif isinstance(user_id, str) and user_id.startswith("channel"):
                user_id = None

            username = None  # В JSON-экспорте username не хранится
            full_name = msg.get("from") or msg.get("actor")
            text = extract_text(msg.get("text", ""))
            media_type = detect_media_type(msg)
            media_path = msg.get("photo") or msg.get("file") or msg.get("video_file")
            reply_to = msg.get("reply_to_message_id")

            try:
                await db.execute(
                    """INSERT OR IGNORE INTO messages
                    (chat_id, message_id, topic_id, user_id, username, full_name,
                     text, media_type, media_path, reply_to_message_id, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chat_id, message_id, None, user_id, username, full_name,
                        text, media_type, media_path, reply_to, ts,
                    ),
                )
                inserted += 1
            except Exception as e:
                print(f"  ERROR msg {message_id}: {e}")
                skipped += 1

        await db.commit()
        print(f"\nГотово: вставлено {inserted}, пропущено (дубли/ошибки) {skipped}")
        print(f"DB: {db_path}")

        # Итоговая статистика
        async with db.execute(
            "SELECT count(*), min(ts), max(ts) FROM messages WHERE chat_id=?",
            (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            print(f"Итого в БД для chat_id={chat_id}: {row[0]} сообщений ({row[1]} … {row[2]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Импорт Telegram JSON-экспорта в DuckSidian DB")
    parser.add_argument("json_file", help="Путь к result.json из Telegram Desktop экспорта")
    parser.add_argument("--chat-id", type=int, default=None, help="Принудительный chat_id")
    parser.add_argument("--db", default=str(ROOT / "bot/data/bot.sqlite3"), help="Путь к SQLite БД")
    parser.add_argument("--dry-run", action="store_true", help="Не писать в БД, только показать")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Файл не найден: {json_path}")
        sys.exit(1)

    db_path = Path(args.db)

    asyncio.run(import_messages(json_path, db_path, args.chat_id, args.dry_run))


if __name__ == "__main__":
    main()
