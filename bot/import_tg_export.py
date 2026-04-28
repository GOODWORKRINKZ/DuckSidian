"""Импорт Telegram Desktop HTML-экспорта в vault DuckSidian.

Использование:
    python -m bot.import_tg_export <путь_к_ChatExport_*> [--chat-name main] [--dry-run]

Что делает:
- Парсит messages*.html, группирует по дате
- Для каждого дня создаёт raw/daily/YYYY-MM-DD.md (если уже есть — пропускает, если --force — перезаписывает)
- Копирует медиафайлы в vault/<chat>/raw/assets/<YYYY-MM-DD>/
- Идемпотентен: повторный запуск с новыми данными дописывает только новые даты
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from bs4 import BeautifulSoup  # type: ignore[import]
except ImportError:
    print("Требуется beautifulsoup4: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)


# ── Парсинг ───────────────────────────────────────────────────────────────────

def _parse_date(title_attr: str) -> datetime | None:
    """'28.04.2025 18:59:09 UTC+03:00' → aware datetime."""
    if not title_attr:
        return None
    # Telegram Desktop format: DD.MM.YYYY HH:MM:SS UTC±HH:MM
    try:
        # Убираем "UTC" и парсим offset вручную
        parts = title_attr.strip().split()
        if len(parts) < 3:
            return None
        date_str = parts[0]        # DD.MM.YYYY
        time_str = parts[1]        # HH:MM:SS
        offset_str = parts[2]      # UTC+03:00 или UTC-05:00
        dt_naive = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M:%S")
        # offset_str = "UTC+03:00" → "+03:00"
        tz_part = offset_str.replace("UTC", "")
        sign = 1 if "+" in tz_part else -1
        tz_part = tz_part.lstrip("+-")
        h, m = (int(x) for x in tz_part.split(":"))
        from datetime import timedelta, timezone as tz
        offset = timedelta(hours=sign * h, minutes=sign * m)
        return dt_naive.replace(tzinfo=tz(offset))
    except Exception:
        return None


def _get_media(body_div) -> tuple[str | None, str | None]:
    """Извлечь тип медиа и относительный путь из тега body сообщения.

    Возвращает (media_type, href) или (None, None).
    """
    wrap = body_div.find(class_="media_wrap")
    if not wrap:
        return None, None

    # Фото
    a = wrap.find("a", class_="photo_wrap")
    if a:
        return "photo", a.get("href")

    # Видео / GIF
    a = wrap.find("a", class_="video_file_wrap")
    if a:
        return "video", a.get("href")
    a = wrap.find("a", class_="animated_wrap")
    if a:
        return "gif", a.get("href")

    # Круглое видео
    a = wrap.find("a", class_="roundvideo_wrap")
    if a:
        return "video_note", a.get("href")

    # Документ / файл
    a = wrap.find("a", class_=lambda c: c and "media_file" in c)
    if a:
        return "document", a.get("href")

    # Голосовое
    a = wrap.find("a", class_=lambda c: c and "voice_message" in c)
    if a:
        return "voice", a.get("href")

    return None, None


def parse_html_file(html_path: Path) -> list[dict]:
    """Распарсить один messages*.html → список словарей-сообщений."""
    html = html_path.read_text("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    messages: list[dict] = []
    # from_name "прилипает" к следующим joined-сообщениям
    current_from = None

    for div in soup.find_all("div", class_="message"):
        classes = div.get("class", [])
        if "service" in classes:
            current_from = None
            continue

        msg_id_str = div.get("id", "")
        msg_id = int(msg_id_str.replace("message-", "").replace("-", "")) if msg_id_str else 0

        body = div.find(class_="body")
        if not body:
            continue

        # Дата
        date_tag = body.find(class_="date")
        dt = _parse_date(date_tag.get("title", "")) if date_tag else None

        # Автор
        from_tag = body.find(class_="from_name")
        if from_tag:
            current_from = from_tag.get_text(strip=True)

        # Текст
        text_tag = body.find(class_="text")
        text = text_tag.get_text(separator="\n", strip=True) if text_tag else ""

        # Подпись (caption) может быть внутри media_wrap
        caption_tag = body.find(class_="caption")
        caption = caption_tag.get_text(strip=True) if caption_tag else ""

        # Медиа
        media_type, media_href = _get_media(body)

        # Reply
        reply_tag = body.find(class_="reply_to")
        reply_to_id = None
        if reply_tag:
            a = reply_tag.find("a")
            if a and a.get("href", "").startswith("#go_to_message"):
                raw = a.get("href").replace("#go_to_message-", "").replace("-", "")
                try:
                    reply_to_id = int(raw)
                except ValueError:
                    pass

        messages.append({
            "id": msg_id,
            "dt": dt,
            "from": current_from or "Unknown",
            "text": text or caption,
            "media_type": media_type,
            "media_href": media_href,  # относительный путь от root экспорта
            "reply_to": reply_to_id,
        })

    return messages


# ── Запись в vault ─────────────────────────────────────────────────────────────

MEDIA_EMOJI = {
    "photo": "🖼",
    "video": "🎬",
    "gif": "🎞",
    "video_note": "⭕",
    "voice": "🎤",
    "document": "📎",
}

def _safe_name(fname: str) -> str:
    """Убрать небезопасные символы из имени файла."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in fname)


def _copy_media(src: Path, export_root: Path, assets_dir: Path) -> str | None:
    """Скопировать медиафайл в vault/assets. Возвращает имя файла или None."""
    orig = export_root / src
    if not orig.exists():
        return None
    dst_name = _safe_name(orig.name)
    dst = assets_dir / dst_name
    if not dst.exists():
        shutil.copy2(orig, dst)
    return dst_name


def write_daily_md(
    date_str: str,
    messages: list[dict],
    export_root: Path,
    chat_wiki_root: Path,
    dry_run: bool,
    force: bool,
) -> bool:
    """Записать один raw/daily/YYYY-MM-DD.md. Возвращает True если записан."""
    daily_dir = chat_wiki_root / "raw" / "daily"
    assets_dir = chat_wiki_root / "raw" / "assets" / date_str
    out_path = daily_dir / f"{date_str}.md"

    if out_path.exists() and not force:
        print(f"  SKIP {out_path.name} (уже есть, используй --force для перезаписи)")
        return False

    if not dry_run:
        daily_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# raw/{date_str} (импорт TG Desktop export)",
        f"> Источник: ChatExport  |  Сообщений: {len(messages)}",
        "",
    ]

    for msg in messages:
        dt = msg["dt"]
        time_str = dt.strftime("%H:%M") if dt else "??:??"
        from_name = msg["from"]
        msg_id = msg["id"]
        text = msg["text"]
        media_type = msg["media_type"]
        media_href = msg["media_href"]
        reply_to = msg["reply_to"]

        lines.append(f"---\n> **{from_name}** {time_str} `[id:{msg_id}]`")
        if reply_to:
            lines.append(f"> ↩ reply to `[id:{reply_to}]`")
        if text:
            for tline in text.split("\n"):
                lines.append(f"> {tline}")
        if media_type and media_href:
            emoji = MEDIA_EMOJI.get(media_type, "📎")
            if not dry_run:
                fname = _copy_media(Path(media_href), export_root, assets_dir)
            else:
                fname = Path(media_href).name
            if fname:
                rel = f"raw/assets/{date_str}/{fname}"
                lines.append(f"> {emoji} `[[{rel}]]`")
            else:
                lines.append(f"> {emoji} _{media_href} (файл не найден)_")
        lines.append("")

    content = "\n".join(lines)
    if dry_run:
        print(f"  DRY-RUN {out_path.name}: {len(messages)} сообщений")
    else:
        out_path.write_text(content, encoding="utf-8")
        print(f"  WROTE {out_path.name}: {len(messages)} сообщений, assets: {assets_dir}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Импорт TG Desktop HTML-экспорта в vault")
    ap.add_argument("export_dir", help="Путь к папке ChatExport_*")
    ap.add_argument("--chat-name", default="main", help="Имя чата в vault (default: main)")
    ap.add_argument("--vault", default=None, help="Путь к vault (default: из .env/VAULT_PATH)")
    ap.add_argument("--dry-run", action="store_true", help="Не записывать файлы, только показать")
    ap.add_argument("--force", action="store_true", help="Перезаписать существующие daily файлы")
    args = ap.parse_args()

    export_root = Path(args.export_dir).expanduser().resolve()
    if not export_root.is_dir():
        print(f"Ошибка: {export_root} не является директорией", file=sys.stderr)
        sys.exit(1)

    # Определить vault_path
    if args.vault:
        vault_path = Path(args.vault).expanduser().resolve()
    else:
        try:
            from .config import settings
            vault_path = settings.vault_path
        except ImportError:
            vault_path = Path(__file__).parent.parent / "vault"

    chat_wiki_root = vault_path / args.chat_name
    print(f"Export: {export_root}")
    print(f"Vault chat root: {chat_wiki_root}")
    print(f"Dry-run: {args.dry_run}  Force: {args.force}")
    print()

    # Парсим все html файлы
    all_messages: list[dict] = []
    for html_path in sorted(export_root.glob("messages*.html")):
        msgs = parse_html_file(html_path)
        print(f"Parsed {html_path.name}: {len(msgs)} сообщений")
        all_messages.extend(msgs)

    # Группируем по дате
    by_date: dict[str, list[dict]] = defaultdict(list)
    no_date = 0
    for msg in all_messages:
        if msg["dt"] is None:
            no_date += 1
            continue
        date_str = msg["dt"].strftime("%Y-%m-%d")
        by_date[date_str].append(msg)

    # Сортируем сообщения внутри каждого дня
    for date_str in by_date:
        by_date[date_str].sort(key=lambda m: m["dt"] or datetime.min.replace(tzinfo=timezone.utc))

    print(f"\nВсего: {len(all_messages)} сообщений, {len(by_date)} дней, без даты: {no_date}")
    print(f"Период: {min(by_date)} → {max(by_date)}")
    print()

    written = 0
    skipped = 0
    for date_str in sorted(by_date):
        ok = write_daily_md(
            date_str,
            by_date[date_str],
            export_root,
            chat_wiki_root,
            dry_run=args.dry_run,
            force=args.force,
        )
        if ok:
            written += 1
        else:
            skipped += 1

    print(f"\nГотово: записано {written}, пропущено {skipped}")
    if not args.dry_run and written > 0:
        print("\nСледующий шаг: запусти ingest чтобы агент обработал импортированные данные:")
        print("  /ingest  — в боте")


if __name__ == "__main__":
    main()
