"""Импорт архивов (zip), присланных боту: экспорт чатов WhatsApp / Telegram → raw/daily,
прочее → распаковка текстов в raw/imports/ для чтения агентом.

Безопасность: защита от zip-slip, лимиты на число файлов и общий размер,
распаковываем только в raw/imports/<stem>/ внутри vault.
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .media_parser import TEXT_EXTENSIONS
from .wiki import Wiki

log = logging.getLogger(__name__)

MAX_FILES = 2_000
MAX_TOTAL_BYTES = 300 * 1024 * 1024
ARCHIVE_EXTENSIONS = {".zip"}


@dataclass
class ArchiveInfo:
    kind: str                       # whatsapp | telegram_html | telegram_json | generic
    root: Path                      # куда распаковано (абсолютный путь)
    rel_root: str                   # rel путь внутри vault
    files: list[Path] = field(default_factory=list)
    text_files: list[str] = field(default_factory=list)   # rel пути текстовых файлов
    dates: list[str] = field(default_factory=list)        # записанные raw/daily/<date>.md
    messages: int = 0
    notes: list[str] = field(default_factory=list)

    def summary_md(self) -> str:
        kind_h = {"whatsapp": "экспорт чата WhatsApp", "telegram_html": "экспорт чата Telegram (HTML)",
                  "telegram_json": "экспорт чата Telegram (JSON)", "generic": "архив"}[self.kind]
        lines = [f"📦 Распаковал {kind_h}: {len(self.files)} файл(ов) → `{self.rel_root}/`"]
        if self.dates:
            lines.append(f"Сообщений: {self.messages}, дней: {len(self.dates)} "
                         f"({self.dates[0]} … {self.dates[-1]}) → `raw/daily/`.")
        elif self.text_files:
            lines.append("Текстовые файлы:")
            lines += [f"- `{p}`" for p in self.text_files[:15]]
            if len(self.text_files) > 15:
                lines.append(f"- … ещё {len(self.text_files) - 15}")
        lines += [f"_{n}_" for n in self.notes]
        return "\n".join(lines)


def _safe_member_path(name: str) -> str | None:
    name = name.replace("\\", "/")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or name.startswith("/"):
        return None
    return "/".join(parts)


def safe_extract(zip_path: Path, dst_dir: Path) -> list[Path]:
    """Распаковать zip в dst_dir с защитой от zip-slip и лимитами. Возвращает файлы."""
    out: list[Path] = []
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if len(members) > MAX_FILES:
            raise ValueError(f"слишком много файлов в архиве ({len(members)} > {MAX_FILES})")
        for info in members:
            if info.is_dir():
                continue
            rel = _safe_member_path(info.filename)
            if rel is None:
                log.warning("archive: skip unsafe member %r", info.filename)
                continue
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("архив слишком большой после распаковки (>300MB)")
            target = dst_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            out.append(target)
    return out


# ── WhatsApp ──────────────────────────────────────────────────────────────────

# [10/9/25, 10:22:09 AM] Name: text   |  [09.10.2025, 10:22:09] Name: text  |  10/9/25, 10:22 - Name: text
_WA_LINE = re.compile(
    r"^‎?\[?(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)\]?\s*(?:-\s*)?"
    r"(?P<name>[^:]{1,80}?):\s(?P<text>.*)$"
)
_WA_SYSTEM_HINTS = ("end-to-end encrypted", "сквозным шифрованием", "Messages and calls are")


def _parse_wa_date(d: str, t: str) -> datetime | None:
    d = d.strip(); t = t.strip().replace(" ", " ").replace(".", "")
    fmts_d = ["%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y", "%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"]
    fmts_t = ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]
    for fd in fmts_d:
        for ft in fmts_t:
            try:
                return datetime.strptime(f"{d} {t}", f"{fd} {ft}").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def looks_like_whatsapp(text: str) -> bool:
    head = text[:4000]
    hits = sum(1 for ln in head.splitlines()[:60] if _WA_LINE.match(ln))
    return hits >= 3 or any(h in head for h in _WA_SYSTEM_HINTS) and hits >= 1


def parse_whatsapp(text: str) -> list[dict]:
    """WhatsApp .txt → список сообщений в формате Wiki.write_daily_raw."""
    msgs: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        m = _WA_LINE.match(line)
        if m:
            dt = _parse_wa_date(m.group("date"), m.group("time"))
            if dt is None:
                if cur is not None:
                    cur["text"] += "\n" + line
                continue
            cur = {
                "message_id": len(msgs) + 1,
                "chat_id": None, "topic_id": None, "reply_to_message_id": None,
                "full_name": m.group("name").strip().lstrip("‎"),
                "username": None, "user_id": None,
                "text": m.group("text").strip(),
                "ts": dt.isoformat(), "_dt": dt,
                "media_type": None, "media_path": None,
            }
            msgs.append(cur)
        elif cur is not None and line.strip():
            cur["text"] += "\n" + line
    # системные строки без автора отбрасываются регэкспом; медиа-заглушки WhatsApp помечаем
    for m in msgs:
        low = m["text"].lower()
        if "<media omitted>" in low or "<без медиафайлов>" in low or "медиафайл отсутствует" in low:
            m["media_type"] = "media (не вложено в экспорт)"
    return msgs


# ── Telegram JSON ─────────────────────────────────────────────────────────────

def parse_telegram_json(path: Path) -> list[dict]:
    import json
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    raw_msgs = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(raw_msgs, list):
        return []
    chat_id = data.get("id")
    if isinstance(chat_id, int) and chat_id > 0 and data.get("type") in ("private_supergroup", "public_supergroup", "private_channel", "public_channel"):
        chat_id = int(f"-100{chat_id}")
    out: list[dict] = []
    for m in raw_msgs:
        if m.get("type") != "message":
            continue
        text_field = m.get("text")
        if isinstance(text_field, list):
            text = "".join(seg if isinstance(seg, str) else seg.get("text", "") for seg in text_field)
        else:
            text = text_field or ""
        try:
            dt = datetime.strptime(m["date"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        media_type = None
        for key in ("photo", "file", "media_type", "sticker_emoji"):
            if m.get(key):
                media_type = m.get("media_type") or key
                break
        out.append({
            "message_id": m.get("id"), "chat_id": chat_id, "topic_id": None,
            "reply_to_message_id": m.get("reply_to_message_id"),
            "full_name": m.get("from") or m.get("actor") or "?", "username": None,
            "user_id": None, "text": text.strip(), "ts": dt.isoformat(), "_dt": dt,
            "media_type": media_type, "media_path": None,
        })
    return out


# ── запись по дням ────────────────────────────────────────────────────────────

def write_messages_as_daily(wiki: Wiki, msgs: list[dict], *, source: str, force: bool = False) -> list[str]:
    """Сгруппировать сообщения по дате и записать raw/daily/<date>.md (пропуская существующие)."""
    by_day: dict[str, list[dict]] = {}
    for m in msgs:
        day = m["_dt"].strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(m)
    written: list[str] = []
    for day in sorted(by_day):
        rel = f"raw/daily/{day}.md"
        if (wiki.root / rel).exists() and not force:
            continue
        clean = [{k: v for k, v in m.items() if k != "_dt"} for m in by_day[day]]
        wiki.write_daily_raw(day, clean)
        # пометим источник в шапке
        p = wiki.root / rel
        text = p.read_text(encoding="utf-8")
        text = text.replace("Дамп сообщений из рабочего чата за сутки.", f"Импорт: {source}.", 1)
        p.write_text(text, encoding="utf-8")
        written.append(day)
    return written


# ── главный вход ──────────────────────────────────────────────────────────────

def import_archive(wiki: Wiki, zip_rel: str) -> ArchiveInfo:
    """Распаковать архив из vault (rel путь) в raw/imports/<stem>/ и импортировать."""
    zip_abs = wiki.resolve(zip_rel)
    stem = re.sub(r"[^\w.-]+", "_", zip_abs.stem)[:60] or "archive"
    rel_root = f"raw/imports/{stem}"
    dst = wiki.root / rel_root
    dst.mkdir(parents=True, exist_ok=True)
    files = safe_extract(zip_abs, dst)
    info = ArchiveInfo(kind="generic", root=dst, rel_root=rel_root, files=files)
    info.text_files = [
        str(f.relative_to(wiki.root)).replace("\\", "/")
        for f in files if f.suffix.lower() in TEXT_EXTENSIONS | {".html", ".htm"}
    ]
    source = f"архив {zip_abs.name}"

    # Telegram JSON
    json_files = [f for f in files if f.name == "result.json"]
    if json_files:
        msgs = parse_telegram_json(json_files[0])
        if msgs:
            info.kind = "telegram_json"
            info.messages = len(msgs)
            info.dates = write_messages_as_daily(wiki, msgs, source=source)
            return info
    # Telegram HTML
    html_files = sorted(f for f in files if re.match(r"messages\d*\.html$", f.name))
    if html_files:
        try:
            from .import_tg_export import parse_html_file  # noqa: PLC0415
            msgs: list[dict] = []
            for hf in html_files:
                for m in parse_html_file(hf):
                    dt = m.get("dt")
                    if not dt:
                        continue
                    msgs.append({
                        "message_id": m.get("id"), "chat_id": None, "topic_id": None,
                        "reply_to_message_id": m.get("reply_to"), "full_name": m.get("from") or "?",
                        "username": None, "user_id": None, "text": m.get("text") or "",
                        "ts": dt.isoformat(), "_dt": dt,
                        "media_type": m.get("media_type"), "media_path": None,
                    })
            if msgs:
                info.kind = "telegram_html"
                info.messages = len(msgs)
                info.dates = write_messages_as_daily(wiki, msgs, source=source)
                return info
        except ImportError:
            info.notes.append("beautifulsoup4 не установлен — HTML-экспорт Telegram не разобран, файлы лежат как есть")
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram html import failed: %s", exc)
            info.notes.append(f"HTML-экспорт не разобран: {exc}")
    # WhatsApp txt (предпочитаем .txt, он структурнее .md)
    for f in sorted(files, key=lambda p: (p.suffix.lower() != ".txt", p.name)):
        if f.suffix.lower() not in (".txt", ".md"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if looks_like_whatsapp(text):
            msgs = parse_whatsapp(text)
            if msgs:
                info.kind = "whatsapp"
                info.messages = len(msgs)
                info.dates = write_messages_as_daily(wiki, msgs, source=f"WhatsApp, {f.name}")
                return info
    if not info.text_files:
        info.notes.append("текстовых файлов внутри нет — читать нечего")
    return info
