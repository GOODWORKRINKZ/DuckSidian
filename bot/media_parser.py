"""Извлечение текста из медиафайлов для ingest-пайплайна.

Поддерживаемые форматы:
- PDF         — pypdf (pure python, нет системных зависимостей)
- DOCX/ODT    — python-docx / odfpy (опц.)
- XLSX/ODS    — openpyxl (опц.)
- TXT/MD/CSV  — прямо
- Голосовые   — faster-whisper (STT, модель tiny)
- Изображения — DeepSeek VL через describe_image

Все зависимости опциональны — при отсутствии возвращается заглушка.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# ── PDF ──────────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import]
    except ImportError:
        return "(pypdf не установлен — pip install pypdf)"
    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[стр. {i + 1}]\n{text.strip()}")
        if not pages:
            return "(PDF без извлекаемого текста — возможно, сканированный)"
        total = sum(len(p) for p in pages)
        result = "\n\n".join(pages)
        if total > 8000:
            result = result[:8000] + f"\n...(обрезано, всего ~{total} символов)"
        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("PDF extract failed: %s", exc)
        return f"(ошибка чтения PDF: {exc})"


# ── DOCX ─────────────────────────────────────────────────────────────────────

def _extract_docx(path: Path) -> str:
    try:
        import docx  # type: ignore[import]
    except ImportError:
        return "(python-docx не установлен — pip install python-docx)"
    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        if len(text) > 8000:
            text = text[:8000] + "\n...(обрезано)"
        return text or "(документ пуст)"
    except Exception as exc:  # noqa: BLE001
        log.warning("DOCX extract failed: %s", exc)
        return f"(ошибка чтения DOCX: {exc})"


# ── XLSX ─────────────────────────────────────────────────────────────────────

def _extract_xlsx(path: Path) -> str:
    try:
        import openpyxl  # type: ignore[import]
    except ImportError:
        return "(openpyxl не установлен — pip install openpyxl)"
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        lines: list[str] = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            lines.append(f"## Лист: {sheet}")
            row_count = 0
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    lines.append("\t".join(cells))
                    row_count += 1
                    if row_count >= 200:
                        lines.append("...(строки обрезаны)")
                        break
        text = "\n".join(lines)
        if len(text) > 8000:
            text = text[:8000] + "\n...(обрезано)"
        return text or "(таблица пуста)"
    except Exception as exc:  # noqa: BLE001
        log.warning("XLSX extract failed: %s", exc)
        return f"(ошибка чтения XLSX: {exc})"


# ── PLAIN TEXT ────────────────────────────────────────────────────────────────

def _extract_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 8000:
            text = text[:8000] + f"\n...(обрезано, всего {len(text)} символов)"
        return text
    except Exception as exc:  # noqa: BLE001
        return f"(ошибка чтения файла: {exc})"


# ── VOICE / AUDIO STT ─────────────────────────────────────────────────────────

def _transcribe_audio_sync(path: Path) -> str:
    """Синхронная транскрипция через openai-whisper (модель tiny, ffmpeg-based)."""
    try:
        import whisper  # type: ignore[import]
    except ImportError:
        return "(openai-whisper не установлен — pip install openai-whisper)"
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(str(path))
        text = (result.get("text") or "").strip()
        lang = result.get("language", "?")
        log.info("STT %s: lang=%s text=%r", path.name, lang, text[:80])
        return text or "(пустая транскрипция)"
    except Exception as exc:  # noqa: BLE001
        log.warning("STT failed for %s: %s", path, exc)
        return f"(ошибка STT: {exc})"


async def transcribe_audio(path: Path) -> str:
    """Асинхронная обёртка для STT (запускает в thread-pool)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_audio_sync, path)


# ── MAIN ENTRY ────────────────────────────────────────────────────────────────

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".py",
                   ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".opus"}


async def extract_text_from_file(path: Path) -> str | None:
    """Попытаться извлечь текст из файла по расширению.

    Возвращает строку с текстом или None если тип не поддерживается.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract_pdf, path)

    if ext in {".docx"}:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract_docx, path)

    if ext in {".xlsx", ".xls"}:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract_xlsx, path)

    if ext in TEXT_EXTENSIONS:
        return _extract_text(path)

    if ext in AUDIO_EXTENSIONS:
        return await transcribe_audio(path)

    return None
