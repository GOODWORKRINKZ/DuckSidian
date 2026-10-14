"""Приём вложений в личке: документы, картинки, архивы → текст, доступный агенту.

Для каждого вложения создаётся sidecar `raw/imports/<имя>.md` с извлечённым
текстом (полный текст документа, OCR/описание картинки, транскрипт), чтобы
агент мог читать/грепать его позже, а не только в момент получения.

- PDF/DOCX/XLSX/PPTX/текст → извлечение текста (media_parser);
  PDF без текстового слоя → рендер страниц (pdftoppm) → vision-OCR (MiniMax-M3).
- Картинки → vision: описание + распознанный текст.
- ZIP → archive_import (экспорты WhatsApp/Telegram → raw/daily; прочее → файлы).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .archive_import import ARCHIVE_EXTENSIONS, ArchiveInfo, import_archive
from .media_parser import AUDIO_EXTENSIONS, TEXT_EXTENSIONS, VIDEO_EXTENSIONS, extract_text_from_file
from .wiki import Wiki

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
DOC_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".pptx"} | TEXT_EXTENSIONS
OCR_MAX_PAGES = 6
SIDECAR_MAX_CHARS = 200_000

_OCR_PROMPT = (
    "Это страница документа или изображение с текстом. Сначала ПОЛНОСТЬЮ перепиши весь "
    "читаемый текст (сохраняя структуру: заголовки, списки, таблицы — таблицы как markdown). "
    "Затем после строки '---' кратко (2–4 предложения) опиши, что это за документ/изображение "
    "и что на нём важного (схемы, графики, подписи). По-русски, если текст не на другом языке."
)
_IMG_PROMPT = (
    "Опиши изображение по делу (3–6 предложений): что показано, объекты/устройства/детали, "
    "контекст. Если есть текст, надписи, цифры, UI — процитируй их точно после строки "
    "'Текст на изображении:'. По-русски."
)


@dataclass
class IntakeResult:
    kind: str                      # archive | document | scan | image | audio | video | other
    rel: str                       # исходный файл (rel путь в vault)
    text: str = ""                 # извлечённый текст / описание (полный)
    sidecar: str | None = None     # rel путь .md с текстом
    archive: ArchiveInfo | None = None
    notes: list[str] = field(default_factory=list)

    def excerpt(self, n: int = 2500) -> str:
        t = self.text.strip()
        return t if len(t) <= n else t[:n].rsplit("\n", 1)[0] + f"\n…(всего {len(t)} символов, полный текст: `{self.sidecar}`)"

    def agent_block(self) -> str:
        """Блок для промпта агента."""
        head = {
            "archive": "📦 АРХИВ", "document": "📄 ДОКУМЕНТ", "scan": "📄 СКАН (OCR)",
            "image": "🖼 ИЗОБРАЖЕНИЕ", "audio": "🎤 АУДИО", "video": "🎬 ВИДЕО", "other": "📎 ФАЙЛ",
        }[self.kind]
        lines = [f"[{head}: `{self.rel}`" + (f" → текст сохранён в `{self.sidecar}`" if self.sidecar else "") + "]"]
        if self.archive is not None:
            lines.append(self.archive.summary_md())
            if self.archive.dates:
                lines.append("Сообщения уже разложены по raw/daily/<дата>.md — читай/грепай их при необходимости.")
        if self.text.strip():
            lines.append(self.excerpt())
        lines += [f"_{n}_" for n in self.notes]
        return "\n".join(lines)


def _sidecar_rel(rel: str) -> str:
    name = Path(rel).name
    safe = re.sub(r"[^\w.-]+", "_", name)[:80]
    return f"raw/imports/{safe}.md"


def _write_sidecar(wiki: Wiki, rel: str, kind: str, text: str, extra_fm: dict | None = None) -> str:
    side = _sidecar_rel(rel)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    fm = {"type": "extracted", "kind": kind, "source": rel, "extracted": now}
    fm.update(extra_fm or {})
    body = "---\n" + "\n".join(f"{k}: {v}" for k, v in fm.items()) + "\n---\n\n"
    body += f"# {Path(rel).name}\n\nИсточник: [[{rel}]]\n\n{text.strip()[:SIDECAR_MAX_CHARS]}\n"
    p = wiki.root / side
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return side


async def _render_pdf_pages(pdf: Path, max_pages: int = OCR_MAX_PAGES) -> list[Path]:
    """PDF → PNG страниц через pdftoppm (poppler). Пусто, если утилиты нет."""
    out_dir = Path(tempfile.mkdtemp(prefix="ocr_"))
    try:
        proc = await asyncio.create_subprocess_exec(
            "pdftoppm", "-r", "110", "-png", "-f", "1", "-l", str(max_pages),
            str(pdf), str(out_dir / "page"),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            log.warning("pdftoppm failed rc=%s: %s", proc.returncode, (err or b"")[-200:])
            return []
    except Exception as exc:  # noqa: BLE001
        log.info("pdftoppm unavailable: %s", exc)
        return []
    return sorted(out_dir.glob("page*.png"))


async def _vision(path: Path, prompt: str) -> str | None:
    """Описание/OCR картинки через MiniMax vision (если доступен)."""
    try:
        from .media_ai import get_media_ai  # noqa: PLC0415
        ai = get_media_ai()
        if not ai.vision_enabled:
            return None
        return await ai.describe_image_file(path, prompt=prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("vision failed for %s: %s", path.name, exc)
        return None


async def _ocr_pdf(pdf: Path) -> str | None:
    pages = await _render_pdf_pages(pdf)
    if not pages:
        return None
    parts: list[str] = []
    for i, pg in enumerate(pages, 1):
        txt = await _vision(pg, _OCR_PROMPT)
        if txt:
            parts.append(f"## Страница {i}\n\n{txt.strip()}")
        try:
            pg.unlink()
        except OSError:
            pass
    try:
        pages[0].parent.rmdir()
    except OSError:
        pass
    return "\n\n".join(parts) if parts else None


async def process_attachment(wiki: Wiki, rel: str, media_type: str | None = None) -> IntakeResult:
    """Главный вход: разобрать вложение из vault и вернуть текст + sidecar."""
    abs_path = wiki.resolve(rel)
    ext = abs_path.suffix.lower()
    size_kb = abs_path.stat().st_size // 1024

    if ext in ARCHIVE_EXTENSIONS:
        res = IntakeResult(kind="archive", rel=rel)
        try:
            info = import_archive(wiki, rel)
            res.archive = info
            # Для «обычных» архивов — вытащим текст из вложенных документов (первые 10).
            if info.kind == "generic":
                chunks: list[str] = []
                for f in info.files[:10]:
                    if f.suffix.lower() in DOC_EXTENSIONS:
                        try:
                            t = await extract_text_from_file(f, limit=20_000)
                        except TypeError:
                            t = await extract_text_from_file(f)
                        if t and not t.startswith("("):
                            frel = str(f.relative_to(wiki.root)).replace("\\", "/")
                            chunks.append(f"### {frel}\n\n{t.strip()}")
                if chunks:
                    res.text = "\n\n".join(chunks)
                    res.sidecar = _write_sidecar(wiki, rel, "archive", res.text,
                                                 {"files": len(info.files)})
        except Exception as exc:  # noqa: BLE001
            log.exception("archive import failed")
            res.notes.append(f"не удалось распаковать: {exc}")
        return res

    if ext in IMAGE_EXTENSIONS or media_type in {"photo", "sticker", "animation"}:
        res = IntakeResult(kind="image", rel=rel)
        desc = await _vision(abs_path, _IMG_PROMPT)
        if desc:
            res.text = desc
            res.sidecar = _write_sidecar(wiki, rel, "image", desc)
        else:
            res.notes.append(f"изображение {ext}, {size_kb} KB — vision недоступен")
        return res

    if ext in DOC_EXTENSIONS or media_type == "document":
        res = IntakeResult(kind="document", rel=rel)
        try:
            text = await extract_text_from_file(abs_path, limit=SIDECAR_MAX_CHARS)
        except TypeError:
            text = await extract_text_from_file(abs_path)
        if ext == ".pdf" and (not text or text.startswith("(")):
            ocr = await _ocr_pdf(abs_path)
            if ocr:
                res.kind = "scan"
                text = ocr
                res.notes.append(f"текстового слоя нет — распознано первых {OCR_MAX_PAGES} страниц")
        if text and not text.startswith("("):
            res.text = text
            res.sidecar = _write_sidecar(wiki, rel, res.kind, text)
        elif text:
            res.notes.append(text.strip("()"))
        else:
            res.kind = "other"
            res.notes.append(f"формат {ext or '?'} не поддерживается ({size_kb} KB)")
        return res

    if ext in AUDIO_EXTENSIONS or media_type in {"voice", "audio"}:
        res = IntakeResult(kind="audio", rel=rel)
        text = await extract_text_from_file(abs_path)
        if text and not text.startswith("("):
            res.text = text
            res.sidecar = _write_sidecar(wiki, rel, "audio", text)
        elif text:
            res.notes.append(text.strip("()"))
        return res

    if ext in VIDEO_EXTENSIONS or media_type in {"video", "video_note"}:
        from .media_parser import transcribe_video  # noqa: PLC0415
        res = IntakeResult(kind="video", rel=rel)
        text = await transcribe_video(abs_path)
        if text and not text.startswith("("):
            res.text = text
            res.sidecar = _write_sidecar(wiki, rel, "video", text)
        elif text:
            res.notes.append(text.strip("()"))
        return res

    return IntakeResult(kind="other", rel=rel, notes=[f"бинарный файл {ext or '?'}, {size_kb} KB"])


def b64_of(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


FILE_URL_RE = re.compile(
    r"https?://[^\s<>\"']+\.(?:zip|pdf|docx?|xlsx?|pptx?|txt|md|csv|json|html?)(?:\?[^\s<>\"']*)?",
    re.IGNORECASE,
)
DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024


def filename_from_url(url: str, content_disposition: str | None = None, content_type: str | None = None) -> str:
    """Имя файла из Content-Disposition / пути URL / content-type."""
    import urllib.parse
    name = ""
    if content_disposition:
        m = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition) or re.search(r'filename="?([^";]+)"?', content_disposition)
        if m:
            name = urllib.parse.unquote(m.group(1).strip())
    if not name:
        path = urllib.parse.urlparse(url).path
        name = urllib.parse.unquote(path.rsplit("/", 1)[-1]) if path else ""
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._") or "download"
    if "." not in name and content_type:
        ext = {"application/zip": ".zip", "application/pdf": ".pdf", "text/plain": ".txt",
               "application/json": ".json", "text/html": ".html"}.get(content_type.split(";")[0].strip())
        if ext:
            name += ext
    return name[:100]


async def download_url_to_vault(wiki: Wiki, url: str, *, max_bytes: int = DOWNLOAD_MAX_BYTES) -> str:
    """Скачать файл по URL (стримом) в raw/assets/<date>/. Возвращает rel путь.

    Обход лимита Telegram (боты не получают файлы > 20 МБ): пользователь кладёт
    файл в облако и присылает прямую ссылку.
    """
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DuckSidian/1.0)"}
    date_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20, read=120, write=30, pool=10),
                                 follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if ct.split(";")[0].strip() == "text/html" and not url.lower().split("?")[0].endswith((".html", ".htm")):
                raise ValueError("по ссылке отдаётся HTML-страница, а не файл — нужна ПРЯМАЯ ссылка на скачивание")
            name = filename_from_url(str(resp.url), resp.headers.get("content-disposition"), ct)
            safe = "".join(c for c in name if c.isalnum() or c in "._-") or "download.bin"
            dst = wiki.root / "raw" / "assets" / date_iso / safe
            dst.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with dst.open("wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        f.close()
                        dst.unlink(missing_ok=True)
                        raise ValueError(f"файл больше {max_bytes // 1024 // 1024} МБ")
                    f.write(chunk)
    log.info("downloaded %s -> %s (%d KB)", url, dst.name, total // 1024)
    return str(dst.relative_to(wiki.root)).replace("\\", "/")
