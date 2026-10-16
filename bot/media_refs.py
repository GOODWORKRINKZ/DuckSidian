"""Блок IMAGES: в ответе агента — поиск, разбор путей, проверка файлов в vault.

Модель оформляет блок по-разному: «IMAGES:», «**IMAGES:**», «IMAGES :», «## IMAGES:»;
пути — голые, в бэктиках, в [[ ]] / ![[ ]], с «- » в начале. Всё это должно
распознаваться одинаково и в агенте (guard), и в хендлерах (отправка файлов).
"""
from __future__ import annotations

import re
from pathlib import Path

# Строка-маркер блока: допускаем markdown-обрамление (**, __, #, >) и пробел перед «:».
IMAGES_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(?:[#>]+[ \t]*)?[*_]*[ \t]*IMAGES[ \t]*[*_]*[ \t]*:[ \t]*[*_]*[ \t]*$"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
GIF_EXTS = {".gif"}
MEDIA_EXTS = IMG_EXTS | VID_EXTS | GIF_EXTS

_EXT_ALT = "|".join(sorted((e[1:] for e in MEDIA_EXTS), key=len, reverse=True))
# Путь медиафайла внутри vault: без пробелов/скобок/кавычек, с медиа-расширением.
_PATH_RE = re.compile(
    rf"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:{_EXT_ALT}))(?![\w])",
    re.IGNORECASE | re.UNICODE,
)
_WHOLE_PATH_RE = re.compile(rf"[^\s`<>|\"']+(?: [^\s`<>|\"']+)*\.(?:{_EXT_ALT})", re.IGNORECASE)
# Любое упоминание raw/assets/… в тексте (вне блока IMAGES).
_ASSET_RE = re.compile(rf"raw/assets/[^\s`\)\]\}}\"'<>|]+?\.(?:{_EXT_ALT})(?![\w])", re.IGNORECASE)


def find_images_marker(answer: str) -> re.Match[str] | None:
    """Последний маркер блока IMAGES (блок всегда в конце ответа)."""
    last = None
    for m in IMAGES_MARKER_RE.finditer(answer or ""):
        last = m
    return last


# Блок VOICE: — текст, который бот озвучит дословно и пришлёт голосовым (а не краткий
# пересказ ответа). Оформление такое же терпимое, как у IMAGES.
VOICE_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(?:[#>]+[ \t]*)?[*_]*[ \t]*VOICE[ \t]*[*_]*[ \t]*:[ \t]*[*_]*[ \t]*$"
)


def find_voice_marker(answer: str) -> re.Match[str] | None:
    """Первый маркер блока VOICE (после него — только текст для озвучки и, возможно, IMAGES)."""
    return VOICE_MARKER_RE.search(answer or "")


def find_tail_marker(answer: str) -> re.Match[str] | None:
    """Самый ранний из служебных маркеров (IMAGES / VOICE): всё, что бот дописывает
    к ответу (футер «Изменено», пометки), должно встать ДО него."""
    cands = [m for m in (find_images_marker(answer), find_voice_marker(answer)) if m]
    return min(cands, key=lambda m: m.start()) if cands else None


def split_voice_block(answer: str) -> tuple[str, str | None]:
    """(ответ без блока VOICE, текст для озвучки или None). Блок VOICE тянется до
    маркера IMAGES (если он идёт после) или до конца; IMAGES при этом остаётся в ответе."""
    a = answer or ""
    m = find_voice_marker(a)
    if not m:
        return a.strip(), None
    tail = a[m.end():]
    im = find_images_marker(tail)
    if im:
        speech, rest = tail[: im.start()], tail[im.start():]
    else:
        speech, rest = tail, ""
    speech = speech.strip().strip("`").strip()
    text = a[: m.start()].rstrip()
    if rest.strip():
        text = f"{text}\n\n{rest.strip()}"
    return text.strip(), (speech or None)


def has_images_block(answer: str) -> bool:
    return find_images_marker(answer) is not None


def strip_images_block(answer: str) -> tuple[str, bool]:
    """(текст без блока IMAGES, был ли блок)."""
    m = find_images_marker(answer)
    if not m:
        return (answer or "").strip(), False
    return answer[: m.start()].rstrip(), True


def _clean_line(line: str) -> str:
    line = line.strip().lstrip("-*•+").strip()
    line = line.strip("`").strip()
    if line.startswith("!"):
        line = line[1:]
    if line.startswith("[[") and "]]" in line:
        line = line[2:line.index("]]")].split("|", 1)[0]
    return line.strip().strip("`").strip().lstrip("/")


def parse_images_block(answer: str) -> list[str]:
    """Относительные пути из блока IMAGES (по порядку, без дублей). Блок
    заканчивается пустой строкой после путей или первой строкой без пути."""
    m = find_images_marker(answer)
    if not m:
        return []
    rels: list[str] = []
    for raw_line in answer[m.end():].splitlines():
        line = _clean_line(raw_line)
        if not line:
            if rels:
                break
            continue
        # Вся строка — путь (в т.ч. с пробелами внутри [[ ]] / бэктиков).
        if _WHOLE_PATH_RE.fullmatch(line):
            rel = line
        elif pm := _PATH_RE.search(line):
            rel = pm.group(1)
        else:
            if re.fullmatch(r"\S+\.\w{2,5}", line):   # путь к не-медиа (pdf, …) — пропускаем
                continue
            break
        rel = rel[2:] if rel.startswith("./") else rel
        if rel not in rels:
            rels.append(rel)
    return rels


def resolve_in_vault(rel: str, root: Path) -> Path | None:
    """Абсолютный путь файла в vault или None (нет файла / выход за пределы vault)."""
    try:
        root = Path(root).resolve()
        target = (root / rel).resolve()
        target.relative_to(root)
    except (OSError, ValueError):
        return None
    return target if target.is_file() else None


def split_existing(rels: list[str], root: Path) -> tuple[list[Path], list[str]]:
    """(существующие файлы, пути, которых в vault нет)."""
    found: list[Path] = []
    missing: list[str] = []
    for rel in rels:
        p = resolve_in_vault(rel, root)
        if p is None:
            missing.append(rel)
        elif p not in found:
            found.append(p)
    return found, missing


def mentioned_media(answer: str) -> list[str]:
    """Все медиапути ответа: из блока IMAGES и упоминания raw/assets/… в тексте."""
    rels = parse_images_block(answer)
    for m in _ASSET_RE.finditer(answer or ""):
        if m.group(0) not in rels:
            rels.append(m.group(0))
    return rels


def missing_note(missing: list[str]) -> str:
    """Одна строка для пользователя о файлах, которых нет в vault."""
    names = ", ".join(missing[:5])
    more = f" и ещё {len(missing) - 5}" if len(missing) > 5 else ""
    return f"⚠️ Не нашёл в vault файлы: {names}{more} — не отправляю."
