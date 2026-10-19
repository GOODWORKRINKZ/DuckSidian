"""Чистые хелперы для правки страниц vault: оглавление, устойчивый str-replace,
вставка в секцию, проверка wikilinks и поиск похожих страниц. Без I/O агента —
вызываются из ToolExecutor и тестируются отдельно."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# Шаблонные заглушки из AGENTS.md / _template.md — не ссылки.
_PLACEHOLDER_RE = re.compile(r"YYYY|<[^>]+>|\.\.\.|^slug$|^Slug$|^имя", re.IGNORECASE)


def outline(text: str) -> list[tuple[int, str]]:
    """Заголовки markdown (вне ``` блоков) с номерами строк (1-based)."""
    out: list[tuple[int, str]] = []
    in_fence = False
    for i, ln in enumerate(text.splitlines(), 1):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(ln)
        if m:
            out.append((i, f"{m.group(1)} {m.group(2)}"))
    return out


def big_file_view(path: str, text: str, head_chars: int) -> str:
    """Начало большого файла + полное оглавление с номерами строк — чтобы агент
    дочитывал нужные секции через read_lines/grep_file, а не гадал old_string."""
    lines = text.splitlines()
    head = text[:head_chars]
    cut = head.rfind("\n")
    if cut > head_chars // 2:
        head = head[:cut]
    toc = "\n".join(f"{n:>5}: {h}" for n, h in outline(text)) or "(заголовков нет)"
    return (
        f"[{path}: {len(text)} симв., {len(lines)} строк — БОЛЬШОЙ ФАЙЛ, ниже только начало. "
        "Остальное читай read_lines(path, start, end) по оглавлению или grep_file. "
        "Дописать в секцию — append_to_section(path, heading, content), без old_string.]\n\n"
        f"{head}\n\n...[дальше не показано]\n\n=== ОГЛАВЛЕНИЕ (строка: заголовок) ===\n{toc}"
    )


def _norm_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace(" ", " ").replace("​", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def fuzzy_replace(text: str, old: str, new: str) -> tuple[str | None, str]:
    """str-replace, устойчивый к пробелам/переносам/NBSP. Возвращает
    (новый текст | None, пояснение). Точное совпадение — как обычно; иначе ищем
    единственный фрагмент из тех же строк, совпадающий после нормализации пробелов."""
    cnt = text.count(old)
    if cnt == 1:
        return text.replace(old, new, 1), "exact"
    if cnt > 1:
        return None, (f"old_string встречается {cnt} раз. Добавь соседние строки, "
                      "чтобы фрагмент был уникален.")
    target = _norm_ws(old)
    if not target:
        return None, "old_string пустой после нормализации."
    lines = text.splitlines(keepends=True)
    n_old = max(1, len(old.strip().splitlines()))
    hits: list[tuple[int, int]] = []
    for size in {n_old, n_old - 1, n_old + 1} - {0}:
        for i in range(0, len(lines) - size + 1):
            # Окно с пустой крайней строкой нормализуется так же, как без неё — не дубль.
            if size > 1 and (not lines[i].strip() or not lines[i + size - 1].strip()):
                continue
            if _norm_ws("".join(lines[i:i + size])) == target:
                hits.append((i, i + size))
    hits = sorted(set(hits))
    if len(hits) == 1:
        a, b = hits[0]
        tail = "\n" if "".join(lines[a:b]).endswith("\n") and not new.endswith("\n") else ""
        return "".join(lines[:a]) + new + tail + "".join(lines[b:]), "whitespace-normalized"
    if len(hits) > 1:
        return None, f"old_string (без учёта пробелов) встречается {len(hits)} раз — добавь контекст."
    return None, _closest_hint(lines, old)


def _closest_hint(lines: list[str], old: str) -> str:
    first = next((ln.strip() for ln in old.splitlines() if ln.strip()), "")
    best = difflib.get_close_matches(first, [ln.strip() for ln in lines], n=1, cutoff=0.5)
    msg = "old_string не найден в файле (даже без учёта пробелов)."
    if best:
        idx = next(i for i, ln in enumerate(lines) if ln.strip() == best[0])
        a, b = max(0, idx - 2), min(len(lines), idx + 3)
        ctx = "".join(f"{i + 1:>5}: {lines[i]}" for i in range(a, b)).rstrip()
        msg += f" Похожее место — строки {a + 1}–{b}:\n{ctx}\nСкопируй old_string ровно отсюда."
    else:
        msg += " Прочитай нужный кусок через read_lines/grep_file или используй append_to_section."
    return msg


def append_to_section(text: str, heading: str, content: str) -> tuple[str, str]:
    """Дописать content в конец секции `heading` (сравнение без #, регистра и эмодзи-
    префиксов). Секции нет — создаётся в конце файла уровнем ##. Возвращает
    (новый текст, пояснение)."""
    want = _heading_key(heading)
    lines = text.splitlines()
    start = level = None
    for i, ln in enumerate(lines):
        m = _HEADING_RE.match(ln)
        if m and _heading_key(m.group(2)) == want:
            start, level = i, len(m.group(1))
            break
    body = content.rstrip("\n")
    if start is None:
        name = re.sub(r"^#+\s*", "", heading).strip()
        sep = "" if (not text or text.endswith("\n\n")) else ("\n" if text.endswith("\n") else "\n\n")
        return f"{text}{sep}## {name}\n{body}\n", f"секция «{name}» создана в конце файла"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = _HEADING_RE.match(lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    # Вставляем после последней непустой строки секции.
    ins = end
    while ins > start + 1 and not lines[ins - 1].strip():
        ins -= 1
    new_lines = lines[:ins] + body.split("\n") + ([""] if end < len(lines) else []) + lines[end:]
    # Не плодим пустые строки подряд на стыке.
    out = "\n".join(new_lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out + ("\n" if text.endswith("\n") or not text else ""), f"дописано в «{lines[start].strip()}»"


def replace_section(text: str, heading: str, content: str) -> tuple[str, str]:
    """Заменить тело секции `heading` на content (заголовок остаётся). Секции нет —
    она создаётся сразу под H1 и вводной строкой (если есть), иначе в конце файла.
    Возвращает (новый текст, пояснение)."""
    want = _heading_key(heading)
    lines = text.splitlines()
    body = content.strip("\n").split("\n") if content.strip() else []
    start = level = None
    for i, ln in enumerate(lines):
        m = _HEADING_RE.match(ln)
        if m and _heading_key(m.group(2)) == want:
            start, level = i, len(m.group(1))
            break
    if start is None:
        name = re.sub(r"^#+\s*", "", heading).strip()
        ins = len(lines)
        for i, ln in enumerate(lines):
            if re.match(r"^#\s+\S", ln):
                ins = i + 1
                while ins < len(lines) and not lines[ins].strip():
                    ins += 1
                # вводная строка под H1 — остаётся сверху, секция идёт после неё
                if ins < len(lines) and not _HEADING_RE.match(lines[ins]):
                    ins += 1
                break
        new_lines = lines[:ins] + ["", f"## {name}"] + body + [""] + lines[ins:]
        out = re.sub(r"\n{3,}", "\n\n", "\n".join(new_lines)).strip("\n") + "\n"
        return out, f"секция «{name}» создана"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = _HEADING_RE.match(lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    new_lines = lines[:start + 1] + body + ([""] if end < len(lines) else []) + lines[end:]
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(new_lines)).strip("\n") + "\n"
    return out, f"секция «{lines[start].strip()}» заменена"


def instruction_body(section_text: str) -> str:
    """Тело секции «Инструкция» без заголовка и без шаблонной подсказки (курсивный абзац
    из _template.md). Пусто → '' (инструкции нет)."""
    lines = [ln for ln in (section_text or "").splitlines() if not _HEADING_RE.match(ln)]
    body = "\n".join(lines).strip()
    # Шаблонная подсказка: один абзац целиком в _курсиве_ — не инструкция.
    body = re.sub(r"\A_[^_]*?(?:set_workspace_instruction|/ws prompt)[^_]*?_\s*", "", body, flags=re.DOTALL)
    return body.strip()


def _heading_key(h: str) -> str:
    h = re.sub(r"^#+\s*", "", h).strip().lower()
    h = re.sub(r"^[^\w]+", "", h)  # эмодзи/символы в начале
    return re.sub(r"\s+", " ", h)


def wikilinks(text: str) -> list[str]:
    """Цели [[wikilinks]] вне блоков кода (без #якоря и |алиаса)."""
    text = _FENCE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]


def _vault_names(root: Path) -> set[str]:
    """Все имена, на которые может указывать wikilink: rel-путь без .md и basename."""
    names: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            rel = str(p.relative_to(root)).replace("\\", "/")
            names.update({rel, rel.removesuffix(".md"), p.name, p.stem})
    return names


def broken_links(text: str, root: Path, limit: int = 20) -> list[str]:
    """Wikilinks, которые не ведут ни на один файл этого vault'а."""
    targets = [t for t in dict.fromkeys(wikilinks(text)) if not _PLACEHOLDER_RE.search(t)]
    if not targets:
        return []
    names = _vault_names(root)
    out: list[str] = []
    for t in targets:
        key = t.strip().lstrip("/").removesuffix(".md")
        base = key.rsplit("/", 1)[-1]
        if key and key not in names and base not in names:
            out.append(t)
            if len(out) >= limit:
                break
    return out


_STOP = {"the", "and", "for", "biznes", "wiki", "research", "sources", "notes", "md",
         "kak", "pro", "dlya", "iz", "v", "na", "po", "i", "s", "o", "ot", "do"}


def _tokens(name: str) -> set[str]:
    name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", name.lower())
    return {t for t in re.split(r"[^\w]+", name) if len(t) > 2 and t not in _STOP and not t.isdigit()}


def similar_pages(root: Path, rel: str, limit: int = 5) -> list[str]:
    """Уже существующие страницы wiki/ с похожим slug (по пересечению слов) —
    кандидаты дополнить вместо создания дубля."""
    me = _tokens(Path(rel).stem)
    if len(me) < 2:
        return []
    scored: list[tuple[float, str]] = []
    wiki = root / "wiki"
    if not wiki.is_dir():
        return []
    for p in wiki.rglob("*.md"):
        r = str(p.relative_to(root)).replace("\\", "/")
        if r == rel or p.stem.startswith("_"):
            continue
        other = _tokens(p.stem)
        if not other:
            continue
        inter = len(me & other)
        score = inter / min(len(me), len(other))
        if inter >= 2 and score >= 0.5:
            scored.append((score, r))
    scored.sort(reverse=True)
    return [r for _, r in scored[:limit]]


def sections(text: str, headings: list[str], cap: int) -> str:
    """Текст секций с заданными заголовками (по порядку в файле, с подсекциями),
    суммарно не длиннее cap. Для «закреплённого» контекста страницы проекта."""
    want = {_heading_key(h) for h in headings}
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _HEADING_RE.match(lines[i])
        if m and _heading_key(m.group(2)) in want:
            level = len(m.group(1))
            j = i + 1
            while j < len(lines):
                mj = _HEADING_RE.match(lines[j])
                if mj and len(mj.group(1)) <= level:
                    break
                j += 1
            out.append("\n".join(lines[i:j]).strip())
            i = j
            continue
        i += 1
    joined = "\n\n".join(out)
    if len(joined) > cap:
        joined = joined[:cap].rsplit("\n", 1)[0] + "\n...[обрезано — полностью в файле проекта]"
    return joined
