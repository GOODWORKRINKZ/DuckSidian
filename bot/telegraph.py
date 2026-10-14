"""Публикация markdown-страниц vault'а в Telegraph (telegra.ph) для красивого
просмотра в Telegram (Instant View).

Внимание: страницы Telegraph публичны по ссылке (без индексации, но URL угадать
нельзя). Включается TELEGRAPH_ENABLED; access_token создаётся один раз и
хранится в state БД.

Конвертер — подмножество markdown: заголовки, списки, цитаты, код, жирный/курсив,
ссылки, wikilinks ([[X]] → жирный текст), embed'ы медиа ([[raw/assets/…]] → 📎 имя).
"""
from __future__ import annotations

import html
import logging
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

TELEGRAPH_API = "https://api.telegra.ph"
_MAX_CONTENT_BYTES = 60_000  # лимит Telegraph на content — 64 KB

Node = Any  # str | dict


def _strip_frontmatter(md: str) -> tuple[dict[str, str], str]:
    fm: dict[str, str] = {}
    if md.startswith("---\n"):
        end = md.find("\n---", 4)
        if end > 0:
            for line in md[4:end].splitlines():
                if ":" in line and not line.startswith(" "):
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip().strip('"').strip("'")
            md = md[end + 4:].lstrip("\n")
    return fm, md


_INLINE_RE = re.compile(
    r"(?P<code>`[^`\n]+`)"
    r"|(?P<embed>!\[\[[^\]]+\]\])"
    r"|(?P<wiki>\[\[[^\]]+\]\])"
    r"|(?P<link>\[[^\]\n]+\]\([^)\s]+\))"
    r"|(?P<bold>\*\*[^*\n]+\*\*)"
    r"|(?P<ital>(?<![\w*])\*[^*\n]+\*(?![\w*])|(?<!\w)_[^_\n]+_(?!\w))"
    r"|(?P<url>https?://[^\s<>\")\]]+)"
)


def _inline(text: str) -> list[Node]:
    """Инлайн-разметка → список узлов Telegraph."""
    out: list[Node] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        kind = m.lastgroup
        raw = m.group(0)
        if kind == "code":
            out.append({"tag": "code", "children": [raw[1:-1]]})
        elif kind == "embed":
            inner = raw[3:-2].split("|", 1)[0]
            out.append({"tag": "i", "children": [f"📎 {inner.rsplit('/', 1)[-1]}"]})
        elif kind == "wiki":
            inner = raw[2:-2]
            target, _, alias = inner.partition("|")
            label = (alias or target.rsplit("/", 1)[-1]).split("#", 1)[0] or target
            if target.startswith("raw/assets/"):
                out.append({"tag": "i", "children": [f"📎 {label}"]})
            else:
                out.append({"tag": "b", "children": [label]})
        elif kind == "link":
            label, _, href = raw[1:-1].partition("](")
            href = href.rstrip(")")
            if href.startswith(("http://", "https://", "tg://")):
                out.append({"tag": "a", "attrs": {"href": href}, "children": _inline(label)})
            else:
                out.append(label)
        elif kind == "bold":
            out.append({"tag": "b", "children": _inline(raw[2:-2])})
        elif kind == "ital":
            out.append({"tag": "i", "children": _inline(raw[1:-1])})
        elif kind == "url":
            out.append({"tag": "a", "attrs": {"href": raw}, "children": [raw]})
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return [n for n in out if n != ""]


def md_to_nodes(md: str) -> list[Node]:
    """Markdown → узлы Telegraph (p, h3, h4, ul/ol/li, blockquote, pre, hr, b, i, a, code)."""
    _, body = _strip_frontmatter(md)
    nodes: list[Node] = []
    para: list[str] = []
    list_tag: str | None = None
    list_items: list[Node] = []
    quote: list[str] = []
    in_code = False
    code_lines: list[str] = []
    first_h1_skipped = False

    def flush_para() -> None:
        nonlocal para
        if para:
            nodes.append({"tag": "p", "children": _inline(" ".join(s.strip() for s in para))})
            para = []

    def flush_list() -> None:
        nonlocal list_tag, list_items
        if list_tag and list_items:
            nodes.append({"tag": list_tag, "children": list_items})
        list_tag, list_items = None, []

    def flush_quote() -> None:
        nonlocal quote
        if quote:
            nodes.append({"tag": "blockquote", "children": _inline(" ".join(quote))})
            quote = []

    def flush_all() -> None:
        flush_para(); flush_list(); flush_quote()

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if in_code:
            if line.strip().startswith("```"):
                nodes.append({"tag": "pre", "children": ["\n".join(code_lines)]})
                code_lines, in_code = [], False
            else:
                code_lines.append(raw_line)
            continue
        if line.strip().startswith("```"):
            flush_all(); in_code = True
            continue
        if not line.strip():
            flush_all()
            continue
        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", line):
            flush_all(); nodes.append({"tag": "hr"})
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_all()
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1 and not first_h1_skipped:
                first_h1_skipped = True  # заголовок страницы уже в title
                continue
            nodes.append({"tag": "h3" if level <= 2 else "h4", "children": _inline(text)})
            continue
        m = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            flush_para(); flush_quote()
            tag = "ol" if re.match(r"^\s*\d", line) else "ul"
            if list_tag and list_tag != tag:
                flush_list()
            list_tag = tag
            item = m.group(1)
            item = re.sub(r"^\[( |x|X)\]\s*", lambda mm: "☐ " if mm.group(1) == " " else "☑ ", item)
            list_items.append({"tag": "li", "children": _inline(item)})
            continue
        if line.lstrip().startswith(">"):
            flush_para(); flush_list()
            quote.append(line.lstrip()[1:].strip())
            continue
        if list_tag and line.startswith(("  ", "\t")):
            # продолжение пункта списка
            list_items[-1]["children"].extend([" "] + _inline(line.strip()))
            continue
        flush_list(); flush_quote()
        para.append(line)
    if in_code and code_lines:
        nodes.append({"tag": "pre", "children": ["\n".join(code_lines)]})
    flush_all()
    return nodes or [{"tag": "p", "children": ["(пусто)"]}]


def page_title(md: str, fallback: str) -> str:
    fm, body = _strip_frontmatter(md)
    if fm.get("title"):
        return fm["title"][:256]
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()[:256]
    return fallback[:256]


def _trim_nodes(nodes: list[Node], limit: int = _MAX_CONTENT_BYTES) -> list[Node]:
    import json
    out: list[Node] = []
    size = 2
    for n in nodes:
        s = len(json.dumps(n, ensure_ascii=False).encode("utf-8")) + 1
        if size + s > limit:
            out.append({"tag": "p", "children": [{"tag": "i", "children": ["… (обрезано: страница слишком длинная)"]}]})
            break
        out.append(n); size += s
    return out


class TelegraphClient:
    """Минимальный клиент Telegraph. Токен аккаунта хранится в db.state['telegraph_token']."""

    def __init__(self, db: Any, short_name: str = "DuckSidian", author_name: str = "DuckSidian"):
        self._db = db
        self._short = short_name
        self._author = author_name
        self._token: str | None = None

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token
        tok = await self._db.get_state("telegraph_token", "")
        if not tok:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.post(f"{TELEGRAPH_API}/createAccount",
                                 data={"short_name": self._short, "author_name": self._author})
                r.raise_for_status()
                d = r.json()
            if not d.get("ok"):
                raise RuntimeError(f"telegraph createAccount: {d.get('error')}")
            tok = d["result"]["access_token"]
            await self._db.set_state("telegraph_token", tok)
        self._token = tok
        return tok

    async def publish_markdown(self, md: str, *, title: str, author: str | None = None) -> str:
        """Создать страницу из markdown. Возвращает URL."""
        import json
        token = await self._ensure_token()
        nodes = _trim_nodes(md_to_nodes(md))
        payload = {
            "access_token": token,
            "title": html.unescape(title)[:256] or "Без названия",
            "author_name": (author or self._author)[:128],
            "content": json.dumps(nodes, ensure_ascii=False),
            "return_content": "false",
        }
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{TELEGRAPH_API}/createPage", data=payload)
            r.raise_for_status()
            d = r.json()
        if not d.get("ok"):
            raise RuntimeError(f"telegraph createPage: {d.get('error')}")
        url = d["result"]["url"]
        log.info("telegraph page: %s (%d nodes)", url, len(nodes))
        return url
