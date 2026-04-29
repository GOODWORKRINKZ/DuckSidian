"""Markdown → Telegram HTML converter.

Telegram Bot API понимает только подмножество HTML-тегов:
<b>, <i>, <u>, <s>, <a>, <code>, <pre>, <blockquote>, <span class="tg-spoiler">.
См. https://core.telegram.org/bots/api#html-style и
    https://core.telegram.org/api/entities

DeepSeek-агент возвращает обычный Markdown (заголовки, списки, обсидиан
wiki-ссылки [[...]], кавычки, code fences). Этот модуль превращает такой
текст в безопасный Telegram-HTML, чтобы:
  * экранировать &, <, > в обычном тексте;
  * заголовки `## …` / `### …` стали жирными строками;
  * `**bold**`, `*italic*`, `_italic_`, `~~strike~~` → соответствующие теги;
  * inline ``code`` и тройные ```fences``` → <code>/<pre>;
  * `[text](url)` → <a href="url">text</a>;
  * `[[wiki/path/Name#anchor]]` → курсив с именем (anchor-ссылок Telegram
    не поддерживает, поэтому отображаем читаемый алиас);
  * `> quote` → <blockquote>;
  * `---` / `***` → пустая строка;
  * списки `- ` / `* ` → `• `;
  * не-распарсенные одиночные `*` / `_` не ломают отправку.
"""
from __future__ import annotations

import html
import re

__all__ = ["md_to_tg_html", "split_tg_chunks", "TG_LIMIT"]

TG_LIMIT = 4096

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
_WIKI_RE = re.compile(r"\[\[([^\]\n|]+)(?:\|([^\]\n]+))?\]\]")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"(?<![\*\w])\*(?!\s)([^\*\n]+?)(?<!\s)\*(?![\*\w])")
_ITALIC_UND_RE = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_LIST_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")


def _wiki_alias(target: str, alias: str | None) -> str:
    """Превратить [[wiki/...#anchor]] в человекочитаемый алиас."""
    if alias:
        return alias.strip()
    # отбросить #anchor
    base = target.split("#", 1)[0]
    # последний компонент пути, без .md
    name = base.rsplit("/", 1)[-1]
    if name.endswith(".md"):
        name = name[:-3]
    return name.strip() or target.strip()


def md_to_tg_html(text: str) -> str:
    """Конвертировать markdown-строку в Telegram-совместимый HTML."""
    if not text:
        return ""

    # 1) Защитить code fences и inline code от любых других замен.
    placeholders: dict[str, str] = {}

    def _stash(html_chunk: str) -> str:
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = html_chunk
        return key

    def _fence_sub(m: re.Match[str]) -> str:
        body = m.group(2)
        # убрать единственный финальный \n чтобы не было лишней пустой строки
        if body.endswith("\n"):
            body = body[:-1]
        return _stash(f"<pre>{html.escape(body)}</pre>")

    text = _FENCE_RE.sub(_fence_sub, text)
    text = _INLINE_CODE_RE.sub(
        lambda m: _stash(f"<code>{html.escape(m.group(1))}</code>"), text
    )

    # 2) Ссылки и wiki-ссылки — до экранирования html, иначе скобки потеряются.
    def _link_sub(m: re.Match[str]) -> str:
        label = html.escape(m.group(1))
        url = html.escape(m.group(2), quote=True)
        return _stash(f'<a href="{url}">{label}</a>')

    text = _LINK_RE.sub(_link_sub, text)
    text = _WIKI_RE.sub(
        lambda m: _stash(
            f"<i>{html.escape(_wiki_alias(m.group(1), m.group(2)))}</i>"
        ),
        text,
    )

    # 3) Экранировать всё остальное.
    text = html.escape(text, quote=False)

    # 4) Построчная обработка: заголовки, hr, цитаты, списки.
    out_lines: list[str] = []
    in_quote = False
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        if _HR_RE.match(line):
            if in_quote:
                out_lines.append("</blockquote>")
                in_quote = False
            out_lines.append("")
            continue

        m_h = _HEADING_RE.match(line)
        if m_h:
            if in_quote:
                out_lines.append("</blockquote>")
                in_quote = False
            out_lines.append(f"<b>{m_h.group(2)}</b>")
            continue

        if line.lstrip().startswith("&gt; ") or line.lstrip() == "&gt;":
            content = line.lstrip()[len("&gt;"):].lstrip()
            if not in_quote:
                out_lines.append("<blockquote>")
                in_quote = True
            out_lines.append(content)
            continue

        if in_quote:
            out_lines.append("</blockquote>")
            in_quote = False

        m_l = _LIST_RE.match(line)
        if m_l:
            indent, item = m_l.group(1), m_l.group(2)
            out_lines.append(f"{indent}• {item}")
            continue

        out_lines.append(line)

    if in_quote:
        out_lines.append("</blockquote>")

    text = "\n".join(out_lines)

    # 5) Inline-форматирование: bold/italic/strike. Порядок важен — сначала
    #    двойные обёртки, потом одинарные, чтобы `**bold**` не съел `*italic*`.
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_UNDERSCORE_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    text = _ITALIC_UND_RE.sub(r"<i>\1</i>", text)

    # 6) Восстановить плейсхолдеры.
    for key, value in placeholders.items():
        text = text.replace(key, value)

    # 7) Схлопнуть лишние пустые строки (>2 подряд → 2).
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")

    return text


def split_tg_chunks(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Разбить готовый HTML на куски ≤ limit символов по границам строк.

    Не делит внутри тегов: режет только по `\n`. Для очень длинных строк
    делает жёсткое разрезание по limit.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    cur = 0
    for line in text.split("\n"):
        ln = len(line) + 1
        if cur + ln > limit and buf:
            chunks.append("\n".join(buf))
            buf, cur = [], 0
        if ln > limit:
            # Слишком длинная строка — резать жёстко.
            if buf:
                chunks.append("\n".join(buf))
                buf, cur = [], 0
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue
        buf.append(line)
        cur += ln
    if buf:
        chunks.append("\n".join(buf))
    return chunks
