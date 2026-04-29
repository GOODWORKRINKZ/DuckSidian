"""Описания tools для DeepSeek (OpenAI tool-calling JSON schema) +
их исполнение через Wiki/DB + ask_user pipeline."""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Awaitable, Callable

import httpx

from ..wiki import Wiki, WikiPathError

log = logging.getLogger(__name__)


# JSON-schema описания, отдаём DeepSeek в каждом запросе.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать markdown-файл из vault'а.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": (
                "Прочитать диапазон строк файла (экономит контекст на больших файлах). "
                "start и end — 1-based, включительные. Если end=0 — до конца."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer", "default": 0},
                },
                "required": ["path", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_file",
            "description": (
                "Поиск по regex внутри одного файла. Возвращает список совпавших строк "
                "в формате 'NNN: текст'. context — количество строк вокруг каждого матча."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "context": {"type": "integer", "default": 2},
                    "max_matches": {"type": "integer", "default": 30},
                },
                "required": ["path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Список файлов и папок относительно корня vault'а.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": ""}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": (
                "Подстроковый поиск по всем .md в vault. Возвращает JSON-массив "
                "hit'ов c полями path, line, text + before/after — по 'context' "
                "строк контекста до и после. Используй контекст, чтобы самому "
                "отсеять false-positive: например, query='СОРИК' может "
                "совпасть со словом 'сенСОРИКа' — тогда смотри before/after и "
                "игнорируй. Регистронезависимо, без regex."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "integer", "default": 2},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Создать или перезаписать файл. Только в raw/ (нельзя!), "
                "wiki/, либо index.md/log.md. Реально raw/ запрещено для агента."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Дописать в конец файла.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Точечный str-replace в файле: заменить old_string на new_string. "
                "old_string должен встречаться РОВНО ОДИН раз. Идеально для правки frontmatter "
                "(updated/sources/tags) или патчинга строки без регенерации всего файла."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Задать вопрос команде в forum-топик и подождать ответ. "
                "Возвращает текст ответа человека."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Опц. варианты для inline-кнопок.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_media",
            "description": (
                "Проанализировать медиафайл из vault и вернуть его описание. "
                "Для изображений (jpg, png, webp) — DeepSeek VL vision. "
                "Для видео (mp4, mov, avi, mkv) — транскрипция аудио (STT) + "
                "описание 4 равномерных кейфреймов через vision. "
                "Для голосовых/аудио — STT транскрипция через Whisper. "
                "Для документов (PDF, DOCX, XLSX) — извлечение текста. "
                "path — относительный путь внутри vault, например raw/assets/..."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Завершить работу. Аргумент — итоговый отчёт/ответ.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Поиск в интернете через DuckDuckGo. Используй для проверки фактов, "
                "поиска документации, уточнения терминов. "
                "Возвращает список {title, href, body} (до max_results результатов)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "description": "Максимум результатов (1–10). По умолчанию 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Загрузить содержимое веб-страницы по URL и вернуть очищенный текст. "
                "Используй для конкретных ссылок из сообщений чата (YouTube, GitHub, "
                "статьи, документация и т.п.) — чтобы понять о чём ссылка. "
                "Возвращает заголовок страницы + основной текст (до 3000 символов)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Полный URL включая https://"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_git_log",
            "description": (
                "Получить историю коммитов GitHub-репозитория (источник правды для проектов). "
                "repo — либо 'owner/name', либо полный URL https://github.com/owner/name. "
                "since — необязательная нижняя дата в формате YYYY-MM-DD (включительно). "
                "limit — сколько последних коммитов вернуть (1–100, по умолчанию 30). "
                "Если в frontmatter wiki/projects/<X>.md есть поле repo:, бери оттуда."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "since": {"type": "string", "description": "YYYY-MM-DD"},
                    "limit": {"type": "integer", "default": 30},
                },
                "required": ["repo"],
            },
        },
    },
]


AskUserFn = Callable[[str, list[str] | None], Awaitable[str]]


class ToolExecutor:
    """Исполняет вызовы инструментов от LLM. Сохраняет write-режим: агент
    не может писать в raw/."""

    def __init__(self, wiki: Wiki, ask_user: AskUserFn, deepseek_client: Any = None,
                 file_cache: dict[str, str] | None = None):
        self.wiki = wiki
        self.ask_user = ask_user
        self._ds = deepseek_client
        # Кэш содержимого файлов в рамках run_agent (или прокинутый снаружи —
        # тогда живёт между несколькими run_agent одного дня). Ключ — rel path.
        self._fcache: dict[str, str] = file_cache if file_cache is not None else {}

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                path = arguments["path"]
                # Хардкод-защита: log.md растёт без предела, читать его при ingest бессмысленно.
                if path in ("log.md", "/log.md") or path.endswith("/log.md"):
                    return (
                        "[log.md полностью не читается: файл огромный. "
                        "Используй grep_file('log.md', '<pattern>') для поиска "
                        "или read_lines('log.md', start, end) для куска. "
                        "Для записи новой строки используй append_file('log.md', ...).]"
                    )
                if path in self._fcache:
                    log.info("read_file %s -> cache hit (%d chars)", path, len(self._fcache[path]))
                    return self._fcache[path] + "\n\n[cached: этот файл уже был прочитан/записан ранее в этом ране; не читай его снова]"
                content = self.wiki.read_file(path)
                self._fcache[path] = content
                return content
            if name == "read_lines":
                path = self.wiki.find_md(arguments["path"])
                start = max(1, int(arguments.get("start", 1)))
                end = int(arguments.get("end", 0))
                if path in self._fcache:
                    text = self._fcache[path]
                else:
                    if path in ("log.md", "/log.md") or path.endswith("/log.md"):
                        # Специальный случай: log.md по диапазону МОЖНО читать, это разумно.
                        text = self.wiki.read_file(path, max_bytes=2_000_000)
                    else:
                        text = self.wiki.read_file(path)
                        self._fcache[path] = text
                lines = text.splitlines()
                if end <= 0 or end > len(lines):
                    end = len(lines)
                if start > len(lines):
                    return f"[файл всего {len(lines)} строк, start={start} вне диапазона]"
                slc = lines[start - 1:end]
                numbered = "\n".join(f"{i:>5}: {ln}" for i, ln in enumerate(slc, start))
                return f"[{path} строки {start}–{end} из {len(lines)}]\n{numbered}"
            if name == "grep_file":
                import re as _re
                path = self.wiki.find_md(arguments["path"])
                pattern = arguments["pattern"]
                ctx = max(0, int(arguments.get("context", 2)))
                limit = max(1, int(arguments.get("max_matches", 30)))
                if path in self._fcache:
                    text = self._fcache[path]
                else:
                    if path in ("log.md", "/log.md") or path.endswith("/log.md"):
                        text = self.wiki.read_file(path, max_bytes=2_000_000)
                    else:
                        text = self.wiki.read_file(path)
                        self._fcache[path] = text
                try:
                    rx = _re.compile(pattern, _re.IGNORECASE | _re.MULTILINE)
                except _re.error as exc:
                    return f"ERROR: неверный regex: {exc}"
                lines = text.splitlines()
                hits: list[int] = [i for i, ln in enumerate(lines) if rx.search(ln)]
                if not hits:
                    return f"[нет совпадений для /{pattern}/ в {path}]"
                truncated = len(hits) > limit
                hits = hits[:limit]
                # Объединяем окна контекста
                ranges: list[tuple[int, int]] = []
                for h in hits:
                    a = max(0, h - ctx)
                    b = min(len(lines) - 1, h + ctx)
                    if ranges and a <= ranges[-1][1] + 1:
                        ranges[-1] = (ranges[-1][0], max(ranges[-1][1], b))
                    else:
                        ranges.append((a, b))
                blocks = []
                for a, b in ranges:
                    block = "\n".join(f"{i + 1:>5}: {lines[i]}" for i in range(a, b + 1))
                    blocks.append(block)
                tail = f"\n[срезано до {limit} матчей]" if truncated else ""
                return (
                    f"[grep /{pattern}/ в {path}: {len(hits)} матчей, "
                    f"файл {len(lines)} строк]\n"
                    + "\n---\n".join(blocks) + tail
                )
            if name == "list_dir":
                return "\n".join(self.wiki.list_dir(arguments.get("path", "")))
            if name == "search_wiki":
                hits = self.wiki.search(
                    arguments["query"],
                    context=int(arguments.get("context", 2)),
                )
                return json.dumps(hits, ensure_ascii=False, indent=2)
            if name == "write_file":
                path = arguments.get("path") or arguments.get("filename") or arguments.get("file_path") or arguments.get("filepath")
                if not path:
                    return "ERROR: missing 'path' argument"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                content = arguments["content"]
                rel = self.wiki.write_file(path, content)
                # Актуализируем кэш под реальным rel-path и по исходному.
                self._fcache[rel] = content
                self._fcache[path] = content
                return f"OK: wrote {rel}"
            if name == "append_file":
                path = arguments.get("path") or arguments.get("filename") or arguments.get("file_path") or arguments.get("filepath")
                if not path:
                    return "ERROR: missing 'path' argument"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                path = self.wiki.find_md(path)
                addition = arguments["content"]
                rel = self.wiki.append_file(path, addition)
                # Обновляем кэш: если был — добавим, иначе прочитаем разово с диска.
                if rel in self._fcache:
                    self._fcache[rel] = self._fcache[rel] + addition
                else:
                    try:
                        self._fcache[rel] = self.wiki.read_file(rel)
                    except Exception:
                        pass
                self._fcache[path] = self._fcache.get(rel, "")
                return f"OK: appended {rel}"
            if name == "edit_file":
                path = arguments.get("path") or arguments.get("file_path")
                old_s = arguments.get("old_string", "")
                new_s = arguments.get("new_string", "")
                if not path or not old_s:
                    return "ERROR: missing 'path' or 'old_string'"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                path = self.wiki.find_md(path)
                # Берём из кэша или с диска
                current = self._fcache.get(path)
                if current is None:
                    try:
                        current = self.wiki.read_file(path)
                    except WikiPathError as exc:
                        return f"ERROR: {exc}"
                cnt = current.count(old_s)
                if cnt == 0:
                    return (
                        "ERROR: old_string не найден в файле. "
                        "Проверь точное совпадение (пробелы/переносы) или используй read_lines."
                    )
                if cnt > 1:
                    return (
                        f"ERROR: old_string встречается {cnt} раз. "
                        "Добавь контекст (соседние строки), чтобы old_string был уникален."
                    )
                updated = current.replace(old_s, new_s, 1)
                rel = self.wiki.write_file(path, updated)
                self._fcache[rel] = updated
                self._fcache[path] = updated
                return f"OK: edited {rel} (замена {len(old_s)} → {len(new_s)} симв.)"
            if name == "ask_user":
                opts = arguments.get("options") or None
                ans = await self.ask_user(arguments["question"], opts)
                return ans or "(no answer / timeout)"
            if name == "web_search":
                return await self._web_search(
                    arguments["query"],
                    min(int(arguments.get("max_results", 5)), 10),
                )
            if name == "fetch_url":
                return await self._fetch_url(arguments["url"])
            if name == "fetch_git_log":
                return await self._fetch_git_log(
                    arguments["repo"],
                    arguments.get("since"),
                    min(int(arguments.get("limit", 30)), 100),
                )
            if name == "describe_media":
                return await self._describe_media(arguments["path"])
            return f"ERROR: unknown tool {name}"
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return f"ERROR: {exc}"

    async def _describe_media(self, rel_path: str) -> str:
        """Анализ медиафайла: vision для изображений, STT/parse для остальных."""
        from ..media_parser import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, extract_text_from_file, analyze_video

        try:
            file_path = self.wiki.resolve(rel_path)
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        if not file_path.is_file():
            return f"ERROR: файл не найден: {rel_path}"

        ext = file_path.suffix.lower()
        size_kb = file_path.stat().st_size // 1024

        # Изображения — vision через DeepSeek VL
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            if self._ds is None:
                return f"[изображение {ext}, {size_kb} KB — vision API недоступен]"
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".bmp": "image/bmp",
            }
            mime = mime_map.get(ext, "image/jpeg")
            raw = file_path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            description = await self._ds.describe_image(b64, mime)
            return f"[изображение {ext}, {size_kb} KB]\n{description}"

        # Видео — STT + keyframes
        if ext in VIDEO_EXTENSIONS:
            if self._ds is None:
                return f"[видео {ext}, {size_kb} KB — vision API недоступен]"
            result = await analyze_video(file_path, self._ds.describe_image)
            return f"[видео {ext}, {size_kb} KB]\n{result}"

        # PDF, DOCX, XLSX, текст, аудио/голосовые — через media_parser
        extracted = await extract_text_from_file(file_path)
        if extracted is not None:
            label = "голосовое/аудио (STT)" if ext in AUDIO_EXTENSIONS else f"файл {ext}"
            return f"[{label}, {size_kb} KB]\n{extracted}"

        return f"[файл {ext}, {size_kb} KB — бинарный, анализ недоступен]"

    @staticmethod
    async def _fetch_url(url: str) -> str:
        """Загрузить страницу и вернуть очищенный текст (до 3000 символов)."""
        import re
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"ERROR: неподдерживаемая схема '{parsed.scheme}' (только http/https)"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; DuckSidian/1.0)",
                "Accept-Language": "ru,en;q=0.9",
            }
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "text/html" not in ct and "text/plain" not in ct:
                    return f"[бинарный контент: {ct}]"
                html = resp.text
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_url %s failed: %s", url, exc)
            return f"ERROR fetch_url: {exc}"

        try:
            from bs4 import BeautifulSoup  # type: ignore[import]
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title else ""
            body = soup.find("article") or soup.find("main") or soup.body
            text = body.get_text(separator="\n", strip=True) if body else soup.get_text()
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            title = m.group(1).strip() if m else ""

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        max_chars = 3000
        truncated = "" if len(text) <= max_chars else f"\n...(обрезано, всего {len(text)} символов)"
        text = text[:max_chars] + truncated

        result = f"URL: {url}\nЗаголовок: {title}\n\n{text}"
        log.info("fetch_url %s → %d символов", url, len(result))
        return result

    @staticmethod
    async def _web_search(query: str, max_results: int = 5) -> str:
        """DuckDuckGo text search, запускается в thread-pool (синх. либа)."""
        import asyncio

        def _sync() -> list[dict]:
            try:
                from ddgs import DDGS  # type: ignore[import]
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import]

            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, _sync)
            return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            log.warning("web_search failed: %s", exc)
            return f"ERROR web_search: {exc}"

    @staticmethod
    async def _fetch_git_log(repo: str, since: str | None = None, limit: int = 30) -> str:
        """GitHub commits API. repo: 'owner/name' или полный URL."""
        import re
        m = re.search(r"github\.com[:/]+([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$", repo)
        if m:
            owner, name = m.group(1), m.group(2)
        elif "/" in repo and not repo.startswith("http"):
            owner, _, name = repo.strip("/").partition("/")
        else:
            return f"ERROR fetch_git_log: не парсится repo='{repo}' (ожидаю owner/name или GitHub URL)"

        params: dict[str, str | int] = {"per_page": max(1, min(limit, 100))}
        if since:
            params["since"] = f"{since}T00:00:00Z"

        url = f"https://api.github.com/repos/{owner}/{name}/commits"
        headers = {
            "User-Agent": "DuckSidian/1.0",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            timeout = httpx.Timeout(connect=15.0, read=20.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 404:
                    return f"ERROR fetch_git_log: repo {owner}/{name} не найден (404)"
                if resp.status_code == 403:
                    return f"ERROR fetch_git_log: rate limit / forbidden (403). Повтори позже."
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_git_log %s/%s failed: %s", owner, name, exc)
            return f"ERROR fetch_git_log: {exc}"

        if not isinstance(data, list) or not data:
            return f"[{owner}/{name}] коммитов не найдено (since={since or 'весь период'})"

        lines = [f"# git log {owner}/{name} (since={since or '—'}, {len(data)} коммитов)"]
        for c in data:
            commit = c.get("commit", {})
            author = (commit.get("author") or {}).get("name") or "?"
            date = (commit.get("author") or {}).get("date") or "?"
            msg = (commit.get("message") or "").strip().splitlines()[0][:200]
            sha = (c.get("sha") or "")[:7]
            lines.append(f"- {date[:10]} {sha} {author}: {msg}")
        result = "\n".join(lines)
        log.info("fetch_git_log %s/%s → %d коммитов", owner, name, len(data))
        return result[:5000]
