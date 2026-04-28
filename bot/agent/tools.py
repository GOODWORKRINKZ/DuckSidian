"""Описания tools для DeepSeek (OpenAI tool-calling JSON schema) +
их исполнение через Wiki/DB + ask_user pipeline."""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Awaitable, Callable

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
            "description": "Подстроковый поиск по всем .md в vault.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
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
]


AskUserFn = Callable[[str, list[str] | None], Awaitable[str]]


class ToolExecutor:
    """Исполняет вызовы инструментов от LLM. Сохраняет write-режим: агент
    не может писать в raw/."""

    def __init__(self, wiki: Wiki, ask_user: AskUserFn, deepseek_client: Any = None):
        self.wiki = wiki
        self.ask_user = ask_user
        self._ds = deepseek_client

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return self.wiki.read_file(arguments["path"])
            if name == "list_dir":
                return "\n".join(self.wiki.list_dir(arguments.get("path", "")))
            if name == "search_wiki":
                hits = self.wiki.search(arguments["query"])
                return json.dumps(hits, ensure_ascii=False, indent=2)
            if name == "write_file":
                path = arguments["path"]
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                rel = self.wiki.write_file(path, arguments["content"])
                return f"OK: wrote {rel}"
            if name == "append_file":
                path = arguments["path"]
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                rel = self.wiki.append_file(path, arguments["content"])
                return f"OK: appended {rel}"
            if name == "ask_user":
                opts = arguments.get("options") or None
                ans = await self.ask_user(arguments["question"], opts)
                return ans or "(no answer / timeout)"
            if name == "web_search":
                return await self._web_search(
                    arguments["query"],
                    min(int(arguments.get("max_results", 5)), 10),
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
    async def _web_search(query: str, max_results: int = 5) -> str:
        """DuckDuckGo text search, запускается в thread-pool (синх. либа)."""
        import asyncio

        def _sync() -> list[dict]:
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
