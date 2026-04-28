"""Описания tools для DeepSeek (OpenAI tool-calling JSON schema) +
их исполнение через Wiki/DB + ask_user pipeline."""
from __future__ import annotations

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
            "name": "finish",
            "description": "Завершить работу. Аргумент — итоговый отчёт/ответ.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


AskUserFn = Callable[[str, list[str] | None], Awaitable[str]]


class ToolExecutor:
    """Исполняет вызовы инструментов от LLM. Сохраняет write-режим: агент
    не может писать в raw/."""

    def __init__(self, wiki: Wiki, ask_user: AskUserFn):
        self.wiki = wiki
        self.ask_user = ask_user

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
            return f"ERROR: unknown tool {name}"
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return f"ERROR: {exc}"
