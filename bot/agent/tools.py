"""Описания tools для DeepSeek (OpenAI tool-calling JSON schema) +
их исполнение через Wiki/DB + ask_user pipeline."""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Awaitable, Callable

import httpx

from ..wiki import Wiki, WikiPathError
from ..media_index import MediaIndex
from .wikitools import append_to_section, big_file_view, broken_links, fuzzy_replace, similar_pages

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
                "Подстроковый поиск по всем .md в vault (raw/ и wiki/, включая raw/daily и "
                "raw/imports). НЕ regex: несколько вариантов задавай через '|' — это ИЛИ "
                "(query='винты|M3*12|1688'). Регистронезависимо. path_prefix ограничивает "
                "область: 'raw/daily', 'raw/imports', 'wiki/projects'. Возвращает JSON-массив "
                "hit'ов c полями path, line, text + before/after ('context' строк вокруг). "
                "По контексту сам отсеивай false-positive (СОРИК ⊂ сенСОРИКа)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "integer", "default": 2},
                    "path_prefix": {"type": "string", "default": ""},
                    "max_hits": {"type": "integer", "default": 30},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_dir",
            "description": (
                "Regex-поиск (как grep -r) по ВСЕМ файлам директории vault'а рекурсивно: "
                "например grep_dir('raw/daily', 'винт|M3\\*12|1688') или "
                "grep_dir('raw/imports', 'колес|轮') — ищет сразу во всех днях/файлах, не нужно "
                "перебирать файлы по одному. Возвращает JSON {files_scanned, matches:[{path, line, "
                "text, before, after}], truncated}. Регистронезависимо."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Директория, например raw/daily"},
                    "pattern": {"type": "string", "description": "Python-regex"},
                    "context": {"type": "integer", "default": 1},
                    "max_matches": {"type": "integer", "default": 40},
                },
                "required": ["path", "pattern"],
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
            "name": "append_to_section",
            "description": (
                "Дописать текст в конец секции markdown-страницы по её заголовку (например "
                "heading='Задачи' или '## Заметки'). Регистр/эмодзи/количество # не важны; секции "
                "нет — создастся. Лучший способ добавить задачу/заметку/решение в страницу "
                "проекта: не нужно читать файл целиком и подбирать old_string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "heading": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "heading", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Точечный str-replace в файле: заменить old_string на new_string. "
                "old_string должен встречаться РОВНО ОДИН раз (пробелы/переносы сверяются нестрого). "
                "Копируй old_string из свежего read_lines, а не по памяти. Для добавления в секцию — "
                "append_to_section."
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
            "name": "move_file",
            "description": (
                "Переместить (переименовать) файл внутри wiki/. "
                "Используй для переклассификации страниц, например: "
                "wiki/concepts/misc/1688.md → wiki/entities/companies/1688.md. "
                "raw/ запрещён. Старый файл удаляется, содержимое сохраняется."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "Текущий путь файла"},
                    "dst": {"type": "string", "description": "Новый путь файла"},
                },
                "required": ["src", "dst"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Удалить файл из wiki/. Используй для удаления дублирующих или "
                "устаревших страниц после слияния (merge). raw/ запрещён."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь файла для удаления"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "defer_question",
            "description": (
                "Отложить спорный вопрос в очередь разрешения конфликтов. "
                "Используй вместо ask_user когда вопрос не блокирует работу "
                "прямо сейчас: например, к какому проекту относится сообщение, "
                "стоит ли слить два похожих проекта, правильно ли определён домен. "
                "Агент продолжает работу, а пользователь отвечает позже через /resolve. "
                "НЕ жди ответа — сразу вызывай finish() после defer_question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Краткий вопрос пользователю"},
                    "context": {
                        "type": "string",
                        "description": "Контекст: что произошло, почему возник вопрос, что уже сделано",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Варианты ответа (опц.)",
                    },
                },
                "required": ["question"],
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
                "path — относительный путь внутри vault, например raw/assets/... "
                "Если файл уже тегирован или похож на известный проект — в ответе "
                "будет подсказка с именем проекта."
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
            "name": "tag_media",
            "description": (
                "Пометить медиафайл как принадлежащий конкретному проекту. "
                "Записывает в визуальный индекс: при следующих ingest describe_media "
                "будет сразу показывать подсказку о проекте. "
                "Используй после describe_media когда уверен в принадлежности. "
                "project — название проекта (например 'РОББОКС', 'BiBa'). "
                "notes — ключевые визуальные признаки (цвет, форма, компоненты), "
                "они пополняют визуальный профиль проекта для будущих матчей."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "rel_path файла, например raw/assets/.../file.jpg"},
                    "project": {"type": "string", "description": "Название проекта"},
                    "notes": {"type": "string", "description": "Визуальные признаки (необяз.)"},
                },
                "required": ["path", "project"],
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
            "name": "baidu_search",
            "description": (
                "Поиск в Baidu (百度) — для запросов НА КИТАЙСКОМ: китайские товары и производители "
                "(1688/Taobao/JD), компании, роботы, платы, статьи на 微信/知乎/bilibili/CSDN. "
                "Формулируй query по-китайски (иероглифами), Baidu по-русски/по-английски ищет плохо. "
                "Возвращает список {title, href, body, source, date} (до max_results). "
                "query — до ~36 иероглифов (длиннее обрежется). sites — искать только на этих "
                "сайтах (например ['zhihu.com','csdn.net']), recency — свежесть (week/month/semiyear/year). "
                "Если Baidu недоступен — результаты придут из DuckDuckGo (регион Китай), "
                "об этом будет пометка в первой строке."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Запрос на китайском, например «大陆智源 ANDI 机器人»"},
                    "max_results": {
                        "type": "integer",
                        "description": "Максимум результатов (1–10). По умолчанию 5.",
                        "default": 5,
                    },
                    "sites": {"type": "array", "items": {"type": "string"},
                              "description": "Только эти сайты (опц.)"},
                    "recency": {"type": "string", "enum": ["week", "month", "semiyear", "year"]},
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
                "Используй для конкретных ссылок из сообщений чата (YouTube, GitHub, 1688, "
                "статьи, документация и т.п.) — чтобы понять о чём ссылка. "
                "Возвращает заголовок страницы + основной текст (до 3000 символов). "
                "YouTube: название, канал и ПОЛНАЯ стенограмма с таймкодами (субтитры или ASR), "
                "сохранённая в raw/imports/youtube_<id>.md — используй её как источник для "
                "wiki/sources/<slug>.md. 1688: карточка товара."
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
            "name": "summarize_file",
            "description": (
                "Сжать БОЛЬШОЙ файл (транскрипт видео, экспорт чата, длинный документ) в "
                "структурированный конспект одним вызовом вместо чтения по кускам: тезисы, "
                "факты/цифры/имена, таймкоды или даты, выводы. focus — на что обратить внимание "
                "(например «бизнес-кейсы и ошибки», «решения по моторам»). До ~120k символов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "focus": {"type": "string", "default": ""},
                    "max_words": {"type": "integer", "default": 600},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_file",
            "description": (
                "Скачать ФАЙЛ по прямой ссылке (zip/pdf/docx/xlsx/txt/json/html…) в vault "
                "raw/assets/<date>/ и сразу разобрать (архив → распаковка/импорт чата, документ → "
                "текст). Нужен, когда пользователь прислал ссылку на файл вместо вложения "
                "(Telegram не отдаёт ботам файлы > 20 МБ). Ссылка должна быть прямой (не страница "
                "облака); до 512 МБ. Возвращает путь и сводку содержимого."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_image",
            "description": (
                "Скачать картинку по URL (например фото товара с 1688/Taobao из fetch_url) в vault "
                "raw/assets/<date>/web_*.jpg и вернуть rel-путь. Дальше: describe_media(path), "
                "чтобы посмотреть её, и/или блок IMAGES в finish(), чтобы отправить пользователю. "
                "Лимит 10 МБ, только image/*."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Вернуть текущую дату и время (часовой пояс Europe/Moscow). "
                "Используй когда нужно знать 'сегодня', 'вчера', 'эта неделя' и т.д."
            ),
            "parameters": {"type": "object", "properties": {}},
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


# Инструменты, доступные ТОЛЬКО в личном режиме (DM): доступ к командным wiki
# (read-only) и шаринг артефактов в групповой чат.
PERSONAL_EXTRA_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_team_wiki",
            "description": (
                "Подстроковый поиск по wiki ГРУППОВОГО чата (командные знания, read-only). "
                "chat — слаг чата из списка в контексте (например 'main'). "
                "Возвращает JSON hit'ов {path, line, text, before, after}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat": {"type": "string"},
                    "query": {"type": "string"},
                    "context": {"type": "integer", "default": 2},
                },
                "required": ["chat", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_team_file",
            "description": (
                "Прочитать файл из vault'а ГРУППОВОГО чата (read-only). "
                "chat — слаг чата, path — относительный путь (например 'index.md', "
                "'wiki/projects/X.md'). Для больших файлов используй start/end (1-based строки)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat": {"type": "string"},
                    "path": {"type": "string"},
                    "start": {"type": "integer", "default": 0},
                    "end": {"type": "integer", "default": 0},
                },
                "required": ["chat", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_team_dir",
            "description": "Список файлов в директории vault'а ГРУППОВОГО чата (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat": {"type": "string"},
                    "path": {"type": "string", "default": ""},
                },
                "required": ["chat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_to_chat",
            "description": (
                "Поделиться артефактом из личного vault с групповым чатом: файл "
                "копируется в raw/notes/ (или raw/assets/shared/) vault'а чата, "
                "регистрируется в БД для ingest и анонсируется в Telegram-топике бота. "
                "ТОЛЬКО по явной просьбе владельца. path — rel путь в личном vault "
                "(wiki/... или raw/assets/...), chat — слаг чата, comment — "
                "необязательная подпись."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "chat": {"type": "string"},
                    "comment": {"type": "string", "default": ""},
                },
                "required": ["path", "chat"],
            },
        },
    },
]

PERSONAL_EXTRA_TOOLS.append({
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Сгенерировать картинку по текстовому описанию (MiniMax image-01). "
            "Используй ТОЛЬКО когда пользователь явно просит нарисовать/сгенерировать/"
            "визуализировать. prompt — подробное описание на английском или русском "
            "(до 1500 символов), aspect_ratio — 1:1 | 16:9 | 4:3 | 3:2 | 2:3 | 3:4 | 9:16 | 21:9. "
            "Возвращает rel-пути сохранённых файлов (raw/assets/<date>/gen_*.jpg) — "
            "перечисли их в блоке IMAGES в finish(), чтобы отправить пользователю."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "aspect_ratio": {"type": "string", "default": "1:1"},
                "n": {"type": "integer", "default": 1, "description": "1–4"},
            },
            "required": ["prompt"],
        },
    },
})

PERSONAL_EXTRA_TOOLS += [
    {
        "type": "function",
        "function": {
            "name": "create_workspace",
            "description": (
                "Создать воркспейс = новую тему (вкладку) в личном чате с ботом + страницу проекта "
                "wiki/projects/<Имя>.md. Используй, когда пользователь просит «создай вкладку/тему/"
                "воркспейс <имя>» или «заведи проект <имя> отдельной вкладкой». "
                "Возвращает OK с именем темы или ERROR (например, если темы в личке выключены)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Имя воркспейса (до 128 символов)"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workspaces",
            "description": "Список воркспейсов (тем) пользователя с привязанными проектами.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_workspace",
            "description": (
                "Поделиться ТЕКУЩИМ воркспейсом (проект + связанные страницы + файлы). target — "
                "слаг группового чата (например 'main') ИЛИ пользователь: '@username' / числовой id. "
                "В чат — выгрузка в vault группы; человеку — снимок приходит ему в личку с кнопками "
                "«посмотреть / принять как воркспейс». Только по явной просьбе владельца."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "comment": {"type": "string", "default": ""},
                    "mode": {
                        "type": "string",
                        "description": "Для человека: copy — снимок (копия), coop — совместный воркспейс (одна вкладка на двоих, общий vault). По умолчанию спроси пользователя или coop, если он сказал «вместе/совместно».",
                        "default": "copy",
                    },
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_workspace_icon",
            "description": (
                "Сменить значок (иконку) ТЕКУЩЕЙ темы-воркспейса. Без emoji — подобрать по смыслу "
                "названия темы автоматически; с emoji — поставить указанный (из набора значков тем "
                "Telegram; если такого нет, подбирается ближайший). Работает только внутри темы."
            ),
            "parameters": {
                "type": "object",
                "properties": {"emoji": {"type": "string", "description": "Желаемый эмодзи (опц.)"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_workspace",
            "description": (
                "Переименовать ТЕКУЩИЙ воркспейс (тему, в которой идёт разговор) и перепривязать "
                "его к проекту wiki/projects/<Новое имя>.md. Работает только внутри темы."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]

PERSONAL_EXTRA_TOOLS.append({
    "type": "function",
    "function": {
        "name": "extract_archive",
        "description": (
            "Распаковать zip-архив из vault (raw/assets/...) в raw/imports/<имя>/ и разобрать: "
            "экспорт чата WhatsApp/Telegram раскладывается по raw/daily/<дата>.md, документы "
            "внутри — в текст. Возвращает сводку: что внутри, какие файлы/даты появились."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
})

_PERSONAL_EXCLUDE = {"ask_user", "defer_question", "tag_media"}

# Инструменты, которые реально что-то меняют в vault (или в темах/чатах). Если агент
# ответил «записал/добавил/готово», а ни один из них не вызывался — запись выдумана.
PERSONAL_WRITE_TOOLS = frozenset({
    "write_file", "append_file", "append_to_section", "edit_file", "move_file", "delete_file",
    "fetch_file", "fetch_image", "extract_archive", "generate_image",
    "share_to_chat", "create_workspace", "share_workspace",
    "set_workspace_icon", "rename_workspace",
})


PERSONAL_EXTRA_TOOLS += [
    {
        "type": "function",
        "function": {
            "name": "ask_choice",
            "description": (
                "Спросить пользователя и ДОЖДАТЬСЯ ответа: вопрос + варианты кнопками в его "
                "чате (он может и написать свой ответ). Для уточнения, когда от ответа зависит, "
                "что и куда записать, или для подтверждения слияния/удаления. Варианты — "
                "конкретные (цифры, имена, формулировки из материала), 2–5 шт. Не больше 4 "
                "вопросов за ход; если можно решить самому — не спрашивай."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": (
                "Загрузить инструкцию скила из каталога СКИЛЫ (в контексте) и действовать по "
                "ней. Вызывай, когда запрос подходит под описание/триггеры скила или "
                "пользователь назвал его."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]


def personal_tool_schemas() -> list[dict]:
    """Набор инструментов для личного режима."""
    from ..config import settings  # noqa: PLC0415
    from .one688 import SCHEMA as ONE688_SCHEMA  # noqa: PLC0415
    base = [t for t in TOOL_SCHEMAS if t["function"]["name"] not in _PERSONAL_EXCLUDE]
    extra = [ONE688_SCHEMA] if settings.one688_cli.strip() else []
    return base + PERSONAL_EXTRA_TOOLS + extra


AskUserFn = Callable[[str, list[str] | None], Awaitable[str]]
ShareFn = Callable[[str, str, str], Awaitable[str]]
WorkspaceFn = Callable[[str, dict], Awaitable[str]]


class ToolExecutor:
    """Исполняет вызовы инструментов от LLM. Сохраняет write-режим: агент
    не может писать в raw/."""

    def __init__(self, wiki: Wiki, ask_user: AskUserFn, deepseek_client: Any = None,
                 file_cache: dict[str, str] | None = None, db: Any = None,
                 team_wikis: dict[str, Wiki] | None = None,
                 share_fn: ShareFn | None = None,
                 workspace_fn: WorkspaceFn | None = None,
                 choice_fn: AskUserFn | None = None):
        self.wiki = wiki
        self.ask_user = ask_user
        self._ds = deepseek_client
        self._db = db
        # Личный режим: read-only доступ к wiki групповых чатов + коллбек шаринга.
        self._team_wikis: dict[str, Wiki] = team_wikis or {}
        self._share_fn = share_fn
        self._workspace_fn = workspace_fn
        # Личный режим: вопрос с кнопками в чат пользователя (ask_choice).
        self._choice_fn = choice_fn
        self._media_index = MediaIndex(wiki.root)
        # Кэш содержимого файлов в рамках run_agent (или прокинутый снаружи —
        # тогда живёт между несколькими run_agent одного дня). Ключ — rel path.
        self._fcache: dict[str, str] = file_cache if file_cache is not None else {}

    # Сколько символов read_file отдаёт целиком; больше — начало + оглавление.
    # Согласовано с tool_result_cap в run_agent (6000): иначе хвост молча отрезается.
    READ_FULL_MAX = 5_800
    # Страница проекта крупнее этого — сигнал агенту разгрузить её.
    PAGE_WARN_CHARS = 30_000

    def _lint_after_write(self, rel: str, written: str, *, new_page: bool = False) -> str:
        """Предупреждения к OK записи: битые wikilinks в записанном тексте, похожие
        страницы (для новой), раздутая страница проекта. Пусто — всё чисто."""
        if not rel.endswith(".md"):
            return ""
        notes: list[str] = []
        try:
            bad = broken_links(written, self.wiki.root)
        except Exception as exc:  # noqa: BLE001
            log.debug("lint broken_links failed: %s", exc)
            bad = []
        if bad:
            notes.append(
                "⚠️ битые ссылки (такого файла в этом vault нет): "
                + ", ".join(f"[[{b}]]" for b in bad)
                + ". Исправь на существующую страницу или убери [[ ]] (ссылки в другие "
                "воркспейсы/чаты не работают — пиши текстом «воркспейс X» / «[chat:Y]»)."
            )
        if new_page and rel.startswith("wiki/"):
            sim = similar_pages(self.wiki.root, rel)
            if sim:
                notes.append(
                    "⚠️ похожие страницы уже есть: " + ", ".join(f"`{p}`" for p in sim)
                    + ". Если тема та же — перенеси новое туда (append_to_section) и удали "
                    "эту страницу (delete_file), чтобы не плодить дубли."
                )
        try:
            size = (self.wiki.root / rel).stat().st_size
        except OSError:
            size = 0
        if size > self.PAGE_WARN_CHARS and rel.startswith("wiki/projects/"):
            notes.append(
                f"⚠️ страница проекта {size // 1024} КБ — слишком большая. Историю держи в "
                "log.md, разборы — в wiki/research/, в проекте только Цель/Статус/актуальные "
                "Задачи/Решения со ссылками."
            )
        return ("\n" + "\n".join(notes)) if notes else ""

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "get_current_time":
                import datetime as _dt
                import zoneinfo as _zi
                tz = _zi.ZoneInfo("Europe/Moscow")
                now = _dt.datetime.now(tz)
                yesterday = (now - _dt.timedelta(days=1)).date()
                return (
                    f"Сейчас: {now.strftime('%Y-%m-%d %H:%M')} MSK\n"
                    f"Сегодня: {now.date()}\n"
                    f"Вчера: {yesterday}"
                )
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
                content = self.wiki.read_file(path)
                self._fcache[path] = content
                # Большой файл в контекст целиком не влезет (результат режется) — отдаём
                # начало + оглавление с номерами строк, дальше read_lines по секциям.
                if len(content) > self.READ_FULL_MAX:
                    return big_file_view(path, content, self.READ_FULL_MAX - 1500)
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
                    path_prefix=str(arguments.get("path_prefix") or ""),
                    max_hits=max(1, min(int(arguments.get("max_hits") or 30), 100)),
                )
                if not hits:
                    return ("[] — ничего не найдено. Подсказка: search_wiki ищет подстроку "
                            "(не regex); варианты — через '|'. Для regex используй grep_dir.")
                return json.dumps(hits, ensure_ascii=False, indent=2)
            if name == "grep_dir":
                import re as _re
                try:
                    res = self.wiki.grep_dir(
                        str(arguments.get("path") or ""), arguments["pattern"],
                        context=int(arguments.get("context", 1)),
                        max_matches=max(1, min(int(arguments.get("max_matches") or 40), 200)),
                    )
                except _re.error as exc:
                    return f"ERROR: неверный regex: {exc}"
                if not res["matches"]:
                    return f"[нет совпадений для /{arguments['pattern']}/ в {arguments.get('path')}: просмотрено файлов {res['files_scanned']}]"
                return json.dumps(res, ensure_ascii=False, indent=1)
            if name == "write_file":
                path = arguments.get("path") or arguments.get("filename") or arguments.get("file_path") or arguments.get("filepath")
                if not path:
                    return "ERROR: missing 'path' argument"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                content = arguments["content"]
                is_new = not (self.wiki.root / path).exists()
                rel = self.wiki.write_file(path, content)
                # Актуализируем кэш под реальным rel-path и по исходному.
                self._fcache[rel] = content
                self._fcache[path] = content
                return f"OK: wrote {rel}" + self._lint_after_write(rel, content, new_page=is_new)
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
                return f"OK: appended {rel}" + self._lint_after_write(rel, addition)
            if name == "append_to_section":
                path = arguments.get("path") or arguments.get("file_path") or ""
                heading = str(arguments.get("heading") or "").strip()
                addition = str(arguments.get("content") or "")
                if not path or not heading or not addition.strip():
                    return "ERROR: missing 'path', 'heading' or 'content'"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                path = self.wiki.find_md(path)
                try:
                    current = self.wiki.read_file(path, max_bytes=5_000_000)
                except WikiPathError as exc:
                    return f"ERROR: {exc}"
                updated, info = append_to_section(current, heading, addition)
                rel = self.wiki.write_file(path, updated)
                self._fcache[rel] = updated
                self._fcache[path] = updated
                return f"OK: {rel}: {info}" + self._lint_after_write(rel, addition)
            if name == "edit_file":
                path = arguments.get("path") or arguments.get("file_path")
                old_s = arguments.get("old_string", "")
                new_s = arguments.get("new_string", "")
                if not path or not old_s:
                    return "ERROR: missing 'path' or 'old_string'"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                path = self.wiki.find_md(path)
                # Всегда с диска: кэш мог устареть (другой прогон в общем воркспейсе) или
                # быть обрезан лимитом read_file — запись из него потеряла бы хвост файла.
                try:
                    current = self.wiki.read_file(path, max_bytes=5_000_000)
                except WikiPathError as exc:
                    return f"ERROR: {exc}"
                updated, info = fuzzy_replace(current, old_s, new_s)
                if updated is None:
                    return f"ERROR: {info}"
                rel = self.wiki.write_file(path, updated)
                self._fcache[rel] = updated
                self._fcache[path] = updated
                how = "" if info == "exact" else " (совпадение без учёта пробелов)"
                return (f"OK: edited {rel} (замена {len(old_s)} → {len(new_s)} симв.){how}"
                        + self._lint_after_write(rel, new_s))
            if name == "delete_file":
                path = arguments.get("path", "")
                if not path:
                    return "ERROR: missing 'path'"
                if path.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                try:
                    found = self.wiki.find_md(path)
                except WikiPathError as exc:
                    return f"ERROR: {exc}"
                abs_path = self.wiki.root / found
                if not abs_path.exists():
                    return f"ERROR: файл не найден: {found}"
                abs_path.unlink()
                self._fcache.pop(found, None)
                self._fcache.pop(path, None)
                log.info("delete_file: %s", found)
                return f"OK: deleted {found}"
            if name == "move_file":
                src = arguments.get("src", "")
                dst = arguments.get("dst", "")
                if not src or not dst:
                    return "ERROR: missing 'src' or 'dst'"
                if src.startswith("raw/") or dst.startswith("raw/"):
                    return "ERROR: raw/ is read-only for the agent"
                src_full = self.wiki.find_md(src)
                content = self.wiki.read_file(src_full)
                rel = self.wiki.write_file(dst, content)
                src_abs = self.wiki.root / src_full
                if src_abs.exists():
                    src_abs.unlink()
                    # Invalidate cache for old path
                    self._fcache.pop(src_full, None)
                    self._fcache.pop(src, None)
                self._fcache[rel] = content
                return f"OK: moved {src_full} → {rel}"
            if name == "defer_question":
                question = arguments.get("question", "")
                context = arguments.get("context") or None
                options = arguments.get("options") or None
                if not question:
                    return "ERROR: missing 'question'"
                if self._db is not None:
                    dispute_id = await self._db.add_dispute(question, context, options)
                    log.info("defer_question: dispute_id=%s q=%r", dispute_id, question)
                    return f"OK: спорный вопрос #{dispute_id} отложен. Пользователь ответит через /resolve."
                else:
                    log.warning("defer_question: no db attached, falling back to ask_user")
                    return await self.ask_user(question, options)
            if name == "ask_choice":
                if self._choice_fn is None:
                    return "ERROR: ask_choice недоступен — задай вопрос в ответе (finish)."
                question = str(arguments.get("question") or "").strip()
                opts = [str(o).strip() for o in (arguments.get("options") or []) if str(o).strip()]
                if not question:
                    return "ERROR: missing 'question'"
                ans = await self._choice_fn(question, opts[:8] or None)
                return f"Ответ пользователя: {ans}" if ans else "(нет ответа)"
            if name == "search_1688":
                from ..config import settings  # noqa: PLC0415
                from . import one688  # noqa: PLC0415
                if not settings.one688_cli.strip():
                    return "ERROR: поиск по 1688 не настроен (ONE688_CLI) — используй baidu_search/web_search."
                return await one688.run(settings.one688_cli, dict(arguments or {}), self.wiki.root,
                                        timeout=settings.one688_timeout, profile=settings.one688_profile)
            if name == "use_skill":
                from ..skills import load_skills  # noqa: PLC0415
                skills = load_skills(self.wiki.root)
                sk = skills.get(str(arguments.get("name") or "").strip().lower())
                if sk is None:
                    return "ERROR: нет такого скила. Есть: " + (", ".join(skills) or "—")
                return f"[СКИЛ {sk.name} — следуй этой инструкции]\n{sk.body}"
            if name == "ask_user":
                opts = arguments.get("options") or None
                ans = await self.ask_user(arguments["question"], opts)
                return ans or "(no answer / timeout)"
            if name == "web_search":
                return await self._web_search(
                    arguments["query"],
                    min(int(arguments.get("max_results", 5)), 10),
                )
            if name == "baidu_search":
                return await self._baidu_search(
                    arguments["query"],
                    min(int(arguments.get("max_results", 5)), 10),
                    sites=arguments.get("sites") or None,
                    recency=arguments.get("recency") or None,
                )
            if name == "fetch_url":
                return await self._fetch_url(arguments["url"])
            if name == "fetch_image":
                return await self._fetch_image(str(arguments.get("url", "")))
            if name == "summarize_file":
                if self._ds is None:
                    return "ERROR: LLM недоступен для summarize_file"
                path = self.wiki.find_md(str(arguments.get("path", "")))
                try:
                    text = self.wiki.read_file(path, max_bytes=600_000)
                except WikiPathError as exc:
                    return f"ERROR: {exc}"
                cap = 400_000
                note = ""
                if len(text) > cap:
                    note = f"\n\n[файл {len(text)} символов — сжаты первые {cap}]"
                    text = text[:cap]
                focus = str(arguments.get("focus") or "").strip()
                max_words = max(150, min(int(arguments.get("max_words") or 600), 3000))
                sys_prompt = (
                    "Ты делаешь конспект документа для базы знаний. Верни структурированный markdown: "
                    "1) о чём документ (2–3 предложения); 2) ключевые тезисы/факты/цифры/имена — "
                    "буллетами, КАЖДЫЙ с таймкодом [m:ss] или датой/строкой, если они есть в тексте; "
                    "3) выводы/что полезно. Без воды, по-русски, "
                    f"до {max_words} слов." + (f" Особое внимание: {focus}." if focus else "")
                )
                try:
                    resp = await self._ds.chat(
                        messages=[{"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": f"Файл {path}:\n\n{text}"}],
                        tools=None, temperature=0.2,
                    )
                    out = (resp["choices"][0]["message"].get("content") or "").strip()
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR summarize_file: {exc}"
                import re as _re
                out = _re.sub(r"<think>.*?</think>", "", out, flags=_re.S).strip()
                return f"[конспект {path}]\n{out}{note}"
            if name == "fetch_file":
                from ..intake import download_url_to_vault, process_attachment  # noqa: PLC0415
                url = str(arguments.get("url", "")).strip()
                if not url.startswith(("http://", "https://")):
                    return "ERROR: нужен http(s) URL"
                try:
                    rel = await download_url_to_vault(self.wiki, url)
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR fetch_file: {exc}"
                res = await process_attachment(self.wiki, rel)
                return f"OK: скачано в `{rel}`\n" + res.agent_block()
            if name == "fetch_git_log":
                return await self._fetch_git_log(
                    arguments["repo"],
                    arguments.get("since"),
                    min(int(arguments.get("limit", 30)), 100),
                )
            if name == "describe_media":
                rel = arguments["path"]
                # Проверить визуальный индекс.
                known = self._media_index.get_file(rel)
                desc = await self._describe_media(rel)
                if known:
                    hint = (
                        f"\n\n⚡ ВИЗУАЛЬНЫЙ ИНДЕКС: этот файл уже тегирован как "
                        f"**{known['project']}**."
                        + (f" Заметки: {known['notes']}" if known.get("notes") else "")
                        + " Используй эту информацию при атрибуции артефакта."
                    )
                    return desc + hint
                # Попробовать идентифицировать по профилям.
                hits = self._media_index.identify(desc)
                if hits:
                    top = hits[:2]
                    matches = "; ".join(
                        f"{h.project} (score={h.score}, слова: {', '.join(h.evidence[:5])})"
                        for h in top
                    )
                    hint = (
                        f"\n\n💡 ВИЗУАЛЬНОЕ СОВПАДЕНИЕ: описание похоже на проект(ы): {matches}. "
                        "Проверь по контексту сообщения. Если верно — вызови tag_media."
                    )
                    return desc + hint
                return desc
            if name == "tag_media":
                rel = arguments.get("path", "")
                project = arguments.get("project", "").strip()
                notes = (arguments.get("notes") or "").strip()
                if not rel or not project:
                    return "ERROR: нужны 'path' и 'project'"
                # Получить описание из кэша media (если есть) или не гнать снова.
                media_cache = self.wiki.root / ".cache" / "media" / (rel.replace("/", "__") + ".txt")
                cached_desc = ""
                if media_cache.exists():
                    try:
                        cached_desc = media_cache.read_text(encoding="utf-8")[:400]
                    except OSError:
                        pass
                self._media_index.tag(rel, project, desc=cached_desc, notes=notes or None)
                return f"OK: {rel} → {project}" + (f" ({notes})" if notes else "")
            if name in ("search_team_wiki", "read_team_file", "list_team_dir"):
                chat = str(arguments.get("chat", "")).strip()
                twiki = self._team_wikis.get(chat)
                if twiki is None:
                    avail = ", ".join(sorted(self._team_wikis)) or "(нет)"
                    return f"ERROR: неизвестный чат '{chat}'. Доступны: {avail}"
                if name == "search_team_wiki":
                    hits = twiki.search(
                        arguments["query"], context=int(arguments.get("context", 2)),
                    )
                    for h in hits:
                        h["chat"] = chat
                    return json.dumps(hits, ensure_ascii=False, indent=2)
                if name == "list_team_dir":
                    return "\n".join(twiki.list_dir(arguments.get("path", "")))
                path = twiki.find_md(arguments["path"])
                if path.endswith("log.md"):
                    text = twiki.read_file(path, max_bytes=2_000_000)
                else:
                    text = twiki.read_file(path)
                start = int(arguments.get("start") or 0)
                end = int(arguments.get("end") or 0)
                if start > 0:
                    lines = text.splitlines()
                    if end <= 0 or end > len(lines):
                        end = len(lines)
                    slc = lines[start - 1:end]
                    text = "\n".join(f"{i:>5}: {ln}" for i, ln in enumerate(slc, start))
                    return f"[{chat}:{path} строки {start}–{end} из {len(lines)}]\n{text}"
                return f"[{chat}:{path}]\n{text}"
            if name == "generate_image":
                from ..media_ai import MediaAIError, get_media_ai  # noqa: PLC0415
                from datetime import datetime as _dt, timezone as _tz  # noqa: PLC0415
                ai = get_media_ai()
                if not ai.image_enabled:
                    return "ERROR: генерация картинок недоступна (нужен MINIMAX_API_KEY / IMAGE_PROVIDER)"
                prompt = str(arguments.get("prompt", "")).strip()
                if not prompt:
                    return "ERROR: пустой prompt"
                try:
                    images = await ai.generate_images(
                        prompt,
                        aspect_ratio=str(arguments.get("aspect_ratio") or "1:1"),
                        n=int(arguments.get("n") or 1),
                    )
                except MediaAIError as exc:
                    return f"ERROR: {exc}"
                now = _dt.now(_tz.utc)
                date_iso = now.strftime("%Y-%m-%d")
                stamp = now.strftime("%H%M%S")
                paths: list[str] = []
                for i, data in enumerate(images, 1):
                    rel = self.wiki.save_asset(data, f"gen_{stamp}_{i}.jpg", date_iso)
                    paths.append(rel.replace("\\", "/"))
                return "OK: сгенерировано " + str(len(paths)) + " файл(ов):\n" + "\n".join(paths) + \
                    "\nДобавь нужные в блок IMAGES: в finish()."
            if name == "extract_archive":
                from ..intake import process_attachment  # noqa: PLC0415
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    return "ERROR: missing 'path'"
                res = await process_attachment(self.wiki, rel)
                return res.agent_block()
            if name in ("create_workspace", "list_workspaces", "rename_workspace", "set_workspace_icon",
                        "share_workspace"):
                if self._workspace_fn is None:
                    return "ERROR: управление воркспейсами недоступно в этом режиме"
                return await self._workspace_fn(name, dict(arguments or {}))
            if name == "share_to_chat":
                if self._share_fn is None:
                    return "ERROR: шаринг недоступен в этом режиме"
                path = str(arguments.get("path", "")).strip()
                chat = str(arguments.get("chat", "")).strip()
                comment = str(arguments.get("comment") or "").strip()
                if not path or not chat:
                    return "ERROR: нужны 'path' и 'chat'"
                return await self._share_fn(path, chat, comment)
            return f"ERROR: unknown tool {name}"
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return f"ERROR: {exc}"

    async def _describe_media(self, rel_path: str) -> str:
        """Анализ медиафайла: vision для изображений, STT/parse для остальных.

        Результат кэшируется в .cache/media/<rel_path_escaped>.txt — повторный
        вызов с тем же файлом возвращает кэш мгновенно без запуска STT/vision.
        """
        from ..media_parser import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, extract_text_from_file, analyze_video

        try:
            file_path = self.wiki.resolve(rel_path)
        except WikiPathError as exc:
            return f"ERROR: {exc}"
        if not file_path.is_file():
            return f"ERROR: файл не найден: {rel_path}"

        # Проверить диск-кэш прежде чем запускать STT/vision.
        cache_path = self.wiki.root / ".cache" / "media" / (rel_path.replace("/", "__") + ".txt")
        if cache_path.exists():
            try:
                cached = cache_path.read_text(encoding="utf-8")
                log.info("describe_media %s → cache hit (%d chars)", rel_path, len(cached))
                return cached
            except OSError:
                pass

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
            result = f"[изображение {ext}, {size_kb} KB]\n{description}"

        elif ext in VIDEO_EXTENSIONS:
            # Видео — STT + keyframes
            if self._ds is None:
                return f"[видео {ext}, {size_kb} KB — vision API недоступен]"
            analysis = await analyze_video(file_path, self._ds.describe_image)
            result = f"[видео {ext}, {size_kb} KB]\n{analysis}"

        else:
            # PDF, DOCX, XLSX, текст, аудио/голосовые — через media_parser
            extracted = await extract_text_from_file(file_path)
            if extracted is not None:
                label = "голосовое/аудио (STT)" if ext in AUDIO_EXTENSIONS else f"файл {ext}"
                result = f"[{label}, {size_kb} KB]\n{extracted}"
            else:
                return f"[файл {ext}, {size_kb} KB — бинарный, анализ недоступен]"

        # Сохранить в кэш.
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(result, encoding="utf-8")
        except OSError as exc:
            log.warning("describe_media: не удалось записать кэш %s: %s", cache_path, exc)

        return result

    @staticmethod
    @staticmethod
    async def _fetch_bilibili(bvid: str) -> str:
        """Fetch Bilibili video metadata via public API."""
        import datetime

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer": "https://www.bilibili.com/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}",
                    headers=headers,
                )
                resp.raise_for_status()
                body = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("bilibili api %s failed: %s", bvid, exc)
            return f"ERROR fetch_url bilibili api: {exc}"

        if body.get("code") != 0:
            return f"Bilibili API error: {body.get('message')}"

        d = body["data"]
        title = d.get("title", "")
        desc = (d.get("desc") or "").strip()
        if desc == "-":
            desc = ""
        owner = d.get("owner", {}).get("name", "")
        pubdate = d.get("pubdate")
        date_str = (
            datetime.datetime.fromtimestamp(pubdate).strftime("%Y-%m-%d")
            if pubdate
            else ""
        )
        duration = d.get("duration", 0)
        dur_str = f"{duration // 60}:{duration % 60:02d}" if duration else ""
        url = f"https://www.bilibili.com/video/{bvid}"

        lines = [f"URL: {url}", f"Заголовок: {title}"]
        if owner:
            lines.append(f"Автор: {owner}")
        if date_str:
            lines.append(f"Дата: {date_str}")
        if dur_str:
            lines.append(f"Длительность: {dur_str}")
        if desc:
            lines.append(f"\nОписание: {desc[:500]}")

        result = "\n".join(lines)
        log.info("bilibili api %s → %s", bvid, title)
        return result

    @staticmethod
    def _yt_video_id(url: str) -> str | None:
        """Извлечь video_id из ссылки YouTube."""
        import re
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        # youtu.be/VIDEO_ID
        if parsed.netloc in ("youtu.be",):
            vid = parsed.path.lstrip("/").split("/")[0].split("?")[0]
            return vid or None
        # youtube.com/watch?v=VIDEO_ID  или /shorts/VIDEO_ID  или /embed/VIDEO_ID
        if "youtube.com" in parsed.netloc:
            m = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", url)
            if m:
                return m.group(1)
            m = re.search(r"/(?:shorts|embed|v)/([a-zA-Z0-9_-]{11})", parsed.path)
            if m:
                return m.group(1)
        return None

    async def _fetch_youtube(self, url: str, video_id: str) -> str:
        """Метаданные + ПОЛНАЯ стенограмма YouTube-видео.

        Субтитры (youtube-transcript-api) с таймкодами; если их нет — скачиваем
        аудио yt-dlp и распознаём (MiniMax ASR / Whisper) кусками по 8 минут.
        Полный текст сохраняется в raw/imports/youtube_<id>.md (источник для
        wiki/sources), агенту возвращается выдержка + путь.
        """
        import asyncio

        yt_url = f"https://www.youtube.com/watch?v={video_id}"

        def _fmt_ts(sec: float) -> str:
            sec = int(sec)
            return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:d}:{sec % 60:02d}"

        def _get_transcript() -> tuple[str, str]:
            """Синхронно: стенограмма с таймкодами (строка на ~45 сек) через youtube-transcript-api."""
            try:
                from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound  # type: ignore[import]
                api = YouTubeTranscriptApi()
                try:
                    transcript = api.fetch(video_id, languages=["ru", "en"])
                except NoTranscriptFound:
                    transcript_list = api.list(video_id)
                    transcript = next(iter(transcript_list)).fetch()
                lines: list[str] = []
                cur_start = None
                cur: list[str] = []
                for s in transcript:
                    st = float(getattr(s, "start", 0.0) or 0.0)
                    if cur_start is None:
                        cur_start = st
                    cur.append(str(getattr(s, "text", "")).replace("\n", " ").strip())
                    if st - cur_start >= 45:
                        lines.append(f"[{_fmt_ts(cur_start)}] " + " ".join(x for x in cur if x))
                        cur, cur_start = [], None
                if cur:
                    lines.append(f"[{_fmt_ts(cur_start or 0)}] " + " ".join(x for x in cur if x))
                lang = getattr(transcript, "language_code", "?")
                return "\n".join(lines), lang
            except Exception as exc:  # noqa: BLE001
                return "", str(exc)

        def _get_title() -> str:
            """Синхронно: получить заголовок через oEmbed (без API-ключа)."""
            try:
                import urllib.request
                import json as _json
                oembed_url = f"https://www.youtube.com/oembed?url={yt_url}&format=json"
                with urllib.request.urlopen(oembed_url, timeout=8) as r:  # noqa: S310
                    data = _json.loads(r.read())
                return data.get("title", ""), data.get("author_name", "")
            except Exception:  # noqa: BLE001
                return "", ""

        loop = asyncio.get_running_loop()
        (title, author), (transcript_text, transcript_err) = await asyncio.gather(
            loop.run_in_executor(None, _get_title),
            loop.run_in_executor(None, _get_transcript),
        )
        source = "субтитры YouTube"
        if not transcript_text:
            # Нет субтитров — пробуем аудио → ASR.
            asr_text, asr_err = await self._yt_audio_transcript(yt_url)
            if asr_text:
                transcript_text, source = asr_text, "распознавание аудио (ASR)"
            else:
                transcript_err = f"{transcript_err}; ASR: {asr_err}"

        lines = [f"URL: {yt_url}"]
        if title:
            lines.append(f"Заголовок: {title}")
        if author:
            lines.append(f"Канал: {author}")

        if transcript_text:
            # Полный транскрипт — в raw/imports (источник), агенту — выдержка.
            side = f"raw/imports/youtube_{video_id}.md"
            try:
                from datetime import datetime as _dt, timezone as _tz  # noqa: PLC0415
                body = (
                    "---\ntype: extracted\nkind: youtube\n"
                    f"source: {yt_url}\ntitle: \"{(title or video_id).replace(chr(34), chr(39))}\"\n"
                    f"channel: \"{(author or '').replace(chr(34), chr(39))}\"\n"
                    f"transcript: {source}\nextracted: {_dt.now(_tz.utc).strftime('%Y-%m-%d %H:%M')}\n---\n\n"
                    f"# {title or video_id}\n\n{yt_url}\n\n## Стенограмма\n\n{transcript_text}\n"
                )
                p = self.wiki.root / side
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body, encoding="utf-8")
                lines.append(f"Полная стенограмма ({len(transcript_text)} симв., {source}) сохранена: `{side}` "
                             f"— читай read_file/grep_file при необходимости; ссылайся на неё как [[{side[:-3]}]].")
            except Exception as exc:  # noqa: BLE001
                log.warning("youtube sidecar write failed: %s", exc)
            excerpt = transcript_text if len(transcript_text) <= 6000 else transcript_text[:6000] + "\n…(обрезано)"
            lines.append(f"\nСтенограмма ({source}):\n{excerpt}")
        else:
            lines.append(f"\n[Стенограмма недоступна: {transcript_err}]")

        result = "\n".join(lines)
        log.info("youtube %s → title=%r transcript=%d chars (%s)", video_id, title, len(transcript_text), source)
        return result

    async def _yt_audio_transcript(self, yt_url: str) -> tuple[str, str]:
        """Скачать аудио yt-dlp и распознать кусками по 8 минут (лимит ASR 500 с)."""
        import asyncio
        import shutil
        import tempfile
        from ..media_parser import transcribe_audio  # noqa: PLC0415
        if shutil.which("yt-dlp") is None:
            try:
                import yt_dlp  # type: ignore[import]  # noqa: F401
                cmd = ["python", "-m", "yt_dlp"]
                import sys as _sys
                cmd[0] = _sys.executable
            except ImportError:
                return "", "yt-dlp не установлен"
        else:
            cmd = ["yt-dlp"]
        tmp = Path(tempfile.mkdtemp(prefix="yt_"))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, "-q", "--no-warnings", "--no-playlist", "-x", "--audio-format", "m4a",
                "--max-filesize", "200m", "-o", str(tmp / "audio.%(ext)s"), yt_url,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await asyncio.wait_for(proc.communicate(), timeout=600)
            audio = next(iter(tmp.glob("audio.*")), None)
            if proc.returncode != 0 or audio is None:
                return "", (err or b"").decode(errors="replace")[-300:].strip() or "yt-dlp failed"
            # Режем на сегменты по 480 с (wav 16k mono) — под лимит ASR 500 с.
            seg_dir = tmp / "seg"
            seg_dir.mkdir()
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(audio), "-vn", "-ar", "16000", "-ac", "1",
                "-f", "segment", "-segment_time", "480", str(seg_dir / "part_%03d.wav"),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await asyncio.wait_for(proc.communicate(), timeout=300)
            parts = sorted(seg_dir.glob("part_*.wav"))
            if proc.returncode != 0 or not parts:
                return "", "ffmpeg segment failed"
            if len(parts) > 12:  # ~1.5 часа максимум
                parts = parts[:12]
            out: list[str] = []
            for i, part in enumerate(parts):
                txt = await transcribe_audio(part)
                if txt and not txt.startswith("("):
                    m, s = divmod(i * 480, 60)
                    out.append(f"[{m}:{s:02d}] {txt.strip()}")
            return "\n".join(out), "" if out else "пустая транскрипция"
        except asyncio.TimeoutError:
            return "", "таймаут скачивания/распознавания"
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def _fetch_url(self, url: str) -> str:
        """Загрузить страницу и вернуть очищенный текст (до 3000 символов)."""
        import re
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"ERROR: неподдерживаемая схема '{parsed.scheme}' (только http/https)"

        # Bilibili: use public API instead of scraping (site blocks bots with 412)
        import re as _re
        _bv_match = _re.search(r"/video/(BV[a-zA-Z0-9]+)", url)
        if _bv_match and "bilibili.com" in parsed.netloc:
            return await ToolExecutor._fetch_bilibili(_bv_match.group(1))

        # YouTube: use transcript API
        _yt_id = ToolExecutor._yt_video_id(url)
        if _yt_id:
            return await self._fetch_youtube(url, _yt_id)

        # 1688: десктопная страница — JS-челлендж, но мобильная m.1688.com отдаёт
        # карточку с JSON (название, цена, MOQ, продавец, характеристики, фото).
        if "1688.com" in parsed.netloc:
            _offer_match = _re.search(r"/offer/(\d+)", url) or _re.search(r"offerId=(\d+)", url)
            offer_id = _offer_match.group(1) if _offer_match else ""
            if offer_id:
                return await ToolExecutor._fetch_1688(offer_id)
            return f"URL: {url}\nЗаголовок: страница 1688.com без offer id\n\n[не карточка товара — не разобрано]"

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
    def _parse_1688_html(html: str, offer_id: str) -> dict:
        """Вытащить поля карточки из HTML мобильной страницы 1688."""
        import json as _json
        import re as _re
        d: dict = {"offer_id": offer_id}
        m = _re.search(r"<title>(.*?)</title>", html, _re.S)
        title = (m.group(1).strip() if m else "")
        m2 = _re.search(r'"offerTitle":"((?:[^"\\]|\\.)*)"', html)
        if m2:
            title = _json.loads('"' + m2.group(1) + '"') or title
        d["title"] = title
        m = _re.search(r'"price":"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)"', html)
        d["price"] = m.group(1) if m else ""
        m = _re.search(r'"beginAmount":"?(\d+)', html)
        d["moq"] = m.group(1) if m else ""
        units = [u for u in _re.findall(r'"unit":"([^"]{1,8})"', html) if u not in ("￥", "¥", "$", "元")]
        d["unit"] = units[0] if units else ""
        m = _re.search(r'"saledCount":"([^"]{1,20})"', html)
        d["sold"] = m.group(1) if m else ""
        m = _re.search(r'"companyName":"([^"]{1,80})"', html)
        d["company"] = m.group(1) if m else ""
        m = _re.search(r'"sellerLoginId":"([^"]{1,60})"', html)
        d["seller"] = m.group(1) if m else ""
        props: list[tuple[str, str]] = []
        i = html.find('"propsList":')
        if i != -1:
            try:
                arr, _ = _json.JSONDecoder().raw_decode(html[i + len('"propsList":'):])
                for it in arr:
                    if isinstance(it, dict) and it.get("name"):
                        props.append((str(it.get("name")), str(it.get("value", ""))))
            except Exception:  # noqa: BLE001
                pass
        d["props"] = props
        imgs: list[str] = []
        i = html.find('"offerImgList":')
        if i != -1:
            try:
                arr, _ = _json.JSONDecoder().raw_decode(html[i + len('"offerImgList":'):])
                imgs = [str(x) for x in arr if isinstance(x, str) and x.startswith("http")]
            except Exception:  # noqa: BLE001
                pass
        d["images"] = imgs
        return d

    @staticmethod
    def _format_1688(d: dict) -> str:
        oid = d.get("offer_id", "")
        url = f"https://detail.1688.com/offer/{oid}.html"
        lines = [f"URL: {url}", f"Заголовок: {d.get('title') or '(нет)'} (1688, offer {oid})", ""]
        if d.get("price"):
            unit = f" / {d['unit']}" if d.get("unit") else ""
            lines.append(f"Цена: ¥{d['price']}{unit} (диапазон по объёму/SKU; CNY)")
        if d.get("moq"):
            lines.append(f"Мин. заказ: {d['moq']} {d.get('unit') or 'шт'}")
        if d.get("sold"):
            lines.append(f"Продано: {d['sold']}")
        if d.get("company") or d.get("seller"):
            lines.append(f"Продавец: {d.get('company') or ''} {('(' + d['seller'] + ')') if d.get('seller') else ''}".strip())
        if d.get("props"):
            lines.append("")
            lines.append("Характеристики (кит.):")
            for name, val in d["props"][:14]:
                v = val if len(val) <= 160 else val[:160] + "…"
                lines.append(f"- {name}: {v}")
        if d.get("images"):
            lines.append("")
            lines.append("Фото (скачать: fetch_image(url)):")
            lines += [f"- {u}" for u in d["images"][:3]]
        lines.append("")
        lines.append("[Текст на китайском — переведи ключевое для пользователя сам.]")
        return "\n".join(lines)

    @staticmethod
    async def _fetch_1688(offer_id: str) -> str:
        """Карточка товара 1688 через мобильную страницу m.1688.com."""
        url = f"https://m.1688.com/offer/{offer_id}.html"
        headers = {
            "User-Agent": ("Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36"),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
            "Accept": "text/html,application/xhtml+xml",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
            html = resp.text if resp.status_code == 200 else ""
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch 1688 %s failed: %s", offer_id, exc)
            html = ""
        clean = f"https://detail.1688.com/offer/{offer_id}.html"
        if not html or ('"offerTitle"' not in html and "<title>" not in html):
            return (
                f"URL: {clean}\nЗаголовок: товар на 1688.com (offer {offer_id})\n\n"
                f"[1688 не отдал карточку (анти-бот/капча). Запиши ссылку как есть: {clean}. "
                f"Пользователь может прислать скриншот карточки — я прочитаю его через vision.]"
            )
        d = ToolExecutor._parse_1688_html(html, offer_id)
        if not d.get("title") or "验证" in d.get("title", ""):
            return f"URL: {clean}\nЗаголовок: товар на 1688.com (offer {offer_id})\n\n[страница-заглушка/капча, карточка недоступна]"
        log.info("fetch 1688 %s → %s", offer_id, d.get("title", "")[:60])
        return ToolExecutor._format_1688(d)

    async def _fetch_image(self, url: str) -> str:
        """Скачать картинку по URL в raw/assets/<date>/web_<hash>.<ext>."""
        import hashlib
        import urllib.parse
        from datetime import datetime as _dt, timezone as _tz
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "ERROR: нужен http(s) URL"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DuckSidian/1.0)", "Referer": f"{parsed.scheme}://{parsed.netloc}/"}
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                data = resp.content
        except Exception as exc:  # noqa: BLE001
            return f"ERROR fetch_image: {exc}"
        if not ct.startswith("image/"):
            return f"ERROR fetch_image: не картинка (content-type {ct or '?'})"
        if len(data) > 10 * 1024 * 1024:
            return "ERROR fetch_image: файл больше 10 МБ"
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(ct, ".jpg")
        h = hashlib.sha1(url.encode()).hexdigest()[:10]
        date_iso = _dt.now(_tz.utc).strftime("%Y-%m-%d")
        rel = self.wiki.save_asset(data, f"web_{h}{ext}", date_iso).replace("\\", "/")
        return f"OK: {rel} ({len(data) // 1024} KB, {ct}). Посмотреть: describe_media('{rel}'); отправить — в блоке IMAGES."

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

    # Имитации браузера для Baidu: без них www.baidu.com/s отдаёт капчу (wappass).
    BAIDU_IMPERSONATE = ("chrome_124", "chrome_120", "edge_122", "safari_17.5")

    @staticmethod
    def _baidu_blocked(html: str, final_url: str = "") -> bool:
        """Baidu вместо выдачи показал капчу / страницу проверки."""
        return (not html) or ("wappass.baidu.com" in (final_url or "")) or ("百度安全验证" in html) \
            or ("c-container" not in html)

    @staticmethod
    def _parse_baidu_html(html: str, max_results: int = 5) -> list[dict]:
        """Разобрать выдачу www.baidu.com/s: заголовок, настоящий URL (атрибут mu, а не
        редирект baidu.com/link), сниппет, источник. Агрегатные блоки Baidu (картинки,
        видео, «百度百科»-карточки без mu) пропускаем."""
        from bs4 import BeautifulSoup  # type: ignore[import]
        soup = BeautifulSoup(html, "html.parser")
        out: list[dict] = []
        seen: set[str] = set()
        for d in soup.select("div.c-container"):
            h3 = d.find("h3")
            if h3 is None:
                continue
            title = h3.get_text(" ", strip=True)
            a = h3.find("a")
            href = (d.get("mu") or (a.get("href") if a else "") or "").strip()
            if not href or "baidu.com/link?" in href or href in seen:
                continue
            if any(x in href for x in ("image.baidu.com", "v.baidu.com", "map.baidu.com")):
                continue
            seen.add(href)
            src_el = d.select_one(".c-showurl, [class*=source_], [class*=siteLink], .c-color-gray")
            source = src_el.get_text(" ", strip=True) if src_el else ""
            text = d.get_text(" ", strip=True)
            if text.startswith(title):
                text = text[len(title):].strip(" -—·|")
            if source and text.endswith(source):
                text = text[: -len(source)].rstrip(" -—·|")
            out.append({"title": title, "href": href, "body": text[:300], "source": source})
            if len(out) >= max_results:
                break
        return out

    BAIDU_API_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    _BAIDU_RECENCY = {"week", "month", "semiyear", "year"}

    @staticmethod
    def _baidu_query(query: str, limit: int = 72) -> str:
        """API режет запрос до 72 «единиц», иероглиф = 2. Обрезаем сами — предсказуемо."""
        out, width = [], 0
        for ch in " ".join(query.split()):
            w = 2 if ord(ch) > 0x2E7F else 1
            if width + w > limit:
                break
            out.append(ch)
            width += w
        return "".join(out)

    async def _baidu_api(self, query: str, max_results: int, sites: list[str] | None,
                         recency: str | None, api_key: str) -> list[dict]:
        """Официальный поиск Baidu (Qianfan ai_search/web_search). Ошибка → исключение."""
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": self._baidu_query(query)}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": max(1, min(max_results, 10))}],
        }
        clean_sites = [str(s).strip() for s in (sites or []) if str(s).strip()][:20]
        if clean_sites:
            body["search_filter"] = {"match": {"site": clean_sites}}
        if recency in self._BAIDU_RECENCY:
            body["search_recency_filter"] = recency
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.BAIDU_API_URL, json=body, headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code != 200 or data.get("code"):
            raise RuntimeError(f"HTTP {resp.status_code} {data.get('code', '')} {data.get('message', '')}".strip())
        out = []
        for r in data.get("references") or []:
            out.append({
                "title": r.get("title", ""), "href": r.get("url", ""),
                "body": (r.get("snippet") or r.get("content") or "")[:500],
                "source": r.get("website") or r.get("web_anchor", ""), "date": r.get("date", ""),
            })
        return out[:max_results]

    async def _baidu_search(self, query: str, max_results: int = 5, *,
                            sites: list[str] | None = None, recency: str | None = None) -> str:
        """Поиск в Baidu: официальный API Qianfan, если задан BAIDU_API_KEY; иначе (или при его
        ошибке) — парсинг www.baidu.com через primp. При капче — DuckDuckGo с регионом cn-zh."""
        import asyncio

        from ..config import settings  # noqa: PLC0415
        api_err = ""
        if settings.baidu_api_key.strip():
            try:
                results = await self._baidu_api(query, max_results, sites, recency,
                                                settings.baidu_api_key.strip())
                log.info("baidu_search(api) %r → %d результатов", query[:60], len(results))
                if results:
                    return json.dumps(results, ensure_ascii=False, indent=2)
                return f"[] — Baidu (официальный API) ничего не нашёл по «{self._baidu_query(query)}»."
            except Exception as exc:  # noqa: BLE001
                api_err = str(exc)[:200]
                log.warning("baidu_search api failed (%s), falling back to scraping", api_err)
        if sites:
            query = f"{query} " + " ".join(f"site:{s}" for s in sites[:3])

        def _sync() -> tuple[list[dict], str]:
            import primp  # type: ignore[import]  # ставится вместе с ddgs
            last = "Baidu не ответил"
            for imp in ToolExecutor.BAIDU_IMPERSONATE:
                try:
                    client = primp.Client(impersonate=imp, timeout=20, follow_redirects=True)
                    resp = client.get(
                        "https://www.baidu.com/s",
                        params={"wd": query, "ie": "utf-8", "rn": str(max(10, max_results))},
                        headers={"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5"},
                    )
                except Exception as exc:  # noqa: BLE001
                    last = f"{imp}: {exc}"
                    continue
                html = resp.text or ""
                if ToolExecutor._baidu_blocked(html, getattr(resp, "url", "") or ""):
                    last = f"{imp}: капча/нет выдачи"
                    log.info("baidu_search: blocked with %s", imp)
                    continue
                return ToolExecutor._parse_baidu_html(html, max_results), ""
            return [], last

        loop = asyncio.get_running_loop()
        try:
            results, err = await loop.run_in_executor(None, _sync)
        except Exception as exc:  # noqa: BLE001
            results, err = [], str(exc)
        if results:
            log.info("baidu_search %r → %d результатов", query[:60], len(results))
            note = f"[API Baidu недоступен ({api_err}), результаты парсинга baidu.com]\n" if api_err else ""
            return note + json.dumps(results, ensure_ascii=False, indent=2)
        log.warning("baidu_search failed (%s), falling back to DuckDuckGo cn-zh", err)

        def _ddg() -> list[dict]:
            try:
                from ddgs import DDGS  # type: ignore[import]
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import]
            with DDGS() as ddgs:
                return list(ddgs.text(query, region="cn-zh", max_results=max_results))

        try:
            fallback = await loop.run_in_executor(None, _ddg)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR baidu_search: Baidu недоступен ({err}); DuckDuckGo тоже: {exc}"
        return (f"[Baidu недоступен: {err}. Ниже результаты DuckDuckGo, регион Китай]\n"
                + json.dumps(fallback, ensure_ascii=False, indent=2))

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
