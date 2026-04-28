# DuckSidian

Telegram-бот, который слушает рабочий чат команды и ведёт LLM-Wiki в Obsidian
по [паттерну Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
LLM — DeepSeek через OpenAI-совместимый API. Вся система поднимается одним
скриптом в Docker.

## Что внутри

```
DuckSidian/
├── bootstrap.sh              # one-shot deploy
├── docker-compose.yml        # bot + obsidian (linuxserver KasmVNC)
├── .env.example              # все настройки
├── docker/Dockerfile.bot
├── bot/                      # aiogram 3 + DeepSeek tool-calling агент
│   ├── main.py               # entrypoint
│   ├── config.py             # pydantic-settings
│   ├── db.py                 # SQLite (messages, batches, pending_questions)
│   ├── wiki.py               # FS-обёртки + path traversal guard
│   ├── orchestrator.py       # ingest / query / lint / summary + ask_user
│   ├── scheduler.py          # APScheduler cron daily ingest
│   ├── git_sync.py           # auto-commit & push vault
│   ├── handlers/
│   │   ├── listener.py       # пишет все сообщения чата в SQLite
│   │   └── commands.py       # /ask /ingest /lint /summary /note ...
│   └── agent/
│       ├── deepseek.py       # OpenAI-compatible client
│       ├── tools.py          # JSON-schemas + ToolExecutor
│       ├── loop.py           # tool-calling loop
│       └── prompts.py        # system prompts
├── vault-template/           # эталонный скелет vault'а
│   ├── AGENTS.md             # СХЕМА — главный конфиг агента
│   ├── index.md
│   ├── log.md
│   ├── raw/{daily,notes,assets}/
│   └── wiki/{entities,concepts,sources,daily,queries}/
├── tests/                    # pytest
└── scripts/smoke.sh
```

## Архитектура

Три слоя по Карпатому:

- **`raw/`** — read-only источник истины. Бот сам пишет сюда `raw/daily/<date>.md`,
  агент только читает.
- **`wiki/`** — то что пишет/обновляет агент: страницы сущностей, концепций,
  внешних источников, дневные дайджесты.
- **Schema** — `AGENTS.md` + `index.md` + `log.md`. Именно `AGENTS.md` делает
  агента дисциплинированным wiki-куратором, а не болталкой.

Каждое сообщение Telegram размечается в `raw/daily/*.md` якорем `^msg-<id>` —
агент обязан ставить такие citations в любую новую запись в `wiki/`.

## Поток данных

1. Бот добавлен в супергруппу с `privacy mode = disabled` → ловит **все**
   сообщения и пишет их в SQLite (`bot/data/bot.sqlite3`).
2. По cron (`INGEST_CRON`, по умолчанию 23:00) `scheduler.py` дёргает
   `Orchestrator.ingest_for_date(today)`:
   - выгружает сообщения из БД,
   - формирует `raw/daily/YYYY-MM-DD.md`,
   - запускает агентский цикл с системным промптом ingest.
3. Агент через tool-calls читает `AGENTS.md`, `index.md`, последние строки
   `log.md`, дневной батч; обновляет/создаёт страницы в `wiki/`,
   дописывает `index.md` и `log.md`. Если что-то неоднозначно — вызывает
   `ask_user(question, options)`, бот постит вопрос в forum-топик
   (`TELEGRAM_TOPIC_ID`) с inline-кнопками; команда отвечает кнопкой или
   reply'ем — ответ возвращается агенту.
4. После ingest — `git commit && push` (опц.).

## Команды бота

| Команда | Кто | Что делает |
|---|---|---|
| `/ask <вопрос>` | все | агент-query поверх `wiki/` |
| `/search <строка>` | все | подстроковый поиск по vault'у |
| `/summary day\|week` | все | сводка по `wiki/daily/*` |
| `/note <текст>` | все | ручная запись в `raw/notes/` |
| `/log` | все | последние 10 строк `log.md` |
| `/help` | все | помощь |
| `/ingest [today\|N\|YYYY-MM-DD]` | админ | внеплановый ingest |
| `/lint` | админ | health-check вики |
| `/pause` / `/resume` | админ | приостановить cron |

Список админов — `TELEGRAM_ADMINS` в `.env` (через запятую).

## Запуск

```bash
git clone <this repo>
cd DuckSidian

# 1) первый запуск создаст .env из шаблона
./bootstrap.sh

# 2) заполни .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY и т.д.)
$EDITOR .env

# 3) повторный запуск — поднимет стек
./bootstrap.sh
```

После старта:
- Obsidian web: <http://localhost:3000> (логин/пароль из `.env`).
- Healthcheck бота: `curl http://127.0.0.1:8080/healthz`.
- Логи: `docker compose logs -f bot`.

## Подготовка Telegram

1. Создай бота через [@BotFather](https://t.me/BotFather).
2. **Важно:** `/mybots → твой бот → Bot Settings → Group Privacy → Turn off`.
   Без этого бот в группе видит только сообщения с упоминанием.
3. Добавь бота в рабочую супергруппу. Желательно дать админку (нужно для
   `message_thread_id` и кнопок).
4. Включи в супергруппе **Topics (форум)**, создай отдельный топик
   «DuckSidian» — туда бот будет постить дневные сводки и вопросы.
5. Узнай `chat_id` (отрицательный, начинается с `-100`) и `message_thread_id`
   нужного топика. Самый простой способ — отправить любое сообщение и
   посмотреть в логах бота (`docker compose logs bot | grep chat`).

## Замечания по безопасности

- Все FS-tools агента ограничены `/vault`. Запись агента в `raw/` запрещена
  на уровне `ToolExecutor` и `Wiki.resolve(for_write=True)`.
- Управляющие команды (`/ingest`, `/lint`, `/pause`, `/resume`) только для
  админов из `TELEGRAM_ADMINS`.
- Obsidian web под basic auth (KasmVNC). Для публичного доступа поставь
  reverse-proxy с TLS (caddy/traefik).
- Vault — отдельный git-репо в volume; auto-commit после ingest даёт
  бесплатную версионность.

## Что *не* делает (out of scope)

- Не использует userbot/Telethon (риск бана). Поэтому первый день после
  добавления бота в чат может быть неполным — бот видит только сообщения,
  пришедшие после момента добавления.
- Не строит vector-search/RAG индекс — `index.md` достаточно по Карпатому
  до сотен страниц.
- Не обрабатывает медиа (фото/видео/файлы) в текстовом виде. Расширение —
  скачивание в `raw/assets/<date>/` + OCR-сайдкар.

## Тесты

```bash
pip install -r bot/requirements.txt pytest pytest-asyncio
pytest
```

## Лицензия

MIT.
