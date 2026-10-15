# DuckSidian

Telegram-бот, который слушает рабочий чат команды и ведёт LLM-Wiki в Obsidian
по [паттерну Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
Плюс **личный режим**: каждый сотрудник в личке с ботом ведёт своё пространство
(мысли, идеи, проекты, голосовые → текст, подборки фактов) и делится артефактами
с групповым чатом. LLM — любой OpenAI-совместимый вендор (DeepSeek, MiniMax,
Qwen, OpenAI, OpenRouter, Ollama) с переключением на лету. Вся система
поднимается одним скриптом в Docker.

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
│   ├── db.py                 # SQLite (messages, batches, pending_questions, shares, allowlist)
│   ├── wiki.py               # FS-обёртки + path traversal guard
│   ├── orchestrator.py       # ingest / query / lint / summary / personal_turn / share_to_chat
│   ├── personal.py           # личные пространства users/<id>/, артефакты, шаринг
│   ├── scheduler.py          # APScheduler cron daily ingest
│   ├── git_sync.py           # auto-commit & push vault
│   ├── handlers/
│   │   ├── listener.py       # пишет все сообщения чата в SQLite
│   │   ├── commands.py       # групповые команды /ask /ingest /lint /summary /note ...
│   │   ├── private.py        # личный режим (DM): агент, голос, артефакты, /share
│   │   └── admin_shared.py   # /model /allow /deny /users
│   └── agent/
│       ├── deepseek.py       # LLMClient — OpenAI-compatible, мульти-вендор
│       ├── tools.py          # JSON-schemas + ToolExecutor
│       ├── loop.py           # tool-calling loop
│       └── prompts.py        # system prompts (ingest/query/lint/.../personal)
├── vault-template/           # эталонный скелет vault'а
│   ├── AGENTS.md             # СХЕМА — главный конфиг агента
│   ├── index.md
│   ├── log.md
│   ├── raw/{daily,notes,assets}/
│   ├── wiki/{entities,concepts,sources,daily,queries}/
│   └── personal/             # скелет личного пространства (AGENTS.md, index, шаблон проекта)
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

| `/model [provider] [model]` | админ | показать/переключить LLM-вендора (см. ниже) |
| `/allow <id>` / `/deny <id>` / `/users` | админ | доступ сотрудников к личному режиму |

Список админов — `TELEGRAM_ADMINS` в `.env` (через запятую).

## Личный режим (DM с ботом)

Сотрудник из allowlist (`TELEGRAM_ADMINS`, `TELEGRAM_PRIVATE_USERS` или
добавленный через `/allow <id>`) пишет боту в личку и получает своё
изолированное пространство `vault/users/<user_id>/` с той же трёхслойной
структурой (`raw/inbox`, `raw/assets`, `wiki/{projects,notes,ideas,research}`,
`AGENTS.md`, `index.md`, `log.md`). Чужой пользователь по `/start` видит свой id
и подсказку, что попросить у админа.

Любое сообщение проходит через агента (`PERSONAL_SYSTEM`), который сам
классифицирует его:

- **Запись** — мысль, идея, статус по проекту, расшифровка голосового →
  страница проекта / `wiki/notes` / `wiki/ideas` + строка в `index.md`,
  ответ «что и куда записал».
- **Поиск и проверка** — «посмотри / проверь / найди / сравни …» → поиск по
  личному vault, по wiki групповых чатов (`search_team_wiki`, read-only) и в
  интернете (`web_search`, `fetch_url`, `fetch_git_log`) → подборка с
  источниками, сохраняется в `wiki/research/`.
- **Управление артефактами** — «покажи / удали / отправь в чат» →
  `list_dir`/`edit_file`/`delete_file`/`share_to_chat`.

Входные данные: текст, **голосовые и аудио** (Whisper, модель `WHISPER_MODEL`),
видео и кружочки (STT дорожки через ffmpeg), документы (PDF/DOCX/XLSX/текст),
фото (агент смотрит через `describe_media`). Всё складывается в
`raw/inbox/<date>.md` и `raw/assets/<date>/`, диалог хранится в SQLite — агент
получает последние `PERSONAL_HISTORY_SIZE` реплик как контекст.

Команды в личке: `/ask`, `/note`, `/search`, `/projects`, `/list [kind]`,
`/show <№|имя|путь>`, `/del <…>`, `/share <…> [чат] [комментарий]`
(или `/share` реплаем на ответ бота), `/say`, `/voice`, `/voices`, `/imagine`,
`/ws`, `/space`, `/log`, `/chats`, `/help`.

### Воркспейсы = темы в личке

Telegram с Bot API 9.3/9.4 поддерживает темы (topics) в личных чатах с ботом.
Включи у бота в @BotFather «Threaded mode / Topics» — и каждая тема в личке
становится **воркспейсом**: при создании темы бот заводит страницу
`wiki/projects/<Имя>.md`, всё написанное в теме агент относит к этому проекту
(идеи, задачи, решения, подборки с `project: "[[Имя]]"`), история диалога —
своя на тему. Общий чат без темы — «входящие» без привязки.

- `/ws` — текущий воркспейс и кнопки «📤 в <chat>»; `/ws <имя>` — переименовать;
  `/ws new <имя>` — создать тему+проект; `/ws list` — все.
- Совместный воркспейс (🤝) имеет владельца — того, кто первым поделился им «совместно».
  Приглашать может только владелец; название, уже занятое чужим совместным воркспейсом,
  занять нельзя (бот предложит переименовать тему). `/ws members` — кто в нём,
  `/ws kick <user_id|@username>` — исключить участника (его тема отвязывается от общего vault).
- `/ws share <чат> [topic] [комментарий]` — выгрузить **весь воркспейс**: страница
  проекта и все связанные заметки/идеи/подборки ложатся отдельными заметками в
  `raw/notes/` группы (вложения — в `raw/assets/shared/`), каждая попадает в БД
  для ingest, а анонс уходит в бот-топик группы или, с флагом `topic`, в отдельную
  тему группы с тем же названием («вкладка в вкладку»).

Пока агент работает, в личке виден черновик с прогрессом («🔎 ищу в wiki
команды…», «✍️ пишу…») — через `sendMessageDraft`. Меню команд регистрируется
отдельно для лички и групп (`setMyCommands` со скоупами).

### Голос и картинки (MiniMax)

- **Распознавание речи** — MiniMax ASR (`/v1/speech_to_text`, ~2–3 с на
  голосовое, русский с `STT_LANGUAGE=ru`), fallback — локальный Whisper.
- **Синтез речи** — `speech-2.8-hd`, русские голоса (`/voices`, по умолчанию
  `Russian_ReliableMan`). `/say <текст>` озвучивает, `/voice on|off|auto` —
  отвечать голосом (auto: голосом на голосовые). Файл перекодируется ffmpeg в
  OGG/Opus и уходит как голосовое сообщение.
- **Зрение** — фото и кадры видео описывает `MiniMax-M3` через
  `/v1/chat/completions` с base64 `image_url`; используется и в групповом ingest
  (`describe_media`) вместо Ollama moondream, если задан ключ.
- **Картинки** — `image-01` (`/v1/image_generation`): `/imagine <описание> [16:9]`
  или просьба агенту «нарисуй…» (tool `generate_image`); результат ложится в
  `raw/assets/<date>/gen_*.jpg` личного vault.

**Шаринг в групповой чат** (`/share`, кнопка «📤 в <chat>» под ответом или
просьба агенту) делает три вещи: копирует артефакт в `vault/<chat>/raw/notes/`
(вложения — в `raw/assets/shared/<date>/`, ссылки переписываются), кладёт его
в БД как сообщение чата (попадёт в ближайший ingest и в общую wiki) и
анонсирует в бот-топике группы. Все шаринги пишутся в таблицу `shares`.

## LLM-вендоры

Все вендоры — OpenAI-совместимые `chat/completions` с tool-calling.
Реестр — `LLM_PROVIDERS` в `bot/config.py`:

| provider | base_url по умолчанию | ключ |
|---|---|---|
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `minimax` | `https://api.minimax.io/v1` (модель по умолчанию `MiniMax-M3`) | `MINIMAX_API_KEY` |
| `qwen` | `https://token-plan.maas.qwencloudapi.com/compatible-mode/v1` | `QWEN_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `ollama` | `http://localhost:11434/v1` | — |
| `custom` | `LLM_BASE_URL` | `LLM_API_KEY` |

`LLM_PROVIDER` задаёт стартового вендора, `LLM_MODEL`/`LLM_BASE_URL`/`LLM_API_KEY`
переопределяют его параметры. Админ переключает на лету: `/model` — список и
статус ключей, `/model qwen qwen-max` — переключить; выбор сохраняется в БД и
переживает рестарт. Расход токенов (`/balance`) считается по ценам из реестра.
Старые переменные `DEEPSEEK_*` продолжают работать.

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
- Vision для фото — через локальную Ollama `moondream` (если поднята); без неё
  агент видит только метаданные изображения.

## Тесты

```bash
pip install -r bot/requirements.txt pytest pytest-asyncio
pytest
```

## Лицензия

MIT.
