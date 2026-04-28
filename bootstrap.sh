#!/usr/bin/env bash
# bootstrap.sh — идемпотентная инициализация DuckSidian.
# 1) Проверяет docker / docker compose.
# 2) Создаёт .env из .env.example, если его нет.
# 3) Если ./vault пустой — копирует туда vault-template и инициализирует git.
# 4) Поднимает docker compose.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "==> DuckSidian bootstrap"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker не найден. Установи Docker Engine." >&2
    exit 1
fi

if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    echo "ERROR: ни 'docker compose', ни 'docker-compose' не найдены." >&2
    exit 1
fi

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "==> Создан .env из .env.example."
    echo "    Заполни обязательные поля (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,"
    echo "    DEEPSEEK_API_KEY и др.) и перезапусти ./bootstrap.sh"
    exit 0
fi

# Проверим что обязательные поля непустые.
missing=()
for key in TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID DEEPSEEK_API_KEY; do
    val="$(grep -E "^${key}=" .env | head -n1 | cut -d= -f2- || true)"
    if [[ -z "${val// }" ]]; then
        missing+=("$key")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "ERROR: в .env не заполнены: ${missing[*]}" >&2
    exit 1
fi

mkdir -p vault bot/data obsidian-config

if [[ -z "$(ls -A vault 2>/dev/null || true)" ]]; then
    echo "==> Vault пуст, копирую скелет из vault-template/"
    cp -a vault-template/. vault/
    if command -v git >/dev/null 2>&1; then
        (
            cd vault
            git init -q
            git add -A
            git -c user.name="DuckSidian Bootstrap" \
                -c user.email="bootstrap@ducksidian.local" \
                commit -q -m "init: vault skeleton" || true
        )
    fi
    echo "==> Vault инициализирован."
else
    echo "==> Vault уже не пуст, пропускаю инициализацию."
fi

# uid обсидиана (linuxserver image хочет владельца 1000:1000)
if command -v chown >/dev/null 2>&1; then
    sudo_cmd=""
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        sudo_cmd="sudo"
    fi
    $sudo_cmd chown -R 1000:1000 vault bot/data obsidian-config 2>/dev/null || true
fi

echo "==> Сборка и запуск docker compose"
$DC pull obsidian || true
$DC up -d --build

# Подсказки
echo
echo "==> Готово."
echo "    Obsidian web:  http://localhost:${OBSIDIAN_PORT:-3000}"
echo "    Bot healthz:   http://127.0.0.1:8080/healthz"
echo
echo "Дальше:"
echo " 1) Добавь бота в рабочую супергруппу как админа."
echo " 2) В @BotFather: /setprivacy → Disable (чтобы бот видел все сообщения)."
echo " 3) Создай в супергруппе forum-топик 'DuckSidian' и пропиши TELEGRAM_TOPIC_ID в .env."
echo " 4) Проверь логи: $DC logs -f bot"
