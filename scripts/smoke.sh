#!/usr/bin/env bash
# Простой smoke-тест: поднять стек, дождаться healthz, проверить mount.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DC="docker compose"
$DC up -d --build

echo "==> ждём healthz бота…"
for i in $(seq 1 60); do
    if curl -fs http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
        echo "  bot ok"
        break
    fi
    sleep 2
done

echo "==> проверяю Obsidian web…"
curl -fs -o /dev/null "http://127.0.0.1:${OBSIDIAN_PORT:-3000}/" \
    && echo "  obsidian ok" \
    || echo "  obsidian не отвечает (это нормально первые ~30с)"

echo "==> проверяю vault внутри bot-контейнера…"
$DC exec -T bot ls -la /vault/AGENTS.md /vault/index.md /vault/log.md \
    && echo "  vault примонтирован"
