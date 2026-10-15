#!/usr/bin/env bash
# Перезапуск бота (ручной запуск без systemd, как на стенде) только когда он простаивает:
# ждём GET /busy == 0 (идущие ходы агента + очередь + фоновые разборы), потом SIGTERM и старт.
# Несколько ходов идут параллельно в разных темах, поэтому строки лога для этого не годятся.
#   scripts/restart_when_idle.sh [max_wait_sec=1800]   — FORCE=1 перезапустит и по таймауту.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAX_WAIT="${1:-1800}"

busy() { curl -fs --max-time 5 http://127.0.0.1:8080/busy 2>/dev/null || echo "?"; }

waited=0
while :; do
    b="$(busy)"
    [ "$b" = "0" ] && break
    if [ "$b" = "?" ]; then echo "бот не отвечает на /busy — перезапускаю"; break; fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        if [ "${FORCE:-0}" = "1" ]; then echo "таймаут, busy=$b — FORCE=1, перезапускаю"; break; fi
        echo "таймаут ${MAX_WAIT}s, бот занят (busy=$b) — НЕ перезапускаю"; exit 1
    fi
    echo "busy=$b, жду…"; sleep 10; waited=$((waited + 10))
done

PID="$(pgrep -f '^\.venv/bin/python -m bot\.main' || true)"
if [ -n "$PID" ]; then
    kill -TERM $PID
    for _ in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 1; done
    kill -0 $PID 2>/dev/null && kill -9 $PID
fi
nohup setsid .venv/bin/python -m bot.main >> bot.log 2>&1 < /dev/null &
for _ in $(seq 1 30); do
    if curl -fs http://127.0.0.1:8080/healthz >/dev/null 2>&1; then echo "bot up"; exit 0; fi
    sleep 2
done
echo "бот не поднялся — смотри bot.log"; exit 1
