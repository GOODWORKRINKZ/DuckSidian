"""Цикл агента: chat → tool_calls → results → chat → ... → finish."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .deepseek import DeepSeekClient
from .tools import TOOL_SCHEMAS, ToolExecutor

log = logging.getLogger(__name__)


@dataclass
class AgentRun:
    summary: str = ""
    steps: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False


def _ctx_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


async def _compress_old_tool_results(
    client: DeepSeekClient,
    messages: list[dict[str, Any]],
    *,
    keep_last: int = 4,
    min_len: int = 400,
) -> bool:
    """Суммаризировать старые tool-ответы через LLM, чтобы ужать контекст.

    tool_call_id связки не трогаем (DeepSeek требует совпадения), только
    подменяем `content` на короткое summary. Возвращает True если что-то сжали.
    """
    tool_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" and len(str(m.get("content") or "")) >= min_len
    ]
    if len(tool_idxs) <= keep_last:
        return False
    targets = tool_idxs[:-keep_last]
    parts = []
    for n, i in enumerate(targets, 1):
        name = messages[i].get("name") or "tool"
        content = str(messages[i].get("content") or "")
        parts.append(f"### [{n}] tool={name}\n{content}")
    joined = "\n\n".join(parts)
    sys = (
        "Ты — компрессор контекста. Получишь несколько результатов tool-вызовов. "
        "Для КАЖДОГО верни ОДНУ строку формата `[N] краткое summary` "
        "(2-3 предложения, ключевые факты/имена/пути/числа). "
        "Без вступлений. По одной строке на блок."
    )
    try:
        resp = await client.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": joined},
            ],
            tools=None,
        )
        text = resp["choices"][0]["message"].get("content") or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("context compression call failed: %s", exc)
        return False
    # Распарсить строки `[N] ...`.
    summaries: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            n_str, rest = line[1:].split("]", 1)
            n = int(n_str.strip())
            summaries[n] = rest.strip()
        except (ValueError, IndexError):
            continue
    if not summaries:
        log.warning("compression produced no parseable lines, head=%r", text[:200])
        return False
    for n, i in enumerate(targets, 1):
        s = summaries.get(n)
        if s:
            messages[i]["content"] = f"[compressed] {s}"
    return True


async def run_agent(
    *,
    client: DeepSeekClient,
    executor: ToolExecutor,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 30,
    tool_result_cap: int = 12_000,
    ctx_char_budget: int = 60_000,
) -> AgentRun:
    """Один прогон агента до вызова finish() или исчерпания шагов."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    run = AgentRun(transcript=messages)

    for step in range(max_steps):
        run.steps = step + 1
        # Лог размера контекста — чтобы ловить раздувание (DeepSeek рвёт большие тела).
        ctx_chars = _ctx_chars(messages)
        if ctx_chars > ctx_char_budget:
            log.warning(
                "agent step %d: ctx %d > budget %d, asking LLM to compress old tool outputs",
                step, ctx_chars, ctx_char_budget,
            )
            if await _compress_old_tool_results(client, messages):
                ctx_chars = _ctx_chars(messages)
                log.info("agent step %d: compressed, ctx now %d", step, ctx_chars)
        log.info("agent step %d: ctx_msgs=%d ctx_chars=%d", step, len(messages), ctx_chars)
        # На предпоследнем и последнем шаге форсируем finish, чтобы не выйти
        # из цикла с пустым summary.
        steps_left = max_steps - step
        if steps_left <= 2:
            messages.append({
                "role": "system",
                "content": (
                    f"⚠️ Осталось шагов: {steps_left}. БОЛЬШЕ НЕ ИЩИ — "
                    "вызови finish(summary=...) сейчас с тем разбором, "
                    "что у тебя уже есть. Лучше неполный отчёт, чем пустой."
                ),
            })
        resp = await client.chat(messages=messages, tools=TOOL_SCHEMAS)
        try:
            choice = resp["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError):
            log.error("Bad LLM response: %s", resp)
            break

        # Сохраняем assistant-сообщение как есть (для tool_call_id связки).
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content") or "",
        }
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # Модель решила ответить без tool-call → считаем что закончила.
            run.summary = msg.get("content") or ""
            run.finished = True
            break

        finish_called = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                run.summary = str(args.get("summary", ""))
                run.finished = True
                finish_called = True
                log.info("agent step %d: finish(summary=%r)", step, run.summary[:120])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "content": "OK",
                    }
                )
                continue

            log.info("agent step %d: tool=%s args=%s", step, name, str(args)[:200])
            result = await executor.call(name, args)
            log.info("agent step %d: tool=%s result_len=%d head=%r",
                     step, name, len(result), result[:120])
            # Trim too-large tool outputs to keep context window sane.
            if len(result) > tool_result_cap:
                result = result[:tool_result_cap] + "\n...[truncated]"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": result,
                }
            )

        if finish_called:
            break

    if not run.finished:
        # Форс-финал: модель не вызвала finish() за max_steps. Делаем один
        # запрос БЕЗ tools — пусть просто выдаст текст разбора, его и берём.
        log.warning("agent didn't finish in %d steps, forcing plain-text summary", max_steps)
        try:
            messages.append({
                "role": "system",
                "content": (
                    "СТОП. Шаги кончились. Никаких tool-вызовов — их у тебя "
                    "больше нет. Ответь обычным текстом: ТРИЗ-разбор по тому, "
                    "что уже собрано. Структура: Проблема / Противоречие / "
                    "Идеальный конечный результат / 2-4 решения. На русском, кратко."
                ),
            })
            final = await client.chat(messages=messages, tools=None)
            run.summary = (
                final["choices"][0]["message"].get("content")
                or run.summary
                or "(agent stopped without finish())"
            )
            run.finished = True
        except Exception as exc:  # noqa: BLE001
            log.error("forced summary call failed: %s", exc)
            run.summary = run.summary or "(agent stopped without finish())"
    return run
