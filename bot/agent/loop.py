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
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_cached: int = 0


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
    tool_result_cap: int = 6_000,
    ctx_char_budget: int = 400_000,
    compress_ctx_threshold: int = 80_000,
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
        if ctx_chars > compress_ctx_threshold and step > 0:
            compressed = await _compress_old_tool_results(client, messages, keep_last=3)
            if compressed:
                ctx_chars = _ctx_chars(messages)
                log.info("agent step %d: context compressed, new ctx_chars=%d", step, ctx_chars)
        if ctx_chars > ctx_char_budget:
            log.warning(
                "agent step %d: ctx %d > budget %d",
                step, ctx_chars, ctx_char_budget,
            )
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
        usage = resp.get("usage") or {}
        run.tokens_prompt += usage.get("prompt_tokens", 0)
        run.tokens_completion += usage.get("completion_tokens", 0)
        run.tokens_cached += usage.get("prompt_cache_hit_tokens", 0)
        try:
            choice = resp["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError):
            log.error("Bad LLM response: %s", resp)
            break

        # Сохраняем assistant-сообщение как есть (для tool_call_id связки).
        raw_content = msg.get("content") or ""
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": raw_content,
        }
        # DeepSeek thinking mode: reasoning_content нужно передавать обратно
        # в следующих запросах, иначе API вернёт 400.
        reasoning = msg.get("reasoning_content")
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # DeepSeek иногда пишет tool-calls прямо в content как XML-разметку
        # (с fullwidth || или ASCII ||) вместо структурированного tool_calls.
        # Детектируем по наличию 'DSML' или '<invoke ' в тексте.
        _is_dsml = bool(raw_content) and (
            "DSML" in raw_content or "<invoke " in raw_content
        )
        if _is_dsml:
            log.warning(
                "agent step %d: DeepSeek returned XML tool-call in content "
                "(tool_calls=%d), asking to reformat. head=%r",
                step, len(tool_calls), raw_content[:120],
            )
            messages.append({
                "role": "user",
                "content": (
                    "Используй tool_calls API, не пиши XML вручную. "
                    "Повтори то же действие через tool call."
                ),
            })
            continue

        if not tool_calls:
            # Модель решила ответить без tool-call → считаем что закончила.
            run.summary = raw_content
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
            fusage = final.get("usage") or {}
            run.tokens_prompt += fusage.get("prompt_tokens", 0)
            run.tokens_completion += fusage.get("completion_tokens", 0)
            run.tokens_cached += fusage.get("prompt_cache_hit_tokens", 0)
            forced_text = (
                final["choices"][0]["message"].get("content")
                or run.summary
                or "(agent stopped without finish())"
            )
            # Forced call also sometimes returns DSML XML — sanitize immediately.
            if "DSML" in forced_text:
                log.warning("forced summary contains DSML, sanitizing")
                forced_text = _sanitize_summary(forced_text)
            run.summary = forced_text
            run.finished = True
        except Exception as exc:  # noqa: BLE001
            log.error("forced summary call failed: %s", exc)
            run.summary = run.summary or "(agent stopped without finish())"
    run.summary = _sanitize_summary(run.summary)
    log.info(
        "agent done: steps=%d tokens_prompt=%d tokens_completion=%d "
        "tokens_total=%d tokens_cached=%d",
        run.steps,
        run.tokens_prompt,
        run.tokens_completion,
        run.tokens_prompt + run.tokens_completion,
        run.tokens_cached,
    )
    return run


# DeepSeek иногда возвращает в content свой внутренний tool-call формат как plain text:
#   <｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="..."><｜｜DSML｜｜parameter ...>
# Количество fullwidth-pipes варьируется (｜ или ｜｜).
# Чистим: находим первое вхождение DSML, откатываемся до '<', обрезаем.
import re as _re


def _sanitize_summary(s: str) -> str:
    if not s or "DSML" not in s:
        return s.strip()
    idx = s.find("DSML")
    lt = s.rfind("<", 0, idx)
    if lt != -1:
        s = s[:lt].rstrip()
    return s.strip()
