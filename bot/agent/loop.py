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


async def run_agent(
    *,
    client: DeepSeekClient,
    executor: ToolExecutor,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 30,
) -> AgentRun:
    """Один прогон агента до вызова finish() или исчерпания шагов."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    run = AgentRun(transcript=messages)

    for step in range(max_steps):
        run.steps = step + 1
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
            if len(result) > 12_000:
                result = result[:12_000] + "\n...[truncated]"
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
        run.summary = run.summary or "(agent stopped without finish())"
    return run
