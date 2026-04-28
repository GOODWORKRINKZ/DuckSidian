"""Тесты agent/loop.py с моком DeepSeek-клиента."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bot.agent.loop import run_agent
from bot.agent.tools import ToolExecutor
from bot.wiki import Wiki


class FakeClient:
    """Возвращает заранее заготовленные ответы по очереди."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice="auto",
                   temperature=0.2):
        self.calls.append(list(messages))
        if not self._responses:
            return {"choices": [{"message": {"content": "(no more)"}}]}
        return self._responses.pop(0)


def _assistant_with_tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    import json
    return {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }],
            }
        }]
    }


@pytest.fixture
def wiki(tmp_path: Path) -> Wiki:
    (tmp_path / "wiki" / "entities").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("schema content")
    return Wiki(tmp_path)


@pytest.mark.asyncio
async def test_agent_finishes(wiki: Wiki) -> None:
    async def ask_user(_q, _opts):
        return "n/a"

    executor = ToolExecutor(wiki, ask_user)
    client = FakeClient([
        _assistant_with_tool_call("read_file", {"path": "AGENTS.md"}, "c1"),
        _assistant_with_tool_call("write_file",
                                  {"path": "wiki/entities/Foo.md",
                                   "content": "# Foo\n"}, "c2"),
        _assistant_with_tool_call("finish", {"summary": "done"}, "c3"),
    ])
    run = await run_agent(
        client=client,  # type: ignore[arg-type]
        executor=executor,
        system_prompt="sys",
        user_prompt="do it",
        max_steps=10,
    )
    assert run.finished
    assert run.summary == "done"
    assert (wiki.root / "wiki/entities/Foo.md").exists()


@pytest.mark.asyncio
async def test_agent_cannot_write_raw(wiki: Wiki) -> None:
    async def ask_user(_q, _opts):
        return ""

    executor = ToolExecutor(wiki, ask_user)
    client = FakeClient([
        _assistant_with_tool_call(
            "write_file",
            {"path": "raw/daily/x.md", "content": "evil"},
            "c1",
        ),
        _assistant_with_tool_call("finish", {"summary": "stop"}, "c2"),
    ])
    run = await run_agent(
        client=client,  # type: ignore[arg-type]
        executor=executor,
        system_prompt="sys",
        user_prompt="try",
        max_steps=5,
    )
    assert run.finished
    assert not (wiki.root / "raw/daily/x.md").exists()
