"""Отказ модерации провайдера (Qwen 400 data_inspection_failed) не роняет ход агента."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bot.agent.deepseek import ModerationRejected, is_moderation_error
from bot.agent.loop import MODERATION_HIDDEN, hide_last_tool_result, run_agent
from bot.agent.tools import ToolExecutor
from bot.wiki import Wiki

_QWEN_BODY = ('{"error":{"code":"data_inspection_failed","param":null,"message":'
              '"Input text data may contain inappropriate content.","type":"data_inspection_failed"}}')


def _http_error(status: int, body: str) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    resp = httpx.Response(status, text=body, request=req)
    return httpx.HTTPStatusError(f"{status}", request=req, response=resp)


class _Client:
    """Ответы по очереди; элемент-исключение бросается вместо ответа."""

    def __init__(self, items):
        self._items = list(items)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append([dict(m) for m in messages])
        item = self._items.pop(0) if self._items else {"choices": [{"message": {"content": "(no more)"}}]}
        if isinstance(item, BaseException):
            raise item
        return item


def _call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": cid, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }]}}]}


@pytest.fixture
def executor(tmp_path: Path) -> ToolExecutor:
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "a.md").write_text("спам из поисковой выдачи", encoding="utf-8")

    async def ask(_q, _o):
        return ""
    return ToolExecutor(Wiki(tmp_path), ask)


def test_is_moderation_error() -> None:
    assert is_moderation_error(_http_error(400, _QWEN_BODY))
    assert is_moderation_error(_http_error(400, '{"error":{"code":"content_filter"}}'))
    assert not is_moderation_error(_http_error(400, '{"error":{"message":"bad tool_call_id"}}'))
    assert not is_moderation_error(_http_error(500, _QWEN_BODY))
    assert not is_moderation_error(ValueError("x"))


def test_hide_last_tool_result_skips_hidden_and_finish() -> None:
    msgs = [
        {"role": "tool", "name": "read_file", "content": "файл"},
        {"role": "tool", "name": "web_search", "content": "спам"},
        {"role": "tool", "name": "finish", "content": "OK"},
    ]
    assert hide_last_tool_result(msgs) == "web_search"
    assert msgs[1]["content"] == MODERATION_HIDDEN
    assert hide_last_tool_result(msgs) == "read_file"
    assert hide_last_tool_result(msgs) is None
    assert msgs[2]["content"] == "OK"


@pytest.mark.asyncio
async def test_rejected_tool_result_is_hidden_and_turn_continues(executor: ToolExecutor) -> None:
    client = _Client([
        _call("read_file", {"path": "wiki/a.md"}),
        _http_error(400, _QWEN_BODY),                  # выдача со спамом → отказ
        _call("finish", {"summary": "Ответил без спама"}, "c2"),
    ])
    run = await run_agent(client=client, executor=executor, system_prompt="s",
                          user_prompt="u", max_steps=5)
    assert run.finished and run.summary == "Ответил без спама"
    retry = client.calls[2]
    tool_msgs = [m for m in retry if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["content"] == MODERATION_HIDDEN
    assert all("спам из поисковой" not in str(m.get("content")) for m in retry)


@pytest.mark.asyncio
async def test_rejected_user_prompt_raises_clear_error(executor: ToolExecutor) -> None:
    client = _Client([_http_error(400, _QWEN_BODY)])
    with pytest.raises(ModerationRejected, match="модерацией"):
        await run_agent(client=client, executor=executor, system_prompt="s",
                        user_prompt="u", max_steps=5)


@pytest.mark.asyncio
async def test_other_http_errors_still_raise(executor: ToolExecutor) -> None:
    client = _Client([_http_error(400, '{"error":{"message":"bad request"}}')])
    with pytest.raises(httpx.HTTPStatusError):
        await run_agent(client=client, executor=executor, system_prompt="s",
                        user_prompt="u", max_steps=5)
