"""Репосты: защита от «а кто автор?», когда автор уже в пометке; подсказка в промпте."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.agent.loop import run_agent
from bot.agent.tools import ToolExecutor
from bot.config import settings
from bot.forwards import AUTHOR_NUDGE, FORWARD_MARK, asks_for_author, forward_hint, is_forward
from bot.personal import PersonalSpace


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            return {"choices": [{"message": {"content": "(no more)"}}]}
        return self._responses.pop(0)


def _call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": cid, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }]}}]}


def _text(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PersonalSpace:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    (tmp_path / "AGENTS.md").write_text("# schema", encoding="utf-8")
    return PersonalSpace(42, "Иван")


def test_asks_for_author_detection() -> None:
    yes = [
        "Но подожди — ты сказал, что это репост. Кто автор? Скинь имя или перешли сообщение с контактом.",
        "Кто автор репоста? (Имя/ник в Телеге, чтобы я мог подколоть именно его)",
        "От кого это сообщение?",
        "Перешли ещё раз с видимым именем.",
        "Мне нужно знать, от кого сообщение, чтобы привязать троллинг.",
        "Чей это репост — скажи, докручу.",
    ]
    no = [
        "Денис, это гениально 😂",
        "Автор — Denis M (@deanitz), и он явно не шарит в шахматах.",
        "Кто хочет стать миллионером — классика.",
        "Записал в проект, автор репоста указан в Истории.",
        "",
    ]
    for t in yes:
        assert asks_for_author(t), t
    for t in no:
        assert not asks_for_author(t), t


def test_is_forward_and_hint() -> None:
    assert is_forward(f"{FORWARD_MARK}, автор — пользователь X. Текст репоста ниже:\n\nпривет")
    assert not is_forward("привет")
    hint = forward_hint("Krikz Ster")
    assert "Krikz Ster" in hint and "VOICE:" in hint and "НЕ спрашивай" in hint


@pytest.mark.asyncio
async def test_answer_check_nudges_once(space: PersonalSpace) -> None:
    async def ask(_q, _o):
        return ""
    client = _FakeClient([
        _text("Классный ролик! Но кто автор? Перешли с именем."),
        _text("Денис, ролик огонь — а ты, как тот брат Володя, не понял вопроса 😄"),
    ])

    def check(ans: str) -> str | None:
        return AUTHOR_NUDGE if asks_for_author(ans) else None

    run = await run_agent(client=client, executor=ToolExecutor(space.wiki, ask), system_prompt="s",
                          user_prompt="u", max_steps=6, answer_check=check)
    assert run.check_nudged and run.finished
    assert run.summary.startswith("Денис")
    nudges = [m for m in client.calls[-1] if m.get("content") == AUTHOR_NUDGE]
    assert len(nudges) == 1


@pytest.mark.asyncio
async def test_answer_check_not_triggered(space: PersonalSpace) -> None:
    async def ask(_q, _o):
        return ""
    client = _FakeClient([_text("Денис, это гениально.")])
    run = await run_agent(client=client, executor=ToolExecutor(space.wiki, ask), system_prompt="s",
                          user_prompt="u", max_steps=3,
                          answer_check=lambda a: AUTHOR_NUDGE if asks_for_author(a) else None)
    assert not run.check_nudged and run.finished and len(client.calls) == 1
