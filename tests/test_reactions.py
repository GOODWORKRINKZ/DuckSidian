"""Реакции: только разрешённые Telegram эмодзи, REACTION_INVALID не шумит."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from bot.handlers import commands
from bot.handlers.commands import REACT_ERROR, REACT_QUEUED, REACT_SEEN, TG_REACTIONS, _react


class _FakeBot:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.sent: list[str] = []

    async def set_message_reaction(self, *, chat_id, message_id, reaction):
        self.sent.append(reaction[0].emoji)
        if self.fail:
            raise RuntimeError(self.fail)


def _msg(chat_id: int = 1, message_id: int = 10):
    return SimpleNamespace(chat=SimpleNamespace(id=chat_id), message_id=message_id)


@pytest.fixture(autouse=True)
def _reset_bad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "_bad_reactions", set())


def test_constants_are_allowed() -> None:
    assert {REACT_SEEN, REACT_QUEUED, REACT_ERROR} <= TG_REACTIONS
    assert "⏳" not in TG_REACTIONS and "❌" not in TG_REACTIONS


@pytest.mark.asyncio
async def test_legacy_emoji_mapped() -> None:
    bot = _FakeBot()
    await _react(bot, _msg(), "⏳")   # type: ignore[arg-type]
    await _react(bot, _msg(), "❌")   # type: ignore[arg-type]
    await _react(bot, _msg(), "✍️")   # с variation selector
    assert bot.sent == [REACT_QUEUED, REACT_ERROR, "✍"]


@pytest.mark.asyncio
async def test_unknown_emoji_skipped() -> None:
    bot = _FakeBot()
    await _react(bot, _msg(), "✅")   # type: ignore[arg-type]
    assert bot.sent == []


@pytest.mark.asyncio
async def test_reaction_invalid_quiet_and_remembered(caplog: pytest.LogCaptureFixture) -> None:
    bot = _FakeBot(fail="Telegram server says - Bad Request: REACTION_INVALID")
    with caplog.at_level(logging.DEBUG, logger="bot.handlers.commands"):
        await _react(bot, _msg(chat_id=-100), REACT_QUEUED)   # type: ignore[arg-type]
        await _react(bot, _msg(chat_id=-100), REACT_QUEUED)   # type: ignore[arg-type]
    assert bot.sent == [REACT_QUEUED]                         # второй раз даже не пробуем
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    await _react(bot, _msg(chat_id=5), REACT_QUEUED)          # type: ignore[arg-type]
    assert len(bot.sent) == 2                                 # в другом чате пробуем


@pytest.mark.asyncio
async def test_other_errors_still_warn(caplog: pytest.LogCaptureFixture) -> None:
    bot = _FakeBot(fail="Bad Request: message to react not found")
    with caplog.at_level(logging.WARNING, logger="bot.handlers.commands"):
        await _react(bot, _msg(), REACT_SEEN)   # type: ignore[arg-type]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
