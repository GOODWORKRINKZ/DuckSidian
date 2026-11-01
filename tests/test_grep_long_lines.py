"""grep_file по файлу с одной огромной строкой (выгрузка flashcards, 2026-10-30):
совпадение в середине строки обрезалось лимитом результата, агент раздувал regex
40 шагов. Теперь — окна вокруг совпадений и стоп на серии grep по одному файлу."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.agent.tools import ToolExecutor
from bot.config import settings
from bot.personal import PersonalSpace

DECK = "raw/imports/flashcards.html.md"


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolExecutor:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    (tmp_path / "AGENTS.md").write_text("# schema", encoding="utf-8")
    space = PersonalSpace(42, "Иван")
    vocab = [{"hanzi": f"字{i}", "pinyin": "zì", "translation": f"слово {i}"} for i in range(400)]
    vocab[300] = {"hanzi": "家", "pinyin": "jiā", "translation": "дом, семья"}
    (space.wiki.root / "raw/imports").mkdir(parents=True, exist_ok=True)
    (space.wiki.root / DECK).write_text(
        "<script>\nconst VOCAB = " + json.dumps(vocab, ensure_ascii=False) + ";\n</script>\n",
        encoding="utf-8")

    async def ask(question, options):  # pragma: no cover - не вызывается
        return options[0]

    return ToolExecutor(space.wiki, ask)


async def test_long_line_shows_window_around_match(executor):
    out = await executor.call("grep_file", {"path": DECK, "pattern": '"家"', "context": 0})
    assert len(out) < 2000
    assert '"translation": "дом, семья"' in out
    assert "длинная строка" in out


async def test_short_lines_unchanged(executor):
    out = await executor.call("grep_file", {"path": DECK, "pattern": "^const|script", "context": 0})
    assert "<script>" in out


async def test_grep_streak_on_one_file_is_stopped(executor):
    for i in range(ToolExecutor.GREP_STREAK_LIMIT):
        out = await executor.call("grep_file", {"path": DECK, "pattern": f'"家".{{0,{i}}}'})
        assert "СТОП" not in out
    out = await executor.call("grep_file", {"path": DECK, "pattern": '"家".{0,99}'})
    assert "СТОП" in out


async def test_other_tool_resets_grep_streak(executor):
    for _ in range(ToolExecutor.GREP_STREAK_LIMIT):
        await executor.call("grep_file", {"path": DECK, "pattern": "VOCAB"})
    await executor.call("list_dir", {"path": "raw"})
    out = await executor.call("grep_file", {"path": DECK, "pattern": "VOCAB"})
    assert "СТОП" not in out
