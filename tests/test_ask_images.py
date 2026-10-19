"""ask_choice с картинками: пути из vault уходят пользователю ДО вопроса (иначе он
выбирает между вариантами, которых не видел)."""
from __future__ import annotations

from pathlib import Path

import pytest

from bot.agent.tools import ToolExecutor
from bot.wiki import Wiki


@pytest.mark.asyncio
async def test_ask_choice_passes_images_only_when_given(tmp_path: Path) -> None:
    calls: list = []

    async def choice(q, opts, images=None):
        calls.append((q, opts, images))
        return "2"

    ex = ToolExecutor(Wiki(tmp_path), lambda q, o: None, choice_fn=choice)  # type: ignore[arg-type]
    await ex.call("ask_choice", {"question": "Какой?", "options": ["1", "2"]})
    assert calls[-1] == ("Какой?", ["1", "2"], None)
    res = await ex.call("ask_choice", {"question": "Какой?", "options": ["1", "2"],
                                       "images": ["raw/assets/a.jpg", " ", "raw/assets/b.jpg"]})
    assert res == "Ответ пользователя: 2"
    assert calls[-1][2] == ["raw/assets/a.jpg", "raw/assets/b.jpg"]


@pytest.mark.asyncio
async def test_ask_choice_images_compatible_with_two_arg_callback(tmp_path: Path) -> None:
    async def choice(q, opts):
        return "ok"
    ex = ToolExecutor(Wiki(tmp_path), lambda q, o: None, choice_fn=choice)  # type: ignore[arg-type]
    assert await ex.call("ask_choice", {"question": "q", "options": ["a"]}) == "Ответ пользователя: ok"


@pytest.mark.asyncio
async def test_ask_private_sends_images_before_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from bot.orchestrator import Orchestrator

    sent: list[str] = []

    class FakeBot:
        async def send_photo(self, chat_id, photo, **kw):
            sent.append("photo")

        async def send_media_group(self, chat_id, media, **kw):
            sent.append(f"group{len(media)}")

        async def send_message(self, **kw):
            sent.append("question")
            return SimpleNamespace(message_id=5)

    class FakeDB:
        async def add_question(self, **kw):
            return 7

        async def attach_question_post(self, qid, mid):
            pass

        async def answer_question(self, qid, ans):
            pass

    orch = Orchestrator.__new__(Orchestrator)
    orch.bot, orch.db = FakeBot(), FakeDB()
    orch._waiters, orch._private_pending, orch._asked_in = {}, {}, set()
    img1, img2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    img1.write_bytes(b"x"), img2.write_bytes(b"y")

    import asyncio

    async def answer_soon():
        await asyncio.sleep(0.01)
        await orch.deliver_answer(7, "1")

    asyncio.get_running_loop().create_task(answer_soon())
    ans = await orch.ask_private("Какой?", ["1", "2"], 10, None, timeout=2, images=[img1, img2])
    assert ans == "1" and sent == ["group2", "question"]
    sent.clear()
    asyncio.get_running_loop().create_task(answer_soon())
    await orch.ask_private("Какой?", ["1"], 10, None, timeout=2, images=[img1])
    assert sent == ["photo", "question"]
