"""Хелперы правки vault, скилы, guard по успешным записям и живая плашка прогресса."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.agent.loop import run_agent
from bot.agent.tools import PERSONAL_WRITE_TOOLS, ToolExecutor, personal_tool_schemas
from bot.agent.wikitools import (
    append_to_section, big_file_view, broken_links, fuzzy_replace, outline, sections, similar_pages,
)
from bot.handlers.private import render_progress
from bot.orchestrator import add_footer, writes_footer
from bot.skills import load_skills, match_command
from bot.wiki import Wiki

PAGE = """---
title: P
---
# P

## Цель
Сделать робота.

## Задачи
- [ ] раз
- [ ] два

## Решения
- 2026-10-01: колёса BLDC
"""


# --- fuzzy_replace --------------------------------------------------------

def test_fuzzy_replace_exact_and_whitespace() -> None:
    out, how = fuzzy_replace(PAGE, "- [ ] два", "- [x] два")
    assert how == "exact" and "- [x] два" in out
    out, how = fuzzy_replace(PAGE, "Сделать   робота.\r\n", "Сделать ровер.")
    assert how == "whitespace-normalized"
    assert "Сделать ровер.\n\n## Задачи" in out


def test_fuzzy_replace_not_found_points_to_closest_line() -> None:
    out, info = fuzzy_replace(PAGE, "- [ ] двa (опечатка)", "x")
    assert out is None and "Похожее место" in info and "два" in info


def test_fuzzy_replace_ambiguous() -> None:
    out, info = fuzzy_replace("a\nb\na\n", "a", "c")
    assert out is None and "2 раз" in info


# --- append_to_section ----------------------------------------------------

def test_append_to_section_inserts_at_end_of_section() -> None:
    out, info = append_to_section(PAGE, "задачи", "- [ ] три")
    assert "- [ ] два\n- [ ] три\n\n## Решения" in out
    assert "Задачи" in info


def test_append_to_section_ignores_emoji_and_hashes() -> None:
    page = "## 🔥 Задачи\n- a\n"
    out, _ = append_to_section(page, "## Задачи", "- b")
    assert out == "## 🔥 Задачи\n- a\n- b\n"


def test_append_to_section_creates_missing() -> None:
    out, info = append_to_section(PAGE, "Артефакты", "- [[x]]")
    assert out.endswith("## Артефакты\n- [[x]]\n") and "создана" in info


# --- outline / big file / sections ----------------------------------------

def test_outline_skips_code_fences() -> None:
    text = "# A\n```\n# not heading\n```\n## B\n"
    assert outline(text) == [(1, "# A"), (5, "## B")]


def test_big_file_view_has_toc_with_line_numbers() -> None:
    text = PAGE + "\n".join(f"строка {i}" for i in range(3000))
    view = big_file_view("wiki/projects/P.md", text, 500)
    assert "БОЛЬШОЙ ФАЙЛ" in view and "ОГЛАВЛЕНИЕ" in view
    assert "    9: ## Задачи" in view
    assert len(view) < 2000


def test_sections_pinned_block() -> None:
    out = sections(PAGE, ["Цель", "Решения"], 1000)
    assert "Сделать робота" in out and "колёса BLDC" in out and "раз" not in out
    assert "обрезано" in sections(PAGE, ["Цель", "Решения"], 20)


# --- links / similar pages ------------------------------------------------

def test_broken_links(tmp_path: Path) -> None:
    (tmp_path / "wiki" / "projects").mkdir(parents=True)
    (tmp_path / "wiki" / "projects" / "BiBa.md").write_text("x", encoding="utf-8")
    (tmp_path / "raw" / "imports").mkdir(parents=True)
    (tmp_path / "raw" / "imports" / "yt_1.md").write_text("x", encoding="utf-8")
    text = ("[[BiBa]] [[BiBa|алиас]] [[raw/imports/yt_1]] [[wiki/projects/BiBa]] [[Нет-такой]] "
            "`[[в-коде]]` [[YYYY-MM-DD-slug]] [[raw/daily/2025-01-01#msg-1]]")
    assert broken_links(text, tmp_path) == ["Нет-такой", "raw/daily/2025-01-01"]


def test_similar_pages(tmp_path: Path) -> None:
    d = tmp_path / "wiki" / "research"
    d.mkdir(parents=True)
    (d / "2026-10-15-poisk-postavshchikov-v-kitae.md").write_text("x", encoding="utf-8")
    (d / "2026-10-14-pogoda-bataysk.md").write_text("x", encoding="utf-8")
    sim = similar_pages(tmp_path, "wiki/research/2026-10-16-poisk-postavshchikov-cn.md")
    assert sim == ["wiki/research/2026-10-15-poisk-postavshchikov-v-kitae.md"]


# --- ToolExecutor integration --------------------------------------------

def _exec(root: Path) -> ToolExecutor:
    async def ask(_q, _o):
        return ""
    return ToolExecutor(Wiki(root), ask)


@pytest.mark.asyncio
async def test_read_big_file_then_append_to_section(tmp_path: Path) -> None:
    (tmp_path / "wiki" / "projects").mkdir(parents=True)
    big = PAGE + "\n## Заметки\n" + "\n".join(f"- заметка {i}" for i in range(2000)) + "\n"
    (tmp_path / "wiki" / "projects" / "P.md").write_text(big, encoding="utf-8")
    ex = _exec(tmp_path)
    view = await ex.call("read_file", {"path": "wiki/projects/P.md"})
    assert "ОГЛАВЛЕНИЕ" in view and len(view) <= ex.READ_FULL_MAX + 500
    res = await ex.call("append_to_section",
                        {"path": "wiki/projects/P.md", "heading": "Задачи", "content": "- [ ] три"})
    assert res.startswith("OK") and "битые" not in res
    disk = (tmp_path / "wiki" / "projects" / "P.md").read_text(encoding="utf-8")
    assert "- [ ] два\n- [ ] три" in disk and disk.endswith("- заметка 1999\n")  # хвост цел


@pytest.mark.asyncio
async def test_edit_reads_disk_not_stale_cache(tmp_path: Path) -> None:
    (tmp_path / "wiki" / "notes").mkdir(parents=True)
    f = tmp_path / "wiki" / "notes" / "n.md"
    f.write_bytes("a\nb\n".encode())
    ex = _exec(tmp_path)
    await ex.call("read_file", {"path": "wiki/notes/n.md"})
    f.write_bytes("a\nb\nc от другого прогона\n".encode())
    res = await ex.call("edit_file", {"path": "wiki/notes/n.md", "old_string": "b", "new_string": "B"})
    assert res.startswith("OK")
    assert f.read_text(encoding="utf-8") == "a\nB\nc от другого прогона\n"


@pytest.mark.asyncio
async def test_write_warns_broken_links_and_similar(tmp_path: Path) -> None:
    (tmp_path / "wiki" / "research").mkdir(parents=True)
    (tmp_path / "wiki" / "research" / "2026-10-15-kargo-iz-kitaya-2026.md").write_text("x", encoding="utf-8")
    ex = _exec(tmp_path)
    res = await ex.call("write_file", {"path": "wiki/research/2026-10-16-kargo-kitaya-2026-podborka.md",
                                       "content": "см. [[Чужой-проект]]"})
    assert res.startswith("OK") and "битые ссылки" in res and "Чужой-проект" in res
    assert "похожие страницы" in res and "kargo-iz-kitaya-2026" in res


@pytest.mark.asyncio
async def test_ask_choice_and_use_skill(tmp_path: Path) -> None:
    asked: list = []

    async def choice(q, opts):
        asked.append((q, opts))
        return "вариант B"
    ex = ToolExecutor(Wiki(tmp_path), lambda q, o: None, choice_fn=choice)  # type: ignore[arg-type]
    res = await ex.call("ask_choice", {"question": "Какой?", "options": ["A", "вариант B"]})
    assert res == "Ответ пользователя: вариант B" and asked == [("Какой?", ["A", "вариант B"])]
    res = await ex.call("use_skill", {"name": "grill"})
    assert res.startswith("[СКИЛ grill") and "ask_choice" in res
    names = {t["function"]["name"] for t in personal_tool_schemas()}
    assert {"ask_choice", "use_skill", "append_to_section"} <= names
    assert "append_to_section" in PERSONAL_WRITE_TOOLS


# --- write guard: упавшая запись — не запись ------------------------------

class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def chat(self, messages, tools=None, **_kw):
        return self._responses.pop(0) if self._responses else {"choices": [{"message": {"content": "?"}}]}


def _call(name: str, args: dict, cid: str) -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": cid, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}}]}


@pytest.mark.asyncio
async def test_failed_edit_does_not_count_as_write(tmp_path: Path) -> None:
    (tmp_path / "wiki" / "notes").mkdir(parents=True)
    (tmp_path / "wiki" / "notes" / "n.md").write_text("a\n", encoding="utf-8")
    client = _FakeClient([
        _call("edit_file", {"path": "wiki/notes/n.md", "old_string": "zzz", "new_string": "y"}, "c1"),
        _call("finish", {"summary": "Записал цель в wiki/notes/n.md"}, "c2"),
        _call("finish", {"summary": "Записал цель в wiki/notes/n.md"}, "c3"),
    ])
    run = await run_agent(client=client, executor=_exec(tmp_path), system_prompt="s",
                          user_prompt="u", write_tools=PERSONAL_WRITE_TOOLS)
    assert run.tools_called == ["edit_file"] and run.tools_ok == []
    assert run.write_nudged and run.unconfirmed_write and run.writes == {}


@pytest.mark.asyncio
async def test_successful_writes_footer(tmp_path: Path) -> None:
    client = _FakeClient([
        _call("write_file", {"path": "wiki/notes/a.md", "content": "x"}, "c1"),
        _call("finish", {"summary": "Готово"}, "c2"),
    ])
    run = await run_agent(client=client, executor=_exec(tmp_path), system_prompt="s",
                          user_prompt="u", write_tools=PERSONAL_WRITE_TOOLS)
    assert run.writes == {"wiki/notes/a.md": "write_file"}
    footer = writes_footer(run.writes)
    assert footer == "✍️ Изменено: `wiki/notes/a.md` (создан/перезаписан)"
    assert add_footer("Ок\n\nIMAGES:\nraw/a.png", footer) == f"Ок\n\n{footer}\n\nIMAGES:\nraw/a.png"


# --- skills ---------------------------------------------------------------

def test_load_skills_builtin_and_vault_override(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "grill.md").write_text(
        "---\nname: grill\ndescription: мой грилл\n---\nтело", encoding="utf-8")
    (tmp_path / "skills" / "Bad Name.md").write_text("тело", encoding="utf-8")
    sk = load_skills(tmp_path)
    assert {"grill", "tidy", "interview"} <= set(sk)
    assert sk["grill"].description == "мой грилл" and sk["grill"].source == "skills/grill.md"
    assert "bad name" not in sk


def test_match_command() -> None:
    sk = load_skills(None)
    m = match_command("/grill план запуска", sk)
    assert m and m[0].name == "grill" and m[1] == "план запуска"
    m = match_command("/skill tidy", sk)
    assert m and m[0].name == "tidy" and m[1] == ""
    assert match_command("/start", sk) is None and match_command("grill", sk) is None


# --- progress plaque ------------------------------------------------------

def test_render_progress() -> None:
    meta = {"t0": 100.0, "phase": "thinking", "phase_t0": 150.0, "step": 3, "max_steps": 40,
            "ctx_chars": 48_200, "tokens": 120_000, "tokens_cached": 90_000, "note": "проверю статус"}
    text = render_progress(["📖 читаю: wiki/a.md", "✍️ пишу: wiki/b.md"], meta, now=183.0)
    first, *rest = text.splitlines()
    assert first == "⏳ 1:23 · шаг 3/40 · контекст 48k симв · токены 120k (кэш 90k)"
    assert rest[0] == "🧠 думаю… 33с" and rest[1] == "💭 проверю статус"
    assert rest[-1] == "✍️ пишу: wiki/b.md"


# --- остановка хода -------------------------------------------------------

@pytest.mark.asyncio
async def test_request_stop_cancels_only_for_allowed_user() -> None:
    import asyncio

    from bot.handlers.private import RunningTurn, _running_turns, request_stop, stop_turns_in
    started = asyncio.Event()

    async def long_turn() -> str:
        started.set()
        await asyncio.sleep(30)
        return "done"
    task = asyncio.create_task(long_turn())
    await started.wait()
    _running_turns["t1"] = RunningTurn(task, chat_id=10, thread_id=5, allowed={42, None})
    try:
        assert request_stop("t1", 7) == "forbidden" and not task.cancelled()
        assert stop_turns_in(10, 6, 42) == 0          # другая тема
        assert stop_turns_in(10, 5, 42) == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _running_turns["t1"].stopped
        assert request_stop("t1", 42) == "gone"
        assert request_stop("nope", 42) == "gone"
    finally:
        _running_turns.pop("t1", None)


def test_busy_count_counts_running_queued_and_ingest(monkeypatch) -> None:
    from bot.handlers import private

    class _T:
        def __init__(self, done: bool) -> None:
            self._d = done

        def done(self) -> bool:
            return self._d
    monkeypatch.setattr(private, "_running_turns", {"a": private.RunningTurn(_T(False), 1, None, {1}),
                                                    "b": private.RunningTurn(_T(True), 1, None, {1})})
    monkeypatch.setattr(private, "_queue_depth", {1: 2})
    monkeypatch.setattr(private, "_ingest_tasks", {1: _T(False)})
    assert private.busy_count() == 4


def test_free_text_option_detection() -> None:
    from bot.handlers.listener import is_free_text_option as f
    for yes in ("Другое — напишу", "Опишу словами сам", "Свой вариант", "другое", "✍️ напишу сам", "Other"):
        assert f(yes), yes
    for no in ("Другой поставщик — Yongkang", "Да, рама общая", "Не начинали вообще — ни названия, ни решения",
               "Пока не сверяли, надо посмотреть CAD BiBa3"):
        assert not f(no), no


@pytest.mark.asyncio
async def test_interjection_reaches_running_agent(tmp_path: Path) -> None:
    inbox = ["не понял сути вопроса"]
    seen: list = []

    class _C:
        def __init__(self):
            self.n = 0

        async def chat(self, messages, tools=None, **_kw):
            seen.append([m.get("content", "") for m in messages])
            self.n += 1
            if self.n == 1:
                return _call("list_dir", {"path": "wiki"}, "c1")
            return _call("finish", {"summary": "ок"}, "c2")

    def take():
        out = list(inbox)
        inbox.clear()
        return out
    run = await run_agent(client=_C(), executor=_exec(tmp_path), system_prompt="s", user_prompt="u",
                          interject=take)
    assert run.summary == "ок"
    assert any("не понял сути вопроса" in c for c in seen[0])      # до первого шага
    assert sum("не понял сути вопроса" in c for c in seen[1]) == 1   # не дублируется
