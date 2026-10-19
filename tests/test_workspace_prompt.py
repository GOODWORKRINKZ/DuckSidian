"""Инструкция воркспейса (мастер-промпт): секция страницы проекта, инструмент агента,
блок в промпте хода; пачка сообщений подряд; очередь и /stop; голос по автору в общем vault."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot.agent.tools import PERSONAL_WRITE_TOOLS, personal_tool_schemas
from bot.agent.wikitools import instruction_body, replace_section, sections
from bot.config import settings
from bot.forwards import BATCH_MARK, BUNDLE_MARK, FORWARD_MARK, bundle_text, is_forward
from bot.handlers import private as pm
from bot.orchestrator import Orchestrator
from bot.personal import PersonalSpace
from bot.skills import load_skills

PAGE = """---
type: project
title: "Учим китайский"
---

# Учим китайский

_Одна строка._

## Инструкция
_Мастер-промпт воркспейса: … (set_workspace_instruction). Посмотреть/задать вручную: /ws prompt._

## Цель
Выучить HSK2.

## Текущий статус
Урок 3.
"""


# ---------------------------------------------------------------- wikitools

def test_replace_section_replaces_body_and_keeps_rest() -> None:
    out, info = replace_section(PAGE, "Инструкция", "- пиньинь только в VOICE\n- отвечай коротко")
    assert "заменена" in info
    assert "- пиньинь только в VOICE\n- отвечай коротко\n\n## Цель\nВыучить HSK2." in out
    assert "Мастер-промпт" not in out and "## Текущий статус\nУрок 3." in out


def test_replace_section_creates_after_h1_intro() -> None:
    page = "# Проект\n\n_Одна строка._\n\n## Цель\nстарое\n"
    out, info = replace_section(page, "Инструкция", "делай так")
    assert "создана" in info
    assert out.index("## Инструкция\nделай так") < out.index("## Цель")
    assert out.index("_Одна строка._") < out.index("## Инструкция")


def test_instruction_body_drops_template_placeholder() -> None:
    assert instruction_body(sections(PAGE, ["Инструкция"], 5000)) == ""
    page = PAGE.replace("_Мастер-промпт воркспейса: … (set_workspace_instruction). Посмотреть/задать вручную: /ws prompt._",
                        "- на каждое слово — карточка\n- VOICE: только китайская фраза")
    assert instruction_body(sections(page, ["Инструкция"], 5000)) == \
        "- на каждое слово — карточка\n- VOICE: только китайская фраза"


# ---------------------------------------------------------------- инструмент и промпт

def test_set_workspace_instruction_is_a_personal_write_tool() -> None:
    names = {t["function"]["name"] for t in personal_tool_schemas()}
    assert "set_workspace_instruction" in names
    assert "set_workspace_instruction" in PERSONAL_WRITE_TOOLS


def test_builtin_skills_include_workspace_setup_and_make_skill() -> None:
    sk = load_skills()
    assert {"workspace-setup", "make-skill"} <= set(sk)
    assert "set_workspace_instruction" in sk["workspace-setup"].body
    assert "skills/<name>.md" in sk["make-skill"].body


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    return tmp_path


def test_project_instruction_from_page(vault: Path) -> None:
    space = PersonalSpace(7, "Денис")
    slug = space.ensure_project("Учим китайский")
    page = (space.root / f"wiki/projects/{slug}.md").read_text(encoding="utf-8")
    assert "## Инструкция" in page                  # шаблон завёл секцию
    assert Orchestrator._project_instruction(space, slug) == ""   # подсказка курсивом — не инструкция
    updated, _ = replace_section(page, "Инструкция", "- отвечай по-китайски")
    (space.root / f"wiki/projects/{slug}.md").write_text(updated, encoding="utf-8")
    assert Orchestrator._project_instruction(space, slug) == "- отвечай по-китайски"
    # Закреплённый блок инструкцию не дублирует
    assert "отвечай по-китайски" not in Orchestrator._project_pinned(space, slug)


# ---------------------------------------------------------------- пачка сообщений

def test_bundle_text_batch_vs_forward_bundle() -> None:
    batch = bundle_text(["вот мои данные, разбери", "[📄 ДОКУМЕНТ: a.pdf]", "[📄 ДОКУМЕНТ: b.pdf]"])
    assert batch.startswith(BATCH_MARK) and "3 сообщений" in batch and "ОДНА" in batch
    assert "[1] вот мои данные" in batch and "[3] [📄 ДОКУМЕНТ: b.pdf]" in batch
    assert not is_forward(batch)
    fwd = bundle_text([f"{FORWARD_MARK}, автор — X. Текст репоста ниже:\n\nпривет", "ещё"])
    assert fwd.startswith(BUNDLE_MARK) and is_forward(fwd)


def test_bundles_detects_albums_files_and_forwards() -> None:
    def m(**kw):
        base = dict(forward_origin=None, forward_from=None, forward_sender_name=None, forward_from_chat=None,
                    media_group_id=None, document=None, photo=None, video=None, animation=None)
        base.update(kw)
        return SimpleNamespace(**base)
    assert pm._bundles(m(media_group_id="123"))
    assert pm._bundles(m(document=object()))
    assert pm._bundles(m(photo=[object()]))
    assert not pm._bundles(m())                 # одиночный текст — без задержки
    assert not pm._bundles(m(voice=object()))   # голосовое — тоже сразу


# ---------------------------------------------------------------- очередь и /stop

@pytest.mark.asyncio
async def test_drop_queued_cancels_only_unstarted_in_topic() -> None:
    pm._queued.clear()
    gate = asyncio.Event()

    async def wait() -> None:
        await gate.wait()

    t1, t2, t3 = (asyncio.create_task(wait()) for _ in range(3))
    q1 = pm.QueuedEntry(t1, 10, 5, 42)         # ждёт в теме 5
    q2 = pm.QueuedEntry(t2, 10, 5, 42)
    q2.started = True                           # уже выполняется — /stop очереди его не трогает
    q3 = pm.QueuedEntry(t3, 10, 9, 42)          # другая тема
    pm._queued.update({q1, q2, q3})
    assert pm.queued_count(10) == 2
    assert pm.drop_queued(10, 5, 42) == 1
    await asyncio.sleep(0)
    assert q1.dropped and t1.cancelled() and not q3.dropped and not t2.cancelled()
    assert pm.drop_queued(10, None, 42, any_thread=True) == 1   # /stop all — остальные темы
    await asyncio.sleep(0)
    assert t3.cancelled()
    gate.set()
    await asyncio.gather(t2, return_exceptions=True)
    pm._queued.clear()


@pytest.mark.asyncio
async def test_drop_queued_respects_author() -> None:
    pm._queued.clear()
    t = asyncio.create_task(asyncio.sleep(10))
    pm._queued.add(pm.QueuedEntry(t, 10, 5, 42))
    assert pm.drop_queued(10, 5, 99) == 0        # чужой запрос в общем воркспейсе
    assert pm.drop_queued(10, 5, 42) == 1
    await asyncio.gather(t, return_exceptions=True)
    pm._queued.clear()


# ---------------------------------------------------------------- голос по автору

def test_voice_owner_is_author_in_shared_vault(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # _voice_owner живёт внутри setup(); воспроизводим его логику через PersonalSpace.shared.
    shared = PersonalSpace.shared("semya", "Денис")
    assert shared.user_id == 0 and shared.is_shared
    msg = SimpleNamespace(from_user=SimpleNamespace(id=495039871, is_bot=False))
    # Правило: в общем vault настройки голоса — автора реплики, не user_id=0.
    owner = msg.from_user.id if shared.is_shared and msg.from_user and not msg.from_user.is_bot else shared.user_id
    assert owner == 495039871
    personal = PersonalSpace(42, "Иван")
    owner = msg.from_user.id if personal.is_shared and msg.from_user and not msg.from_user.is_bot else personal.user_id
    assert owner == 42
