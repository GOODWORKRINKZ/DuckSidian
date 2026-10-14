"""Тесты личного режима: PersonalSpace, шаринг, мульти-вендор LLM, personal tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot import personal as personal_mod
from bot.agent.deepseek import _completions_url, extract_usage
from bot.agent.loop import run_agent
from bot.agent.tools import ToolExecutor, personal_tool_schemas
from bot.config import LLM_PROVIDERS, ChatConfig, settings
from bot.personal import PersonalSpace, share_artifact, user_cfg
from bot.wiki import Wiki, WikiPathError


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "vault_path", tmp_path)
    (tmp_path / "AGENTS.md").write_text("# schema", encoding="utf-8")
    return tmp_path


@pytest.fixture
def space(vault: Path) -> PersonalSpace:
    return PersonalSpace(42, "Иван")


# ---------------------------------------------------------------- PersonalSpace

def test_space_seeded(space: PersonalSpace, vault: Path) -> None:
    root = vault / "users" / "42"
    assert (root / "AGENTS.md").exists()
    assert "личное пространство" in (root / "AGENTS.md").read_text(encoding="utf-8")
    assert (root / "index.md").exists() and (root / "log.md").exists()
    for sub in ("raw/inbox", "raw/assets", "wiki/projects", "wiki/notes",
                "wiki/ideas", "wiki/research"):
        assert (root / sub).is_dir()
    assert (root / "wiki/projects/_template.md").exists()
    assert user_cfg(42).name == "users/42"
    assert personal_mod.is_user_cfg(user_cfg(42))
    assert not personal_mod.is_user_cfg(ChatConfig(chat_id=-1, name="main"))


def test_inbox_append(space: PersonalSpace) -> None:
    rel = space.append_inbox(who="Иван", text="мысль раз\nстрока два", message_id=7,
                             media_type="voice", media_path="raw/assets/2026-01-01/v.ogg")
    text = (space.root / rel).read_text(encoding="utf-8")
    assert rel.startswith("raw/inbox/")
    assert "^msg-7" in text
    assert "> мысль раз" in text and "> строка два" in text
    assert "[[raw/assets/2026-01-01/v.ogg]]" in text
    # второй append — файл тот же, заголовок один
    space.append_inbox(who="Иван", text="ещё", message_id=8)
    text = (space.root / rel).read_text(encoding="utf-8")
    assert text.count("# Inbox") == 1 and "^msg-8" in text


def test_artifacts_list_resolve_delete(space: PersonalSpace) -> None:
    space.wiki.write_file("wiki/notes/2026-01-02-alpha.md",
                          "---\ntype: note\ntitle: \"Альфа\"\nupdated: 2026-01-02\n---\n# Альфа\n")
    space.wiki.write_file("wiki/ideas/2026-01-03-beta.md", "# Бета идея\n")
    (space.root / "raw/assets/2026-01-01").mkdir(parents=True)
    (space.root / "raw/assets/2026-01-01/pic.jpg").write_bytes(b"\xff\xd8")
    arts = space.list_artifacts()
    paths = {a.path for a in arts}
    assert paths == {"wiki/notes/2026-01-02-alpha.md", "wiki/ideas/2026-01-03-beta.md",
                     "raw/assets/2026-01-01/pic.jpg"}
    assert not any(a.path.endswith("_template.md") for a in arts)
    kinds = {a.path: a.kind for a in arts}
    assert kinds["wiki/notes/2026-01-02-alpha.md"] == "notes"
    assert kinds["raw/assets/2026-01-01/pic.jpg"] == "files"
    titles = {a.path: a.title for a in arts}
    assert titles["wiki/notes/2026-01-02-alpha.md"] == "Альфа"
    assert titles["wiki/ideas/2026-01-03-beta.md"] == "Бета идея"
    # фильтр по виду
    assert [a.path for a in space.list_artifacts(kind="ideas")] == ["wiki/ideas/2026-01-03-beta.md"]
    # resolve: точный путь, basename без .md, подстрока, картинка
    assert space.resolve_artifact("wiki/notes/2026-01-02-alpha.md") == "wiki/notes/2026-01-02-alpha.md"
    assert space.resolve_artifact("2026-01-02-alpha") == "wiki/notes/2026-01-02-alpha.md"
    assert space.resolve_artifact("бета") == "wiki/ideas/2026-01-03-beta.md"
    assert space.resolve_artifact("pic.jpg") == "raw/assets/2026-01-01/pic.jpg"
    assert space.resolve_artifact("nope") is None
    # delete: wiki ок, AGENTS.md / шаблон / raw/inbox — нельзя
    assert space.delete_artifact("wiki/ideas/2026-01-03-beta.md") == "wiki/ideas/2026-01-03-beta.md"
    assert not (space.root / "wiki/ideas/2026-01-03-beta.md").exists()
    with pytest.raises(WikiPathError):
        space.delete_artifact("AGENTS.md")
    with pytest.raises(WikiPathError):
        space.delete_artifact("wiki/projects/_template.md")
    space.append_inbox(who="x", text="y", message_id=1)
    with pytest.raises(WikiPathError):
        space.delete_artifact(next((space.root / "raw/inbox").glob("*.md")).relative_to(space.root).as_posix())


def test_save_research(space: PersonalSpace) -> None:
    rel = space.save_research("Цены на лидары", "## Вывод\nдорого", query="сколько стоит лидар?")
    text = (space.root / rel).read_text(encoding="utf-8")
    assert rel.startswith("wiki/research/") and rel.endswith("-цены-на-лидары.md")
    assert "type: research" in text and "# Цены на лидары" in text and "дорого" in text


# ---------------------------------------------------------------- sharing

def test_share_markdown_with_assets(space: PersonalSpace, vault: Path) -> None:
    target = ChatConfig(chat_id=-100500, name="main")
    (space.root / "raw/assets/2026-01-01").mkdir(parents=True)
    (space.root / "raw/assets/2026-01-01/scheme.png").write_bytes(b"PNG")
    space.wiki.write_file(
        "wiki/research/2026-01-05-lidar.md",
        "---\ntype: research\ntitle: \"Лидары\"\n---\n# Лидары\n\nСм. ![[raw/assets/2026-01-01/scheme.png]] и факт.\n",
    )
    res = share_artifact(space, "wiki/research/2026-01-05-lidar.md", target,
                         author="Иван", comment="глянь")
    troot = vault / "main"
    assert res.title == "Лидары" and not res.is_binary
    assert res.dst_note.startswith("raw/notes/") and "shared-лидары" in res.dst_note
    note = (troot / res.dst_note).read_text(encoding="utf-8")
    assert "type: shared" in note and 'from: "Иван"' in note
    assert "source: \"users/42/wiki/research/2026-01-05-lidar.md\"" in note
    assert "> 💬 глянь" in note
    # вложение скопировано и ссылка переписана
    assert len(res.copied_assets) == 1
    copied = res.copied_assets[0]
    assert copied.startswith("raw/assets/shared/") and copied.endswith("u42_scheme.png")
    assert (troot / copied).read_bytes() == b"PNG"
    assert f"![[{copied}]]" in note
    assert "raw/assets/2026-01-01/scheme.png" not in note
    # лог личного пространства
    assert "share |" in (space.root / "log.md").read_text(encoding="utf-8")


def test_share_binary(space: PersonalSpace, vault: Path) -> None:
    target = ChatConfig(chat_id=-100500, name="main")
    (space.root / "raw/assets/2026-01-01").mkdir(parents=True)
    (space.root / "raw/assets/2026-01-01/model.step").write_bytes(b"STEP")
    res = share_artifact(space, "raw/assets/2026-01-01/model.step", target, author="Иван")
    assert res.is_binary and res.title == "model.step"
    assert (vault / "main" / res.copied_assets[0]).read_bytes() == b"STEP"
    note = (vault / "main" / res.dst_note).read_text(encoding="utf-8")
    assert f"[[{res.copied_assets[0]}]]" in note


def test_share_missing(space: PersonalSpace) -> None:
    with pytest.raises(WikiPathError):
        share_artifact(space, "wiki/notes/nope.md", ChatConfig(chat_id=-1, name="main"), author="x")


# ---------------------------------------------------------------- personal tools via executor

@pytest.mark.asyncio
async def test_executor_team_tools_and_share(space: PersonalSpace, vault: Path) -> None:
    team_root = vault / "main"
    (team_root / "wiki").mkdir(parents=True)
    (team_root / "wiki/X.md").write_text("# X\n\nпро лидар Livox\n", encoding="utf-8")
    (team_root / "index.md").write_text("idx\n", encoding="utf-8")
    team = Wiki(team_root)
    shared: list[tuple[str, str, str]] = []

    async def share_fn(path: str, chat: str, comment: str) -> str:
        shared.append((path, chat, comment))
        return "OK: shared"

    async def ask(_q, _o):
        return ""

    ex = ToolExecutor(space.wiki, ask, team_wikis={"main": team}, share_fn=share_fn)
    hits = json.loads(await ex.call("search_team_wiki", {"chat": "main", "query": "livox"}))
    assert hits and hits[0]["path"] == "wiki/X.md" and hits[0]["chat"] == "main"
    out = await ex.call("read_team_file", {"chat": "main", "path": "wiki/X.md"})
    assert "Livox" in out and out.startswith("[main:wiki/X.md]")
    out = await ex.call("read_team_file", {"chat": "main", "path": "wiki/X.md", "start": 3, "end": 3})
    assert "строки 3–3" in out and "Livox" in out
    assert "wiki/" in await ex.call("list_team_dir", {"chat": "main"})
    assert "ERROR" in await ex.call("search_team_wiki", {"chat": "sales", "query": "x"})
    assert await ex.call("share_to_chat", {"path": "wiki/a.md", "chat": "main", "comment": "c"}) == "OK: shared"
    assert shared == [("wiki/a.md", "main", "c")]
    # без share_fn — ошибка, не исключение
    ex2 = ToolExecutor(space.wiki, ask)
    assert "ERROR" in await ex2.call("share_to_chat", {"path": "a", "chat": "main"})
    # командная wiki read-only: write в неё через executor невозможен (нет такого tool)
    names = {t["function"]["name"] for t in personal_tool_schemas()}
    assert {"search_team_wiki", "read_team_file", "list_team_dir", "share_to_chat"} <= names
    assert not {"ask_user", "defer_question", "tag_media"} & names


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append([dict(m) for m in messages])
        self.last_tools = tools
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_run_agent_history_and_tools(space: PersonalSpace) -> None:
    async def ask(_q, _o):
        return ""
    ex = ToolExecutor(space.wiki, ask)
    client = _FakeClient([{"choices": [{"message": {"content": "готово"}}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                     "prompt_tokens_details": {"cached_tokens": 4}}}])
    run = await run_agent(
        client=client, executor=ex, system_prompt="sys", user_prompt="новое",
        history=[{"role": "user", "content": "раньше"}, {"role": "assistant", "content": "ответ"},
                 {"role": "tool", "content": "skip"}],
        tools=personal_tool_schemas(), max_steps=3,
    )
    assert run.finished and run.summary == "готово"
    roles = [m["role"] for m in client.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.calls[0][-1]["content"] == "новое"
    assert client.last_tools is not None and len(client.last_tools) == len(personal_tool_schemas())
    assert (run.tokens_prompt, run.tokens_completion, run.tokens_cached) == (10, 2, 4)


# ---------------------------------------------------------------- multi-vendor LLM

def test_completions_url() -> None:
    assert _completions_url("https://api.deepseek.com") == "https://api.deepseek.com/v1/chat/completions"
    assert _completions_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1/chat/completions"
    assert _completions_url("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/") == \
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert _completions_url("http://localhost:11434") == "http://localhost:11434/v1/chat/completions"


def test_extract_usage_variants() -> None:
    assert extract_usage({"usage": {"prompt_tokens": 5, "completion_tokens": 1,
                                    "prompt_cache_hit_tokens": 3}}) == (5, 1, 3)
    assert extract_usage({"usage": {"prompt_tokens": 5, "completion_tokens": 1,
                                    "prompt_tokens_details": {"cached_tokens": 2}}}) == (5, 1, 2)
    assert extract_usage({}) == (0, 0, 0)


def test_resolve_llm_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "dsk")
    monkeypatch.setattr(settings, "minimax_api_key", "mmk")
    monkeypatch.setattr(settings, "dashscope_api_key", "")
    monkeypatch.setattr(settings, "llm_model", "")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "deepseek_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "deepseek_model", "deepseek-chat")
    prov, key, url, model = settings.resolve_llm()
    assert (prov.key, key, url, model) == ("deepseek", "dsk", "https://api.deepseek.com", "deepseek-chat")
    prov, key, url, model = settings.resolve_llm("minimax")
    assert (prov.key, key, model) == ("minimax", "mmk", LLM_PROVIDERS["minimax"].default_model)
    assert url == "https://api.minimax.io/v1"
    prov, key, url, model = settings.resolve_llm("qwen", "qwen-max")
    assert (prov.key, key, model) == ("qwen", "", "qwen-max")
    with pytest.raises(KeyError):
        settings.resolve_llm("nope")
    # LLM_* переопределения действуют только на активного провайдера
    monkeypatch.setattr(settings, "llm_provider", "qwen")
    monkeypatch.setattr(settings, "llm_api_key", "override")
    monkeypatch.setattr(settings, "llm_model", "qwen3-coder-plus")
    prov, key, url, model = settings.resolve_llm()
    assert (prov.key, key, model) == ("qwen", "override", "qwen3-coder-plus")
    prov, key, _, _ = settings.resolve_llm("minimax")
    assert key == "mmk"


def test_llm_client_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.agent.deepseek import LLMClient
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "dsk")
    monkeypatch.setattr(settings, "minimax_api_key", "")
    monkeypatch.setattr(settings, "dashscope_api_key", "q")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model", "")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "deepseek_base_url", "")
    monkeypatch.setattr(settings, "deepseek_model", "")
    c = LLMClient()
    assert c.provider.key == "deepseek" and c.model == "deepseek-v4-flash"
    desc = c.switch("qwen", "qwen-max")
    assert c.provider.key == "qwen" and c.model == "qwen-max" and "Qwen" in desc
    assert not c.provider.supports_thinking_flag
    with pytest.raises(ValueError):
        c.switch("minimax")  # нет ключа
    with pytest.raises(ValueError):
        c.switch("unknown")
    assert c.cost_usd(1_000_000, 0, 0) == pytest.approx(LLM_PROVIDERS["qwen"].price_in)
    assert any(p["key"] == "minimax" and not p["has_key"] for p in c.available_providers())


def test_private_users_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.config import Settings, parse_id_list
    assert parse_id_list("1, 2;3") == [1, 2, 3]
    assert parse_id_list("") == []
    assert parse_id_list(7) == [7]
    assert parse_id_list("[1, 2]") == [1, 2]
    # Через окружение: список через запятую не должен падать на json.loads.
    monkeypatch.setenv("TELEGRAM_ADMINS", "6124427627,495039871")
    monkeypatch.setenv("TELEGRAM_PRIVATE_USERS", "5")
    st = Settings(_env_file=None)  # type: ignore[call-arg]
    assert st.telegram_admins == [6124427627, 495039871]
    assert st.telegram_private_users == [5]


# ---------------------------------------------------------------- media_ai helpers / workspaces

def test_md_for_tts_and_strip_think() -> None:
    from bot.media_ai import _md_for_tts, _strip_think
    src = ("**Жирный** и `код`, ссылка [тут](https://x.y/z) и [[Slug|алиас]] ![[raw/assets/a.jpg]]\n\n"
           "```py\nprint(1)\n```\nhttps://example.com\n\nIMAGES:\nraw/assets/x.jpg")
    out = _md_for_tts(src)
    assert "Жирный" in out and "код" in out and "тут" in out and "алиас" in out
    assert "https://" not in out and "print" not in out and "IMAGES" not in out and "[[" not in out
    assert _strip_think("<think>a</think>Ответ") == "Ответ"


def test_project_slug_and_ensure_project(space: PersonalSpace) -> None:
    from bot.personal import project_slug
    assert project_slug("Роббокс v2 / тесты!") == "Роббокс-v2-тесты"
    assert project_slug("") == "Project"
    slug = space.ensure_project("Доставочный робот")
    assert slug == "Доставочный-робот"
    text = (space.root / f"wiki/projects/{slug}.md").read_text(encoding="utf-8")
    assert "# Доставочный робот" in text and "type: project" in text and "{{" not in text
    assert f"[[{slug}]]" in (space.root / "index.md").read_text(encoding="utf-8")
    assert space.ensure_project("Доставочный робот") == slug  # идемпотентно


@pytest.mark.asyncio
async def test_db_workspaces_and_topic_history(tmp_path: Path) -> None:
    from datetime import datetime, timezone
    from bot.db import DB
    db = DB(tmp_path / "t.sqlite3")
    await db.init()
    await db.upsert_workspace(42, 7, "Роббокс", "Роббокс")
    await db.upsert_workspace(42, 7, "Роббокс 2", None)       # rename, slug сохраняется
    ws = await db.get_workspace(42, 7)
    assert ws["name"] == "Роббокс 2" and ws["project_slug"] == "Роббокс"
    assert len(await db.list_workspaces(42)) == 1
    now = datetime.now(timezone.utc)
    for mid, topic, text in ((1, 7, "в теме"), (2, None, "общее"), (3, 7, "ещё в теме")):
        await db.save_message(chat_id=42, message_id=mid, topic_id=topic, user_id=1, username=None,
                              full_name="x", text=text, reply_to_message_id=None, ts=now)
    assert [r["text"] for r in await db.fetch_last_messages_in_topic(42, 7, 10)] == ["в теме", "ещё в теме"]
    assert [r["text"] for r in await db.fetch_last_messages_in_topic(42, None, 10)] == ["общее"]
    await db.delete_workspace(42, 7)
    assert await db.get_workspace(42, 7) is None
    assert await db.add_private_user(5) and not await db.add_private_user(5)
    assert await db.get_private_users() == [5]
    assert await db.remove_private_user(5)


def test_share_workspace_bundle(space: PersonalSpace, vault: Path) -> None:
    from bot.personal import collect_workspace_pages, share_workspace
    target = ChatConfig(chat_id=-100500, name="main")
    slug = space.ensure_project("Роббокс")
    (space.root / "raw/assets/2026-01-01").mkdir(parents=True)
    (space.root / "raw/assets/2026-01-01/d.png").write_bytes(b"PNG")
    space.wiki.write_file("wiki/notes/2026-01-02-n.md",
                          "---\ntype: note\ntitle: \"Н\"\nproject: \"[[Роббокс]]\"\n---\n# Н\n![[raw/assets/2026-01-01/d.png]]\n")
    space.wiki.write_file("wiki/ideas/2026-01-03-i.md", "# Идея\nсм. [[Роббокс|проект]]\n")
    space.wiki.write_file("wiki/notes/2026-01-04-other.md", "# Другое\nне связано\n")
    pages = collect_workspace_pages(space, slug)
    assert pages[0] == f"wiki/projects/{slug}.md"
    assert set(pages) == {f"wiki/projects/{slug}.md", "wiki/notes/2026-01-02-n.md", "wiki/ideas/2026-01-03-i.md"}
    res = share_workspace(space, slug, target, author="Иван", comment="всё по роббоксу")
    assert res.title == "Роббокс" and len(res.pages) == 3 and len(res.copied_assets) == 1
    assert all((vault / "main" / p.dst_note).exists() for p in res.pages)
    assert "всё по роббоксу" in res.summary_md and "Страниц: 3" in res.summary_md
    with pytest.raises(WikiPathError):
        share_workspace(space, "Нет-такого", target, author="x")


def test_step_line() -> None:
    from bot.handlers.private import _step_line
    assert _step_line("search_team_wiki", {"chat": "main", "query": "biba"}) == "🔎 ищу в wiki команды: main: biba"
    assert _step_line("web_search", {"query": "x" * 100}).endswith("x" * 60)
    assert _step_line("unknown_tool", {}) == "⚙️ unknown_tool"
