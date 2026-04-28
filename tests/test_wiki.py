"""Тесты для bot/wiki.py — главное path traversal guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from bot.wiki import Wiki, WikiPathError


@pytest.fixture
def wiki(tmp_path: Path) -> Wiki:
    (tmp_path / "wiki" / "entities").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "raw" / "daily").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("schema")
    (tmp_path / "index.md").write_text("idx")
    (tmp_path / "log.md").write_text("log")
    return Wiki(tmp_path)


def test_read_existing(wiki: Wiki) -> None:
    assert wiki.read_file("AGENTS.md") == "schema"


def test_path_traversal_rejected(wiki: Wiki) -> None:
    with pytest.raises(WikiPathError):
        wiki.read_file("../etc/passwd")
    with pytest.raises(WikiPathError):
        wiki.write_file("../evil.md", "x")


def test_absolute_path_normalised(wiki: Wiki) -> None:
    # leading slash ok — strip и резолв внутри vault
    wiki.write_file("/wiki/entities/x.md", "hi")
    assert wiki.read_file("wiki/entities/x.md") == "hi"


def test_write_outside_writable_area(wiki: Wiki) -> None:
    with pytest.raises(WikiPathError):
        wiki.write_file("AGENTS.md", "tampering")
    # но index.md и log.md разрешены
    wiki.write_file("index.md", "ok")
    wiki.append_file("log.md", "line")


def test_write_in_wiki_ok(wiki: Wiki) -> None:
    rel = wiki.write_file("wiki/entities/Foo.md", "# Foo")
    assert rel == "wiki/entities/Foo.md"
    assert wiki.read_file(rel).startswith("# Foo")


def test_search(wiki: Wiki) -> None:
    wiki.write_file("wiki/concepts/Bar.md", "# Bar\n\nhello world")
    hits = wiki.search("hello")
    assert any(h["path"] == "wiki/concepts/Bar.md" for h in hits)


def test_daily_raw_format(wiki: Wiki) -> None:
    msgs = [
        {
            "message_id": 42,
            "ts": "2026-04-28T10:00:00+00:00",
            "full_name": "Alice",
            "username": "alice",
            "text": "hello\nworld",
            "reply_to_message_id": None,
        },
        {
            "message_id": 43,
            "ts": "2026-04-28T10:01:00+00:00",
            "full_name": "Bob",
            "username": "bob",
            "text": "reply",
            "reply_to_message_id": 42,
        },
    ]
    rel = wiki.write_daily_raw("2026-04-28", msgs)
    content = wiki.read_file(rel)
    assert "^msg-42" in content
    assert "^msg-43" in content
    assert "(reply → ^msg-42)" in content
    assert "Всего сообщений: **2**" in content
