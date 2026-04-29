from bot.tg_format import md_to_tg_html, split_tg_chunks


def test_escape_specials():
    assert md_to_tg_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_headings():
    assert md_to_tg_html("## Hello") == "<b>Hello</b>"
    assert md_to_tg_html("### Sub title") == "<b>Sub title</b>"


def test_bold_italic_strike():
    assert md_to_tg_html("**bold**") == "<b>bold</b>"
    assert md_to_tg_html("*italic*") == "<i>italic</i>"
    assert md_to_tg_html("_italic_") == "<i>italic</i>"
    assert md_to_tg_html("~~gone~~") == "<s>gone</s>"


def test_inline_code_escapes():
    assert md_to_tg_html("foo `<x>` bar") == "foo <code>&lt;x&gt;</code> bar"


def test_code_fence():
    out = md_to_tg_html("```python\nprint(1 < 2)\n```")
    assert out == "<pre>print(1 &lt; 2)</pre>"


def test_link():
    assert (
        md_to_tg_html("see [docs](https://example.com/?a=1&b=2)")
        == 'see <a href="https://example.com/?a=1&amp;b=2">docs</a>'
    )


def test_wiki_link():
    assert (
        md_to_tg_html("see [[wiki/entities/Techno-Drone#anchor]]")
        == "see <i>Techno-Drone</i>"
    )
    assert md_to_tg_html("[[wiki/x|Алиас]]") == "<i>Алиас</i>"


def test_blockquote():
    out = md_to_tg_html("> цитата\n> вторая\nобычный")
    assert "<blockquote>" in out and "</blockquote>" in out
    assert "цитата" in out


def test_list_and_hr():
    out = md_to_tg_html("- one\n- two\n\n---\n\n- three")
    assert "• one" in out
    assert "• three" in out
    assert "---" not in out


def test_donos_smoke():
    src = (
        "## 📄 ДОНОС\n\n"
        "**Кому:** Комиссии  \n"
        "Цитата:\n> *«Что это за спамер»*\n\n"
        "[[wiki/entities/Techno-Drone#2026-04-28-основные-батчи]]\n"
        "1. пункт `code <x>`\n"
        "---\n"
        "конец"
    )
    out = md_to_tg_html(src)
    assert "<b>📄 ДОНОС</b>" in out
    assert "<blockquote>" in out
    assert "<i>Techno-Drone</i>" in out
    assert "<code>code &lt;x&gt;</code>" in out
    # никаких ## или [[ остаться не должно
    assert "##" not in out
    assert "[[" not in out


def test_split_chunks():
    long = ("abc\n" * 2000)
    parts = split_tg_chunks(long, limit=4096)
    assert all(len(p) <= 4096 for p in parts)
    assert "".join(p + ("\n" if i < len(parts) - 1 else "") for i, p in enumerate(parts)).rstrip("\n") == long.rstrip("\n")


def test_unbalanced_asterisk_safe():
    # одиночные * не должны ломать вывод
    out = md_to_tg_html("a * b * c **bold**")
    assert "<b>bold</b>" in out
    # одиночные звезды остались как есть, не как теги
    assert "<i>" not in out or out.count("<i>") == 0
