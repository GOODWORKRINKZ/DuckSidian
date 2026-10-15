"""Разбор блока IMAGES: терпимость к оформлению и проверка файлов в vault перед отправкой."""
from __future__ import annotations

from pathlib import Path

import pytest

from bot.handlers.commands import _extract_image_refs, _image_refs
from bot.handlers.private import _attach_media, _strip_images_block
from bot.media_ai import _md_for_tts
from bot.media_refs import missing_note, parse_images_block, strip_images_block
from bot.orchestrator import UNCONFIRMED_WRITE_NOTE, mark_unconfirmed_write


@pytest.mark.parametrize("marker", [
    "IMAGES:", "**IMAGES:**", "**IMAGES**:", "IMAGES :", "## IMAGES:", "__IMAGES:__", "images:",
])
def test_marker_variants(marker: str) -> None:
    answer = f"Вот робот.\n\n{marker}\nraw/assets/a.jpg"
    assert strip_images_block(answer) == ("Вот робот.", True)
    assert parse_images_block(answer) == ["raw/assets/a.jpg"]


def test_marker_not_inside_sentence() -> None:
    answer = "Блок IMAGES: добавлю позже.\nraw/assets/a.jpg"
    assert strip_images_block(answer) == (answer, False)
    assert parse_images_block(answer) == []


def test_path_variants() -> None:
    answer = (
        "Текст\n\n**IMAGES:**\n"
        "`raw/assets/2026-10-15/web_a.jpg`\n"
        "- `raw/assets/2026-10-15/web_b.png`\n"
        "* [[raw/assets/2026-10-15/web_c.webp]]\n"
        "![[raw/assets/2026-10-15/web_d.gif|alias]]\n"
        "[[raw/assets/2026-10-15/фото робота.jpg]]\n"
        "raw/assets/2026-10-15/video.mp4 — ролик\n"
        "raw/assets/2026-10-15/doc.pdf\n"
        "/raw/assets/2026-10-15/web_a.jpg\n"
    )
    assert parse_images_block(answer) == [
        "raw/assets/2026-10-15/web_a.jpg",
        "raw/assets/2026-10-15/web_b.png",
        "raw/assets/2026-10-15/web_c.webp",
        "raw/assets/2026-10-15/web_d.gif",
        "raw/assets/2026-10-15/фото робота.jpg",
        "raw/assets/2026-10-15/video.mp4",
    ]


def test_block_ends_at_text() -> None:
    answer = "x\n\nIMAGES:\nraw/assets/a.jpg\n\nP.S. raw/assets/b.jpg\n"
    assert parse_images_block(answer) == ["raw/assets/a.jpg"]


def _touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"img")
    return p


def test_image_refs_checks_vault(tmp_path: Path) -> None:
    real = _touch(tmp_path, "raw/assets/2026-10-01/photo_1.jpg")
    _touch(tmp_path.parent / "outside", "x.jpg")
    answer = (
        "Вот.\n\n**IMAGES:**\n`raw/assets/2026-10-01/photo_1.jpg`\n"
        "`raw/assets/2026-10-15/web_andi-disinfection.jpg`\n../outside/x.jpg\n"
    )
    found, missing = _image_refs(answer, tmp_path)
    assert found == [str(real.resolve())]
    assert missing == ["raw/assets/2026-10-15/web_andi-disinfection.jpg", "../outside/x.jpg"]
    assert _extract_image_refs(answer, tmp_path) == found


def test_missing_note_one_line() -> None:
    note = missing_note(["raw/assets/a.jpg", "raw/assets/b.jpg"])
    assert "\n" not in note and "raw/assets/a.jpg" in note and "raw/assets/b.jpg" in note
    long = missing_note([f"raw/assets/{i}.jpg" for i in range(8)])
    assert "ещё 3" in long and "raw/assets/7.jpg" not in long


class _Msg:
    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    async def answer(self, text: str, **_kw) -> None:
        self.sent.append(("text", text))

    async def answer_photo(self, f, **_kw) -> None:
        self.sent.append(("photo", f.path))

    async def answer_video(self, f, **_kw) -> None:
        self.sent.append(("video", f.path))

    async def answer_animation(self, f, **_kw) -> None:
        self.sent.append(("gif", f.path))


@pytest.mark.asyncio
async def test_attach_media_sends_existing_and_reports_missing(tmp_path: Path) -> None:
    real = _touch(tmp_path, "raw/assets/2026-10-01/photo_1.jpg")
    answer = (
        "Вот.\n\n**IMAGES:**\n- `raw/assets/2026-10-01/photo_1.jpg`\n"
        "- `raw/assets/2026-10-15/web_andi-disinfection.jpg`"
    )
    assert _strip_images_block(answer) == ("Вот.", True)
    msg = _Msg()
    await _attach_media(msg, answer, tmp_path)  # type: ignore[arg-type]
    kinds = [k for k, _ in msg.sent]
    assert kinds == ["text", "photo"]
    assert "web_andi-disinfection.jpg" in str(msg.sent[0][1])
    assert Path(str(msg.sent[1][1])) == real.resolve()


@pytest.mark.asyncio
async def test_attach_media_all_present_no_note(tmp_path: Path) -> None:
    _touch(tmp_path, "raw/assets/a.png")
    msg = _Msg()
    await _attach_media(msg, "x\n\nIMAGES:\nraw/assets/a.png", tmp_path)  # type: ignore[arg-type]
    assert [k for k, _ in msg.sent] == ["photo"]


def test_other_consumers_use_tolerant_marker() -> None:
    answer = "Сделал картинку.\n\n**IMAGES:**\n`raw/assets/a.png`"
    assert "raw/assets" not in _md_for_tts(answer)
    out = mark_unconfirmed_write(answer)
    assert out.index(UNCONFIRMED_WRITE_NOTE) < out.index("**IMAGES:**")
