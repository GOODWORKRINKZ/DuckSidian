"""Скилы — именованные процедуры для агента (формат как у Claude Skills: markdown с
frontmatter name/description/triggers). Встроенные лежат в bot/skills/builtin/,
пользовательские — в `skills/*.md` vault'а (перекрывают встроенные с тем же name).

В промпт попадает только каталог (name — description); тело скила агент получает
инструментом use_skill(name), когда задача подходит."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "builtin"
_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: str
    body: str
    source: str  # "builtin" | путь в vault


def _parse(text: str, source: str, fallback_name: str) -> Skill | None:
    meta: dict[str, str] = {}
    body = text
    m = _FM_RE.match(text)
    if m:
        body = text[m.end():]
        for ln in m.group(1).splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    name = (meta.get("name") or fallback_name).strip().lower()
    if not _NAME_RE.match(name) or not body.strip():
        return None
    return Skill(name=name, description=meta.get("description", "").strip(),
                 triggers=meta.get("triggers", "").strip(), body=body.strip(), source=source)


def load_skills(vault_root: Path | None = None) -> dict[str, Skill]:
    """Встроенные + из vault (`skills/*.md`); vault перекрывает встроенные."""
    out: dict[str, Skill] = {}
    dirs: list[tuple[Path, str | None]] = [(BUILTIN_DIR, "builtin")]
    if vault_root is not None:
        dirs.append((vault_root / "skills", None))
    for d, src in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            source = src or str(p.relative_to(vault_root)).replace("\\", "/")  # type: ignore[arg-type]
            sk = _parse(text, source, p.stem)
            if sk:
                out[sk.name] = sk
    return out


def catalog(skills: dict[str, Skill]) -> str:
    """Короткий каталог для промпта: по строке на скил."""
    lines = []
    for sk in skills.values():
        trig = f" Триггеры: {sk.triggers}." if sk.triggers else ""
        lines.append(f"- {sk.name} — {sk.description}{trig}")
    return "\n".join(lines)


def match_command(text: str, skills: dict[str, Skill]) -> tuple[Skill, str] | None:
    """`/grill что-то` или `/skill grill что-то` → (скил, остаток текста)."""
    m = re.match(r"^/(?:skill\s+)?([a-z0-9_-]+)(?:@\w+)?\s*(.*)$", (text or "").strip(), re.DOTALL)
    if not m:
        return None
    sk = skills.get(m.group(1).lower())
    return (sk, m.group(2).strip()) if sk else None
