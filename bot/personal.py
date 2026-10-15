"""Личный режим: персональное пространство сотрудника в vault'е.

Каждый разрешённый пользователь получает свой изолированный мини-vault:

    vault/users/<user_id>/
      AGENTS.md            — схема личного пространства (из vault-template/personal/)
      index.md, log.md
      raw/inbox/<date>.md  — сырой журнал диалога с ботом (текст, транскрипты, файлы)
      raw/assets/<date>/   — присланные файлы
      wiki/projects/       — проекты сотрудника
      wiki/notes/          — заметки/мысли
      wiki/ideas/          — идеи
      wiki/research/       — подборки/фактчеки, собранные агентом по запросу
      wiki/queries/        — сохранённые ответы /ask

«Артефакт» = любой файл в wiki/ (markdown) или raw/assets/ (бинарник).
Артефакт можно отправить («пошарить») в vault группового чата — он ложится в
raw/notes/ группы и в БД как сообщение, чтобы ingest затащил его в общую wiki.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ChatConfig, settings
from .wiki import Wiki, WikiPathError

log = logging.getLogger(__name__)

USERS_DIR = "users"
SHARED_DIR = "shared"
SHARED_PREFIX = "shared/"   # project_slug воркспейса вида "shared/<slug>" → общий vault
PERSONAL_SUBDIRS = (
    "raw/inbox", "raw/assets",
    "wiki/projects", "wiki/notes", "wiki/ideas", "wiki/research", "wiki/queries",
)
ARTIFACT_KINDS = {
    "projects": "wiki/projects",
    "notes": "wiki/notes",
    "ideas": "wiki/ideas",
    "research": "wiki/research",
    "queries": "wiki/queries",
    "files": "raw/assets",
}


def user_cfg(user_id: int) -> ChatConfig:
    """ChatConfig для личного пространства: chat_id == user_id (DM), name — путь."""
    return ChatConfig(chat_id=user_id, name=f"{USERS_DIR}/{user_id}", topic_id=None)


def is_user_cfg(cfg: ChatConfig | None) -> bool:
    return bool(cfg) and cfg.name.startswith(f"{USERS_DIR}/")


def slugify(text: str, max_len: int = 40) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    if not s:
        s = "item"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


def project_slug(name: str, max_len: int = 48) -> str:
    """Slug страницы проекта: кириллица/латиница сохраняются, пробелы → дефис."""
    s = re.sub(r"[^\w\s-]", "", (name or "").strip(), flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return (s[:max_len].rstrip("-")) or "Project"


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Грубый YAML-frontmatter парсер: {key: value} + тело."""
    fm: dict[str, str] = {}
    if not text.startswith("---\n"):
        return fm, text
    end = text.find("\n---", 4)
    if end < 0:
        return fm, text
    for line in text[4:end].splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip('"').strip("'")
    body = text[end + 4:].lstrip("\n")
    return fm, body


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    return ""


@dataclass
class Artifact:
    path: str          # rel path внутри личного vault
    kind: str          # projects | notes | ideas | research | queries | files | other
    title: str
    updated: str       # YYYY-MM-DD или ""
    size: int

    def label(self) -> str:
        upd = f" · {self.updated}" if self.updated else ""
        return f"{self.title or Path(self.path).stem}{upd}"


class PersonalSpace:
    """Обёртка над личным vault'ом пользователя."""

    def __init__(self, user_id: int, display_name: str | None = None, *,
                 shared_slug: str | None = None):
        self.user_id = user_id
        self.display_name = display_name or str(user_id)
        self.shared_slug = shared_slug
        if shared_slug:
            # Совместный воркспейс: общий vault vault/shared/<slug>/, несколько участников.
            self.cfg = ChatConfig(chat_id=0, name=f"{SHARED_DIR}/{shared_slug}", topic_id=None)
            self.root = settings.vault_path / SHARED_DIR / shared_slug
        else:
            self.cfg = user_cfg(user_id)
            self.root = settings.vault_path / USERS_DIR / str(user_id)
        self.ensure()
        self.wiki = Wiki(self.root)

    @classmethod
    def shared(cls, slug: str, display_name: str | None = None) -> "PersonalSpace":
        return cls(0, display_name or "team", shared_slug=slug)

    @property
    def is_shared(self) -> bool:
        return bool(self.shared_slug)

    # ----- seeding -----

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in PERSONAL_SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        tmpl_dir = Path(__file__).resolve().parent.parent / "vault-template" / "personal"
        agents = self.root / "AGENTS.md"
        if not agents.exists():
            src = tmpl_dir / "AGENTS.md"
            if src.exists():
                shutil.copy2(src, agents)
            else:
                agents.write_text("# AGENTS.md — личное пространство\n", encoding="utf-8")
        for fname in ("index.md", "log.md"):
            p = self.root / fname
            if not p.exists():
                src = tmpl_dir / fname
                if src.exists():
                    shutil.copy2(src, p)
                else:
                    p.write_text(f"# {fname}\n", encoding="utf-8")
        tmpl = self.root / "wiki" / "projects" / "_template.md"
        if not tmpl.exists():
            src = tmpl_dir / "wiki" / "projects" / "_template.md"
            if src.exists():
                shutil.copy2(src, tmpl)

    # ----- inbox (raw journal) -----

    def append_inbox(
        self,
        *,
        who: str,
        text: str,
        ts: datetime | None = None,
        message_id: int | None = None,
        media_type: str | None = None,
        media_path: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """Дописать реплику в raw/inbox/<date>.md. Возвращает rel путь файла.
        chat_id — личка, откуда реплика (в общем vault якорь включает его, см. msg_anchor)."""
        ts = ts or datetime.now(timezone.utc)
        date_iso = ts.strftime("%Y-%m-%d")
        rel = f"raw/inbox/{date_iso}.md"
        p = self.root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                f"# Inbox — {date_iso}\n\n"
                "Сырой журнал диалога с ботом (read-only для агента). "
                "Каждая реплика помечена якорем `^msg-<id>`.\n\n",
                encoding="utf-8",
            )
        lines = [f"### [{ts.strftime('%H:%M')}] {who}"]
        if media_type and media_path:
            lines.append(f"> 📎 *{media_type}*: `[[{media_path}]]`")
        for ln in (text or "").strip().splitlines():
            lines.append(f"> {ln}")
        if not (text or "").strip() and not media_path:
            lines.append("> *(пусто)*")
        if message_id is not None:
            lines.append(f"^{self.msg_anchor(message_id, chat_id)}")
        lines.append("")
        with p.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return rel

    def msg_anchor(self, message_id: int | None, chat_id: int | None = None) -> str:
        """Якорь реплики в raw/inbox: `msg-<id>`. В общем vault реплики приходят из
        разных личек, где message_id пересекаются, — там `msg-<chat_id>-<id>`."""
        if self.is_shared and chat_id is not None:
            return f"msg-{chat_id}-{message_id}"
        return f"msg-{message_id}"

    def log(self, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            self.wiki.append_file("log.md", f"## [{ts}] {line}")
        except Exception as exc:  # noqa: BLE001
            log.warning("personal log append failed: %s", exc)

    # ----- artifacts -----

    def _kind_of(self, rel: str) -> str:
        for kind, prefix in ARTIFACT_KINDS.items():
            if rel.startswith(prefix + "/"):
                return kind
        return "other"

    def list_artifacts(self, kind: str | None = None, limit: int = 60) -> list[Artifact]:
        """Все артефакты (wiki/**.md + raw/assets/**), свежие сверху."""
        out: list[Artifact] = []
        roots: list[Path] = []
        if kind and kind in ARTIFACT_KINDS:
            roots = [self.root / ARTIFACT_KINDS[kind]]
        else:
            roots = [self.root / "wiki", self.root / "raw" / "assets"]
        for base in roots:
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file() or p.name.startswith(("_", ".")):
                    continue
                rel = str(p.relative_to(self.root)).replace("\\", "/")
                k = self._kind_of(rel)
                title, updated = "", ""
                if p.suffix.lower() == ".md":
                    try:
                        text = p.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    fm, body = parse_frontmatter(text)
                    title = fm.get("title") or _first_heading(body) or p.stem
                    updated = fm.get("updated") or fm.get("created") or ""
                else:
                    title = p.name
                if not updated:
                    updated = datetime.fromtimestamp(
                        p.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                out.append(Artifact(rel, k, title, updated, p.stat().st_size))
        out.sort(key=lambda a: (a.updated, a.path), reverse=True)
        return out[:limit]

    def resolve_artifact(self, ref: str) -> str | None:
        """Найти артефакт по точному пути, basename или уникальной подстроке."""
        ref = (ref or "").strip().strip("`").lstrip("/")
        if not ref:
            return None
        try:
            if (self.root / ref).is_file():
                return ref.replace("\\", "/")
            if not ref.endswith(".md") and (self.root / f"{ref}.md").is_file():
                return f"{ref}.md"
        except OSError:
            pass
        arts = self.list_artifacts(limit=10_000)
        low = ref.lower()
        exact = [a for a in arts if Path(a.path).name.lower() in (low, f"{low}.md")]
        if len(exact) == 1:
            return exact[0].path
        sub = [a for a in arts if low in a.path.lower() or low in a.title.lower()]
        if len(sub) == 1:
            return sub[0].path
        return None

    def read_artifact(self, rel: str, max_bytes: int = 200_000) -> str:
        return self.wiki.read_file(rel, max_bytes=max_bytes)

    def delete_artifact(self, rel: str) -> str:
        """Удалить артефакт. Разрешено только в wiki/ и raw/assets/."""
        p = self.wiki.resolve(rel)
        rel_norm = str(p.relative_to(self.root)).replace("\\", "/")
        if not (rel_norm.startswith("wiki/") or rel_norm.startswith("raw/assets/")):
            raise WikiPathError(f"удалять можно только из wiki/ или raw/assets/: {rel_norm}")
        if rel_norm.endswith("_template.md"):
            raise WikiPathError("шаблон не удаляется")
        if not p.is_file():
            raise WikiPathError(f"файл не найден: {rel_norm}")
        p.unlink()
        self.log(f"delete | {rel_norm}")
        return rel_norm

    def ensure_project(self, name: str) -> str:
        """Создать страницу проекта из шаблона (если нет). Возвращает slug."""
        slug = project_slug(name)
        rel = f"wiki/projects/{slug}.md"
        if (self.root / rel).exists():
            return slug
        tmpl = self.root / "wiki" / "projects" / "_template.md"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if tmpl.exists():
            body = tmpl.read_text(encoding="utf-8").replace("{{name}}", name).replace("{{date}}", today)
        else:
            body = f"---\ntype: project\ntitle: \"{name}\"\nstatus: active\ncreated: {today}\nupdated: {today}\n---\n\n# {name}\n"
        self.wiki.write_file(rel, body)
        try:
            self.wiki.append_file("index.md", f"- [[{slug}]] — active; воркспейс создан {today}")
        except Exception:  # noqa: BLE001
            pass
        self.log(f"project | {rel} | created from workspace")
        return slug

    def save_research(self, title: str, body_md: str, query: str = "") -> str:
        """Сохранить подборку агента в wiki/research/<date>-<slug>.md."""
        now = datetime.now(timezone.utc)
        rel = f"wiki/research/{now.strftime('%Y-%m-%d')}-{slugify(title)}.md"
        fm = (
            "---\n"
            "type: research\n"
            f"title: \"{title.replace(chr(34), chr(39))}\"\n"
            f"created: {now.strftime('%Y-%m-%d')}\n"
            f"updated: {now.strftime('%Y-%m-%d')}\n"
            f"query: \"{(query or '').replace(chr(34), chr(39))[:200]}\"\n"
            "---\n\n"
        )
        self.wiki.write_file(rel, fm + f"# {title}\n\n{body_md.strip()}\n")
        return rel


# ----- sharing -----

_ASSET_LINK_RE = re.compile(r"!?\[\[(raw/assets/[^\]|#]+)(?:\|[^\]]*)?\]\]")


def _target_wiki(target: ChatConfig) -> Wiki:
    from .orchestrator import _chat_wiki  # noqa: PLC0415  (избегаем цикла импорта)
    return _chat_wiki(target)


@dataclass
class ShareResult:
    dst_note: str                 # rel путь заметки в vault группы
    copied_assets: list[str]      # rel пути скопированных файлов (в vault группы)
    title: str
    body_md: str                  # что постить в группу (markdown)
    is_binary: bool


def share_artifact(
    space: PersonalSpace,
    rel: str,
    target: ChatConfig,
    *,
    author: str,
    comment: str = "",
) -> ShareResult:
    """Скопировать артефакт из личного vault в vault группового чата.

    Markdown → raw/notes/<ts>-shared-<slug>.md в группе (+ копии вложенных
    raw/assets/* файлов в raw/assets/shared/<date>/). Бинарник → копия в
    raw/assets/shared/<date>/ и короткая сопроводительная заметка.
    Запись в БД/Telegram делает вызывающий (orchestrator.share_to_chat).
    """
    src_abs = space.wiki.resolve(rel)
    if not src_abs.is_file():
        raise WikiPathError(f"файл не найден: {rel}")
    twiki = _target_wiki(target)
    now = datetime.now(timezone.utc)
    date_iso = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    shared_dir_rel = f"raw/assets/shared/{date_iso}"
    (twiki.root / shared_dir_rel).mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    def _copy_asset(asset_rel: str) -> str | None:
        try:
            a_abs = space.wiki.resolve(asset_rel)
        except WikiPathError:
            return None
        if not a_abs.is_file():
            return None
        safe = "".join(c for c in a_abs.name if c.isalnum() or c in "._-") or "file.bin"
        dst_rel = f"{shared_dir_rel}/u{space.user_id}_{safe}"
        dst_abs = twiki.root / dst_rel
        if not dst_abs.exists():
            shutil.copy2(a_abs, dst_abs)
        copied.append(dst_rel)
        return dst_rel

    is_binary = src_abs.suffix.lower() != ".md"
    if is_binary:
        dst_rel = _copy_asset(rel)
        title = src_abs.name
        body = (
            f"Файл от {author}: `[[{dst_rel}]]`"
            + (f"\n\n{comment.strip()}" if comment.strip() else "")
        )
    else:
        text = src_abs.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(text)
        title = fm.get("title") or _first_heading(body) or src_abs.stem
        # Копируем вложения и переписываем ссылки на новый путь.
        def _sub(m: re.Match) -> str:
            new_rel = _copy_asset(m.group(1))
            if not new_rel:
                return m.group(0)
            return m.group(0).replace(m.group(1), new_rel)
        body = _ASSET_LINK_RE.sub(_sub, body)
        if comment.strip():
            body = f"> 💬 {comment.strip()}\n\n{body}"

    slug = slugify(title)
    dst_note = f"raw/notes/{ts}-shared-{slug}.md"
    note = (
        "---\n"
        "type: shared\n"
        f"from: \"{author}\"\n"
        f"from_user_id: {space.user_id}\n"
        f"source: \"users/{space.user_id}/{rel}\"\n"
        f"created: {date_iso}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"_📤 Поделился {author} из личного пространства ({date_iso})._\n\n"
        f"{body.strip()}\n"
    )
    twiki.write_file(dst_note, note)
    space.log(f"share | {rel} → {target.name}:{dst_note}")
    return ShareResult(dst_note, copied, title, body.strip(), is_binary)


@dataclass
class WorkspaceShareResult:
    title: str
    slug: str
    pages: list[ShareResult]
    copied_assets: list[str]
    summary_md: str


_PROJECT_FM_RE = re.compile(r'^project:\s*"?\[\[([^\]|]+)(?:\|[^\]]*)?\]\]"?', re.MULTILINE)


def collect_workspace_pages(space: PersonalSpace, slug: str) -> list[str]:
    """Страница проекта + все wiki-страницы, привязанные к ней
    (frontmatter `project: "[[Slug]]"` или wikilink [[Slug]] в теле)."""
    project_rel = f"wiki/projects/{slug}.md"
    out: list[str] = []
    if (space.root / project_rel).is_file():
        out.append(project_rel)
    needle = f"[[{slug}]]".lower()
    needle_alias = f"[[{slug}|".lower()
    for a in space.list_artifacts(limit=10_000):
        if a.kind == "files" or a.path == project_rel or not a.path.endswith(".md"):
            continue
        try:
            text = (space.root / a.path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        low = text.lower()
        m = _PROJECT_FM_RE.search(text)
        if (m and m.group(1).strip().lower() == slug.lower()) or needle in low or needle_alias in low:
            out.append(a.path)
    return out


def share_workspace(
    space: PersonalSpace,
    slug: str,
    target: ChatConfig,
    *,
    author: str,
    comment: str = "",
) -> WorkspaceShareResult:
    """Поделиться воркспейсом целиком: каждая страница → отдельная shared-заметка
    в vault группы, вложения копируются. Возвращает сводку для анонса."""
    pages_rel = collect_workspace_pages(space, slug)
    if not pages_rel:
        raise WikiPathError(f"у воркспейса {slug} нет страниц (нет wiki/projects/{slug}.md)")
    results: list[ShareResult] = []
    assets: list[str] = []
    for i, rel in enumerate(pages_rel):
        res = share_artifact(space, rel, target, author=author, comment=comment if i == 0 else "")
        results.append(res)
        assets.extend(res.copied_assets)
    title = results[0].title or slug
    lines = [f"📦 **{author}** поделился воркспейсом **{title}**", ""]
    if comment.strip():
        lines += [f"> 💬 {comment.strip()}", ""]
    lines.append(f"Страниц: {len(results)}, файлов: {len(assets)}. В vault чата:")
    for r in results:
        lines.append(f"- {r.title} — `{r.dst_note}`")
    # Краткое содержание проекта (первые ~1200 символов).
    body = results[0].body_md.strip()
    if body:
        lines += ["", "---", body[:1200] + ("…" if len(body) > 1200 else "")]
    space.log(f"share-workspace | {slug} → {target.name} | {len(results)} pages")
    return WorkspaceShareResult(title, slug, results, assets, "\n".join(lines))


def artifact_markdown_table(arts: list[Artifact]) -> str:
    if not arts:
        return "_(пусто)_"
    icons = {"projects": "📁", "notes": "📝", "ideas": "💡", "research": "🔎",
             "queries": "❓", "files": "📎", "other": "📄"}
    lines = []
    for i, a in enumerate(arts, 1):
        lines.append(f"{i}. {icons.get(a.kind, '📄')} **{a.title}** — `{a.path}`"
                     + (f" _{a.updated}_" if a.updated else ""))
    return "\n".join(lines)


def describe_space(space: PersonalSpace) -> dict[str, Any]:
    counts = {k: 0 for k in ARTIFACT_KINDS}
    for a in space.list_artifacts(limit=10_000):
        if a.kind in counts:
            counts[a.kind] += 1
    return {"root": str(space.root), "counts": counts}


# ----- sharing: user → user (снимок воркспейса) -----

_TYPE_DIRS = {
    "project": "wiki/projects", "note": "wiki/notes", "idea": "wiki/ideas",
    "research": "wiki/research", "source": "wiki/sources", "query": "wiki/queries",
}


def _unique_rel(root: Path, rel: str, suffix: str) -> str:
    """Если файл существует — добавить суффикс к имени."""
    if not (root / rel).exists():
        return rel
    p = Path(rel)
    cand = f"{p.parent.as_posix()}/{p.stem}-{suffix}{p.suffix}"
    i = 2
    while (root / cand).exists():
        cand = f"{p.parent.as_posix()}/{p.stem}-{suffix}-{i}{p.suffix}"
        i += 1
    return cand


def stage_workspace_for_user(
    src: PersonalSpace, slug: str, dst: PersonalSpace, token: str, *, author: str, comment: str = "",
) -> dict[str, Any]:
    """Положить снимок воркспейса в raw/incoming/<token>/ получателя (ожидает принятия).

    Вложения сразу копируются в raw/assets/shared/<date>/ получателя, ссылки переписаны.
    Возвращает манифест {title, slug, from, pages:[{src, staged, type}], assets:[...]}.
    """
    pages_rel = collect_workspace_pages(src, slug)
    if not pages_rel:
        raise WikiPathError(f"у воркспейса {slug} нет страниц (нет wiki/projects/{slug}.md)")
    date_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stage_dir = f"raw/incoming/{token}"
    (dst.root / stage_dir).mkdir(parents=True, exist_ok=True)
    assets_dir_rel = f"raw/assets/shared/{date_iso}"
    (dst.root / assets_dir_rel).mkdir(parents=True, exist_ok=True)
    copied_assets: list[str] = []

    def _copy_asset(asset_rel: str) -> str | None:
        try:
            a_abs = src.wiki.resolve(asset_rel)
        except WikiPathError:
            return None
        if not a_abs.is_file():
            return None
        safe = "".join(c for c in a_abs.name if c.isalnum() or c in "._-") or "file.bin"
        new_rel = f"{assets_dir_rel}/u{src.user_id}_{safe}"
        if not (dst.root / new_rel).exists():
            shutil.copy2(a_abs, dst.root / new_rel)
        copied_assets.append(new_rel)
        return new_rel

    pages: list[dict[str, str]] = []
    title = slug
    for i, rel in enumerate(pages_rel):
        text = (src.root / rel).read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(text)
        ptype = fm.get("type") or ("project" if rel.startswith("wiki/projects/") else "note")
        if i == 0:
            title = fm.get("title") or _first_heading(body) or slug
        body = _ASSET_LINK_RE.sub(
            lambda m: m.group(0).replace(m.group(1), _copy_asset(m.group(1)) or m.group(1)), body,
        )
        fm_out = dict(fm)
        fm_out["shared_from"] = author
        fm_out["shared_from_user_id"] = str(src.user_id)
        fm_out["shared_at"] = date_iso
        fm_out.setdefault("type", ptype)
        head = "---\n" + "\n".join(f"{k}: {v}" for k, v in fm_out.items()) + "\n---\n\n"
        staged = f"{stage_dir}/{Path(rel).name}"
        (dst.root / staged).write_text(head + body.lstrip("\n"), encoding="utf-8")
        pages.append({"src": rel, "staged": staged, "type": ptype})
    manifest = {
        "title": title, "slug": slug, "from": author, "from_user_id": src.user_id,
        "to_user_id": dst.user_id, "comment": comment, "pages": pages, "assets": copied_assets,
        "created": date_iso,
    }
    (dst.root / stage_dir / "_manifest.json").write_text(
        json_dumps(manifest), encoding="utf-8",
    )
    src.log(f"share-workspace-to-user | {slug} → users/{dst.user_id} | {len(pages)} pages")
    return manifest


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=1)


def accept_staged_workspace(dst: PersonalSpace, manifest: dict[str, Any]) -> tuple[str, list[str]]:
    """Разложить снимок по wiki/ получателя. Возвращает (slug проекта, список rel страниц)."""
    placed: list[str] = []
    proj_slug = manifest["slug"]
    suffix = project_slug(f"от-{manifest.get('from') or manifest.get('from_user_id')}")[:30]
    for page in manifest["pages"]:
        staged = dst.root / page["staged"]
        if not staged.is_file():
            continue
        ptype = page.get("type") or "note"
        target_dir = _TYPE_DIRS.get(ptype, "wiki/notes")
        rel = f"{target_dir}/{staged.name}"
        rel = _unique_rel(dst.root, rel, suffix)
        if ptype == "project":
            proj_slug = Path(rel).stem
        (dst.root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(dst.root / rel))
        placed.append(rel)
    # если имя проекта изменилось (коллизия) — поправим ссылки [[old]] → [[new]] в принятых страницах
    if proj_slug != manifest["slug"]:
        for rel in placed:
            p = dst.root / rel
            t = p.read_text(encoding="utf-8")
            t2 = t.replace(f"[[{manifest['slug']}]]", f"[[{proj_slug}]]").replace(
                f"[[{manifest['slug']}|", f"[[{proj_slug}|")
            if t2 != t:
                p.write_text(t2, encoding="utf-8")
    try:
        dst.wiki.append_file(
            "index.md",
            f"- [[{proj_slug}]] — получено от {manifest.get('from')} ({manifest.get('created')}), "
            f"{len(placed)} страниц",
        )
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(dst.root / f"raw/incoming/{manifest.get('token', '')}", ignore_errors=True)
    dst.log(f"accept-workspace | {manifest['slug']} от users/{manifest.get('from_user_id')} | {len(placed)} pages → {proj_slug}")
    return proj_slug, placed


def discard_staged_workspace(dst: PersonalSpace, token: str) -> None:
    shutil.rmtree(dst.root / f"raw/incoming/{token}", ignore_errors=True)


# ----- совместный воркспейс: общий vault для нескольких участников -----

def convert_workspace_to_shared(owner: PersonalSpace, slug: str) -> PersonalSpace:
    """Перенести проект (страница + связанные страницы + их вложения) из личного vault
    владельца в общий vault/shared/<slug>/. Пути внутри сохраняются (ссылки не ломаются).
    Если общий vault с таким slug уже есть — страницы докладываются (без перезаписи)."""
    shared = PersonalSpace.shared(slug, owner.display_name)
    pages = collect_workspace_pages(owner, slug)
    moved: list[str] = []
    for rel in pages:
        src_p = owner.root / rel
        dst_p = shared.root / rel
        if not src_p.is_file():
            continue
        try:
            text = src_p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _ASSET_LINK_RE.finditer(text):
            a_rel = m.group(1)
            a_src = owner.root / a_rel
            a_dst = shared.root / a_rel
            if a_src.is_file() and not a_dst.exists():
                a_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(a_src), str(a_dst))
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if dst_p.exists():
            dst_p = shared.root / _unique_rel(shared.root, rel, f"от-{project_slug(owner.display_name)[:20]}")
        shutil.move(str(src_p), str(dst_p))
        moved.append(rel)
    try:
        owner.wiki.append_file("index.md", f"- [[{slug}]] ➜ совместный воркспейс `shared/{slug}` (перенесён)")
    except Exception:  # noqa: BLE001
        pass
    try:
        shared.wiki.append_file("index.md", f"- [[{slug}]] — совместный проект, владелец {owner.display_name}; {len(moved)} страниц")
    except Exception:  # noqa: BLE001
        pass
    owner.log(f"convert-to-shared | {slug} | {len(moved)} pages → shared/{slug}")
    shared.log(f"created from users/{owner.user_id} | {len(moved)} pages")
    return shared


def shared_slug_of(project_slug_value: str | None) -> str | None:
    """'shared/<slug>' → '<slug>', иначе None."""
    if project_slug_value and project_slug_value.startswith(SHARED_PREFIX):
        return project_slug_value[len(SHARED_PREFIX):]
    return None
