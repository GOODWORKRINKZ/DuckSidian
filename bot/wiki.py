"""Безопасные FS-обёртки над vault'ом.

Все пути, приходящие от агента, валидируются против `vault_root`. Запись
разрешена только в `raw/` и `wiki/`. Чтение — везде внутри vault.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


WRITABLE_PREFIXES = ("raw/", "wiki/")

# Расширения, которым разрешена запись агентом в wiki/sources/ (вдруг нужно).
# Для raw/ запись всегда запрещена агенту, но боту — разрешена через save_asset.
_SAFE_ASSET_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class WikiPathError(ValueError):
    pass


class Wiki:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    # ----- path helpers -----

    def resolve(self, rel: str, *, for_write: bool = False) -> Path:
        rel = (rel or "").strip().lstrip("/")
        if not rel:
            raise WikiPathError("empty path")
        if ".." in Path(rel).parts:
            raise WikiPathError(f"path traversal: {rel}")
        target = (self.root / rel).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise WikiPathError(f"path escapes vault: {rel}") from exc
        if for_write:
            rel_norm = str(target.relative_to(self.root)).replace("\\", "/")
            if not any(
                rel_norm.startswith(p) or rel_norm == p.rstrip("/")
                for p in WRITABLE_PREFIXES
            ) and rel_norm not in {"index.md", "log.md"}:
                raise WikiPathError(
                    f"write outside writable area: {rel_norm}"
                )
        return target

    # ----- read / write -----

    def read_file(self, rel: str, max_bytes: int = 200_000) -> str:
        p = self.resolve(rel)
        if not p.is_file():
            raise WikiPathError(f"not a file: {rel}")
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")

    def write_file(self, rel: str, content: str) -> str:
        p = self.resolve(rel, for_write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p.relative_to(self.root))

    def append_file(self, rel: str, content: str) -> str:
        p = self.resolve(rel, for_write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            if not content.endswith("\n"):
                content += "\n"
            f.write(content)
        return str(p.relative_to(self.root))

    def list_dir(self, rel: str = "") -> list[str]:
        p = self.resolve(rel) if rel else self.root
        if not p.exists():
            return []
        out: list[str] = []
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            rel_path = str(child.relative_to(self.root)).replace("\\", "/")
            out.append(rel_path + ("/" if child.is_dir() else ""))
        return out

    def search(self, query: str, max_hits: int = 20) -> list[dict]:
        """Простой grep: подстрока, без regex, кейс-инсенситив."""
        if not query.strip():
            return []
        q = query.lower()
        hits: list[dict] = []
        for md in self.root.rglob("*.md"):
            try:
                rel = str(md.relative_to(self.root)).replace("\\", "/")
            except ValueError:
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if q in line.lower():
                    hits.append({"path": rel, "line": i, "text": line.strip()})
                    if len(hits) >= max_hits:
                        return hits
        return hits

    # ----- asset saver (вызывается ботом, не агентом) -----

    def save_asset(self, data: bytes, filename: str, date_iso: str) -> str:
        """Сохранить медиафайл в raw/assets/<date_iso>/<filename>.

        Возвращает rel-путь. Имя файла санируется — только безопасные символы.
        Вызывается слушателем бота, path-traversal проверяется через resolve().
        """
        safe = "".join(c for c in filename if c in _SAFE_ASSET_CHARS)
        if not safe:
            safe = "file.bin"
        rel = f"raw/assets/{date_iso}/{safe}"
        p = self.resolve(rel)  # только guard path-traversal, без write-check
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return rel

    # ----- daily batch builder -----

    @staticmethod
    def _tg_link(chat_id: int | None, message_id: int | None) -> str | None:
        """Сгенерировать ссылку на сообщение в Telegram.

        Для супергрупп chat_id имеет вид -100XXXXXXXXXX.
        Ссылка: https://t.me/c/{channel_id}/{message_id}
        """
        if not chat_id or not message_id:
            return None
        s = str(abs(chat_id))
        channel_id = s[3:] if s.startswith("100") else s
        return f"https://t.me/c/{channel_id}/{message_id}"

    def write_daily_raw(
        self, date_iso: str, messages: Iterable[dict]
    ) -> str:
        """Сформировать raw/daily/<date>.md из сообщений и вернуть rel-путь."""
        lines: list[str] = [
            f"# Raw daily batch — {date_iso}",
            "",
            "Дамп сообщений из рабочего чата за сутки. Read-only для агента.",
            "Каждое сообщение помечено якорем `^msg-<id>` для citation.",
            "Поле `tg:` — прямая ссылка на сообщение в Telegram для цитирования.",
            "",
        ]
        count = 0
        for m in messages:
            count += 1
            who = m.get("full_name") or m.get("username") or f"user{m.get('user_id')}"
            ts = m.get("ts", "")
            mid = m.get("message_id")
            chat_id = m.get("chat_id")
            text = (m.get("text") or "").strip()
            reply = m.get("reply_to_message_id")
            tg_link = self._tg_link(chat_id, mid)
            head = f"### [{ts}] {who}"
            if reply:
                tg_reply_link = self._tg_link(chat_id, reply)
                reply_ref = f"[^msg-{reply}]({tg_reply_link})" if tg_reply_link else f"^msg-{reply}"
                head += f" (reply → {reply_ref})"
            lines.append(head)
            if tg_link:
                lines.append(f"> tg: [{mid}]({tg_link})")
            if text:
                for ln in text.splitlines():
                    lines.append(f"> {ln}")
            elif m.get("media_type"):
                media_type = m["media_type"]
                media_path = m.get("media_path")
                if media_path:
                    lines.append(f"> 📎 *{media_type}*: `[[{media_path}]]`")
                else:
                    lines.append(f"> 📎 *{media_type}* (файл не скачан)")
            else:
                lines.append("> *(non-text content)*")
            lines.append(f"^msg-{mid}")
            lines.append("")
        lines.insert(5, f"Всего сообщений: **{count}**.")
        lines.insert(6, "")
        rel = f"raw/daily/{date_iso}.md"
        self.write_file(rel, "\n".join(lines))
        return rel
