"""Пересланные сообщения (репосты) в личке: пометка «от кого», подсказка агенту и
защита от ответа «а кто автор?», когда автор уже указан.

Без зависимостей от aiogram: forward_header принимает любой объект с полями Message
(forward_origin / forward_from …), чтобы тесты шли на SimpleNamespace.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

FORWARD_MARK = "↪️ ПЕРЕСЛАНО (репост)"


def user_label(u: Any) -> str:
    """«Имя Фамилия (@username, id=N)» для User/Chat-подобного объекта."""
    name = (getattr(u, "full_name", None) or getattr(u, "title", None)
            or " ".join(x for x in (getattr(u, "first_name", None), getattr(u, "last_name", None)) if x)
            or "").strip()
    extra = [f"@{u.username}" if getattr(u, "username", None) else "",
             f"id={u.id}" if getattr(u, "id", None) is not None else ""]
    extra = [e for e in extra if e]
    return (name or "без имени") + (f" ({', '.join(extra)})" if extra else "")


def forward_header(msg: Any) -> str | None:
    """Пометка для агента и журнала: сообщение переслано (репост) — от кого и когда.
    Telegram даёт forward_origin: user (открытый профиль), hidden_user (только имя —
    у автора закрыта пересылка), chat (от имени группы), channel (пост канала).
    Без пересылки — None."""
    origin = getattr(msg, "forward_origin", None)
    who = ""
    date = None
    if origin is not None:
        kind = getattr(origin, "type", "") or ""
        date = getattr(origin, "date", None)
        if kind == "user" or getattr(origin, "sender_user", None) is not None:
            who = f"пользователь {user_label(origin.sender_user)}"
        elif kind == "hidden_user" or getattr(origin, "sender_user_name", None):
            who = (f"пользователь «{origin.sender_user_name}» (профиль скрыт настройками "
                   "приватности — ни @username, ни id Telegram не отдаёт)")
        elif kind == "chat" or getattr(origin, "sender_chat", None) is not None:
            who = f"чат «{getattr(origin.sender_chat, 'title', '') or ''}»"
            sig = getattr(origin, "author_signature", None)
            if sig:
                who += f", автор {sig}"
        elif kind == "channel" or getattr(origin, "chat", None) is not None:
            ch = origin.chat
            who = f"канал «{getattr(ch, 'title', '') or ''}»"
            if getattr(ch, "username", None):
                who += f" (@{ch.username})"
            sig = getattr(origin, "author_signature", None)
            if sig:
                who += f", автор {sig}"
    else:  # старые поля Bot API (< 7.0)
        if getattr(msg, "forward_from", None) is not None:
            who = f"пользователь {user_label(msg.forward_from)}"
        elif getattr(msg, "forward_sender_name", None):
            who = f"пользователь «{msg.forward_sender_name}» (профиль скрыт настройками приватности)"
        elif getattr(msg, "forward_from_chat", None) is not None:
            who = f"канал/чат «{getattr(msg.forward_from_chat, 'title', '') or ''}»"
        date = getattr(msg, "forward_date", None)
    if not who:
        return None
    when = ""
    if date is not None:
        try:
            when = f", оригинал от {date.strftime('%Y-%m-%d %H:%M')} UTC"
        except Exception:  # noqa: BLE001
            when = ""
    return f"{FORWARD_MARK}, автор — {who}{when}. Текст репоста ниже:"


def is_forward(text: str) -> bool:
    """Текст реплики начинается с пометки репоста или подборки репостов (так их кладёт
    private-хендлер)."""
    t = (text or "").lstrip()
    return t.startswith(FORWARD_MARK) or t.startswith(BUNDLE_MARK)


# «кто автор?», «от кого это?», «перешли с именем/контактом», «скинь имя/ник автора»,
# «кто это написал/переслал», «чей это репост» — но не «автор — Денис».
_ASKS_AUTHOR_RE = re.compile(
    r"(?<!\w)(?:"
    r"кто\s+(?:это\s+|же\s+|именно\s+)?(?:автор|написал|прислал|переслал|запостил|сказал|отправил)"
    r"|от\s+кого\s+(?:это|он|она|оно|сообщение|репост|пост)?"
    r"|чей\s+(?:это\s+)?(?:репост|пост|текст|ролик|скрин)"
    r"|(?:перешли|пересылай|скинь|пришли|напиши|назови|укажи|дай)\s+[^.!?\n]{0,30}?"
    r"(?:с\s+(?:видимым\s+)?(?:именем|контактом|автором|ником)|имя\s+автора|ник\s+автора|автора|имя\s+или\s+ник)"
    r"|(?:нужно|надо|хочу)\s+(?:знать|понять)[^.!?\n]{0,20}?(?:кто|от\s+кого|автора)"
    r"|без\s+(?:имени|автора)\s+не\s+(?:могу|получится|смогу)"
    r")(?!\w)",
    re.IGNORECASE,
)


def asks_for_author(text: str) -> bool:
    """Ответ спрашивает у владельца, кто автор репоста."""
    return bool(text) and bool(_ASKS_AUTHOR_RE.search(text))


def forward_hint(owner: str) -> str:
    """Блок в промпт хода рядом с репостом: автор известен, спрашивать нечего."""
    return (
        "⚠️ ЭТО РЕПОСТ (или подборка репостов — один сюжет, один ответ на всё). Автор оригинала "
        "указан в пометке «↪️ ПЕРЕСЛАНО…» в начале сообщения (у подборки — у каждой части) — "
        "он известен; НЕ спрашивай «кто автор», «от кого», «перешли с именем» и не проси контакт. "
        f"Переслал владелец ({owner}) — обращайся к нему; шутка/разбор — про автора репоста и "
        "его текст. Если автор репоста — сам бот (DuckSidianBot), это твой прежний ответ, "
        "пересланный для контекста. Не копируй свои прошлые ответы из истории: тот же ролик или "
        "текст мог обсуждаться раньше — ответ пиши заново, с учётом автора. Если по цели "
        "воркспейса ответ нужен для пересылки/голосом — положи его целиком в блок VOICE: "
        "(ровно строка «VOICE:» и дальше текст без markdown и эмодзи)."
    )


AUTHOR_NUDGE = (
    "[Служебная проверка бота — пользователь её не видит.] Твой ответ спрашивает, кто автор "
    "репоста или просит переслать сообщение с именем — но автор УЖЕ указан в пометке "
    "«↪️ ПЕРЕСЛАНО (репост), автор — …» в начале сообщения пользователя. Этот вариант ответа "
    "пользователю НЕ отправлен. Возьми автора из пометки и ответь по существу (по цели "
    "воркспейса), без вопросов про автора. Пиши финальный ответ так, будто это твой первый "
    "ответ: не упоминай эту проверку и прошлый вариант."
)


BUNDLE_MARK = "📨 ПОДБОРКА"


def bundle_text(parts: list[str]) -> str:
    """Текст хода из нескольких сообщений, пересланных подряд: один сюжет — одна реплика
    агенту, части пронумерованы по порядку. Одна часть — возвращается как есть."""
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    head = (f"{BUNDLE_MARK}: {len(parts)} сообщений, пересланных подряд — это ОДИН сюжет "
            "(цитата и реакции на неё, по порядку). Ответ один, на весь сюжет; автор каждой части — "
            "в её пометке «↪️ ПЕРЕСЛАНО…», часть без пометки написал сам владелец.")
    return head + "\n\n" + "\n\n".join(f"[{i}] {p}" for i, p in enumerate(parts, 1))


class Bundler:
    """Склейка сообщений, пришедших подряд (пачка форвардов). add(key, item) копит по
    ключу и перезапускает таймер; через delay секунд без новых — flush(key, items).
    Пока ключ открыт (is_open), очередное сообщение того же автора в той же теме
    тоже идёт в пачку."""

    def __init__(self, delay: float, flush: Callable[[Any, list[Any]], Awaitable[None]]) -> None:
        self.delay = delay
        self._flush = flush
        self._items: dict[Any, list[Any]] = {}
        self._timers: dict[Any, asyncio.Task[None]] = {}

    def is_open(self, key: Any) -> bool:
        return key in self._items

    def add(self, key: Any, item: Any) -> int:
        """Положить в пачку; возвращает её текущий размер."""
        self._items.setdefault(key, []).append(item)
        t = self._timers.pop(key, None)
        if t is not None:
            t.cancel()
        self._timers[key] = asyncio.get_running_loop().create_task(self._later(key))
        return len(self._items[key])

    async def _later(self, key: Any) -> None:
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            return
        self._timers.pop(key, None)
        items = self._items.pop(key, [])
        if not items:
            return
        try:
            await self._flush(key, items)
        except Exception:  # noqa: BLE001
            log.exception("bundle flush failed (key=%s, %d items)", key, len(items))
