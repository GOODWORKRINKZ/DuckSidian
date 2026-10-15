"""Админ-команды, общие для группового и личного режимов: /model, /allow, /deny, /users."""
from __future__ import annotations

import logging
from html import escape as _esc

from aiogram.types import Message

from ..config import LLM_PROVIDERS, settings
from ..db import DB
from ..orchestrator import Orchestrator

log = logging.getLogger(__name__)


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in settings.telegram_admins


async def allowed_private_users(db: DB) -> set[int]:
    """Кому разрешён личный режим: админы + .env + добавленные через /allow."""
    users = set(settings.telegram_admins) | set(settings.telegram_private_users)
    users |= set(await db.get_private_users())
    return users


async def is_private_allowed(db: DB, user_id: int | None) -> bool:
    return user_id is not None and user_id in await allowed_private_users(db)


# ---------------------------------------------------------------- /model

def format_model_list(orch: Orchestrator) -> str:
    lines = [f"🧠 Текущая модель: <b>{_esc(orch.model_info())}</b>", "", "<b>Провайдеры:</b>"]
    for p in orch.client.available_providers():
        mark = "🟢" if p["has_key"] else "⚪️"
        cur = " ← активен" if p["key"] == orch.client.provider.key else ""
        models = ", ".join(p["models"][:4]) or "—"
        lines.append(
            f"{mark} <code>{p['key']}</code> — {_esc(LLM_PROVIDERS[p['key']].title)}{cur}\n"
            f"   модели: <i>{_esc(models)}</i>"
        )
    lines += [
        "",
        "Переключить: <code>/model &lt;provider&gt; [model]</code>",
        "Пример: <code>/model qwen qwen3.8-max</code>, <code>/model minimax</code>, "
        "<code>/model deepseek deepseek-v4-flash</code>",
        "🟢 — ключ задан в .env, ⚪️ — ключа нет.",
    ]
    return "\n".join(lines)


async def handle_model_command(msg: Message, args: str, orch: Orchestrator) -> None:
    if not is_admin(msg.from_user.id if msg.from_user else None):
        await msg.reply("⛔ только для админов")
        return
    parts = (args or "").split()
    if not parts:
        await msg.reply(format_model_list(orch), parse_mode="HTML")
        return
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else None
    try:
        desc = await orch.set_model(provider, model)
    except ValueError as exc:
        await msg.reply(f"❌ {_esc(str(exc))}", parse_mode="HTML")
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("set_model failed")
        await msg.reply(f"❌ {type(exc).__name__}: {_esc(str(exc))}", parse_mode="HTML")
        return
    await msg.reply(f"✅ Модель переключена: <b>{_esc(desc)}</b>", parse_mode="HTML")


# ---------------------------------------------------------------- /allow /deny /users

def _parse_user_id(args: str, msg: Message) -> int | None:
    """id из аргумента, либо из реплая (from_user / forward_from)."""
    a = (args or "").strip()
    if a.lstrip("-").isdigit():
        return int(a)
    r = msg.reply_to_message
    if r is not None:
        fwd = getattr(r, "forward_from", None)
        if fwd is not None:
            return fwd.id
        if r.from_user and not r.from_user.is_bot:
            return r.from_user.id
    return None


async def handle_allow_command(msg: Message, args: str, db: DB) -> None:
    if not is_admin(msg.from_user.id if msg.from_user else None):
        await msg.reply("⛔ только для админов")
        return
    uid = _parse_user_id(args, msg)
    if uid is None:
        await msg.reply(
            "Использование: <code>/allow &lt;user_id&gt;</code> "
            "(или реплаем на сообщение/форвард пользователя).\n"
            "Пользователь узнаёт свой id, написав боту /start в личку.",
            parse_mode="HTML",
        )
        return
    added = await db.add_private_user(uid)
    await msg.reply(
        f"✅ <code>{uid}</code> {'добавлен в' if added else 'уже есть в'} списке личного режима.",
        parse_mode="HTML",
    )


async def handle_deny_command(msg: Message, args: str, db: DB) -> None:
    if not is_admin(msg.from_user.id if msg.from_user else None):
        await msg.reply("⛔ только для админов")
        return
    uid = _parse_user_id(args, msg)
    if uid is None:
        await msg.reply("Использование: <code>/deny &lt;user_id&gt;</code>", parse_mode="HTML")
        return
    removed = await db.remove_private_user(uid)
    note = ""
    if uid in settings.telegram_admins or uid in settings.telegram_private_users:
        note = "\n⚠️ Пользователь также прописан в .env (TELEGRAM_ADMINS/TELEGRAM_PRIVATE_USERS) — доступ останется."
    await msg.reply(
        f"{'🗑 Удалён' if removed else 'ℹ️ Не было в списке'}: <code>{uid}</code>{note}",
        parse_mode="HTML",
    )


async def handle_users_command(msg: Message, db: DB) -> None:
    if not is_admin(msg.from_user.id if msg.from_user else None):
        await msg.reply("⛔ только для админов")
        return
    dyn = await db.get_private_users()
    lines = ["<b>Личный режим — кому разрешён:</b>"]
    for uid in settings.telegram_admins:
        lines.append(f"• <code>{uid}</code> — admin (.env)")
    for uid in settings.telegram_private_users:
        if uid not in settings.telegram_admins:
            lines.append(f"• <code>{uid}</code> — .env TELEGRAM_PRIVATE_USERS")
    for uid in dyn:
        if uid not in settings.telegram_admins and uid not in settings.telegram_private_users:
            lines.append(f"• <code>{uid}</code> — /allow")
    if len(lines) == 1:
        lines.append("(никого)")
    lines.append("\n/allow &lt;id&gt; — добавить, /deny &lt;id&gt; — убрать.")
    await msg.reply("\n".join(lines), parse_mode="HTML")
