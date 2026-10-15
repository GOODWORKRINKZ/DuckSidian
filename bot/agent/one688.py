"""Поиск на 1688 через 1688-cli (github.com/superjack2050/1688-cli) — ТОЛЬКО чтение.

CLI водит браузер с залогиненной сессией покупателя (вход — один раз `1688 login` на
сервере, делает человек). У CLI есть команды корзины/заказа/переписки с продавцами —
агенту они недоступны: argv собирается только из белого списка режимов и полей, свободных
аргументов нет. Вывод CLI (JSON) ужимается до компактного текста под лимит tool-результата.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODES = ("search", "image", "offer", "compare", "suppliers", "supplier", "orders", "order")
_SORTS = {"relevance", "best-selling", "price-asc", "price-desc"}
_VERIFIED = {"any", "factory", "business", "super-factory"}
_ORDER_STATUS = {"all", "waitbuyerpay", "waitsellersend", "waitbuyerreceive", "success", "cancel"}
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_ID_RE = re.compile(r"^\d{6,20}$")
_TARGET_RE = re.compile(r"^(\d{6,20}|b2b-[\w-]{4,60})$")
# Браузер с одним профилем — запросы строго по одному.
_lock = asyncio.Lock()

SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_1688",
        "description": (
            "Поиск на 1688.com (китайский опт) — только чтение. mode: "
            "search — товары по ключевому слову (лучше по-китайски, переведи сам: 直流有刷电机 24V 350W); "
            "image — товары по фото (image = путь raw/assets/... из vault или URL картинки); "
            "offer — карточка товара по offer_ids (цены по объёму, MOQ, характеристики, упаковка); "
            "compare — сравнить несколько offer_ids; suppliers — поиск фабрик/поставщиков по слову; "
            "supplier — проверка поставщика (target = offerId или b2b-memberId); "
            "orders — заказы пользователя на 1688; order — детали/логистика заказа (order_id). "
            "Медленный (браузер, 10–60 с) — не дёргай без нужды, max ≤ 10."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": list(MODES)},
                "query": {"type": "string", "description": "ключевое слово (search/suppliers)"},
                "image": {"type": "string", "description": "путь в vault или URL (image)"},
                "offer_ids": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string", "description": "offerId или b2b-memberId (supplier)"},
                "order_id": {"type": "string"},
                "max": {"type": "integer", "description": "сколько результатов (1–10)"},
                "sort": {"type": "string", "enum": sorted(_SORTS)},
                "price_min": {"type": "number"},
                "price_max": {"type": "number"},
                "verified": {"type": "string", "enum": sorted(_VERIFIED)},
                "factory_only": {"type": "boolean"},
                "status": {"type": "string", "enum": sorted(_ORDER_STATUS)},
                "logistics": {"type": "boolean", "description": "order: трекинг вместо деталей"},
            },
            "required": ["mode"],
        },
    },
}


class One688Error(ValueError):
    """Неверные аргументы инструмента — сообщение уходит агенту как ERROR."""


def _text(v: Any, cap: int = 120) -> str:
    s = " ".join(str(v or "").split())
    if any(ch in s for ch in "\x00\n\r"):
        raise One688Error("недопустимые символы")
    return s[:cap]


def _ids(raw: Any, cap: int) -> list[str]:
    ids = [str(x).strip() for x in (raw or []) if str(x).strip()]
    bad = [i for i in ids if not _ID_RE.match(i)]
    if bad or not ids:
        raise One688Error(f"offer_ids — числовые id 1688 (получено: {bad or 'пусто'})")
    return ids[:cap]


def build_args(a: dict[str, Any], vault_root: Path) -> list[str]:
    """Аргументы для CLI из полей инструмента. Только чтение: режимы — из MODES,
    значения валидируются; ничего от модели не попадает в argv как есть-флаг."""
    mode = str(a.get("mode") or "").strip().lower()
    if mode not in MODES:
        raise One688Error(f"mode должен быть одним из: {', '.join(MODES)}")
    n = max(1, min(int(a.get("max") or 8), 10))
    if mode in ("search", "suppliers"):
        q = _text(a.get("query"))
        if not q or q.startswith("-"):
            raise One688Error("нужен query (ключевое слово)")
        if mode == "search":
            args = ["search", q, "--max", str(n), "--exclude-ads"]
            if a.get("sort") in _SORTS:
                args += ["--sort", a["sort"]]
            if a.get("verified") in _VERIFIED:
                args += ["--verified", a["verified"]]
            for k, flag in (("price_min", "--price-min"), ("price_max", "--price-max")):
                if a.get(k) is not None:
                    args += [flag, str(float(a[k]))]
            return args
        args = ["supplier", "search", q, "--max", str(n)]
        if a.get("factory_only"):
            args.append("--factory-only")
        return args
    if mode == "image":
        src = _text(a.get("image"), 600)
        if re.match(r"^https?://\S+$", src):
            return ["image-search", src]
        p = (vault_root / src.lstrip("/")).resolve()
        try:
            p.relative_to(vault_root.resolve())
        except ValueError as exc:
            raise One688Error("image — путь внутри vault (raw/assets/...) или http(s) URL") from exc
        if p.suffix.lower() not in _IMG_EXT or not p.is_file():
            raise One688Error(f"image: нет картинки jpg/png/webp по пути {src}")
        return ["image-search", str(p)]
    if mode == "offer":
        return ["offer", *_ids(a.get("offer_ids"), 5)]
    if mode == "compare":
        ids = _ids(a.get("offer_ids"), 8)
        if len(ids) < 2:
            raise One688Error("compare — минимум 2 offer_ids")
        return ["compare", *ids]
    if mode == "supplier":
        t = _text(a.get("target"), 80)
        if not _TARGET_RE.match(t):
            raise One688Error("target — offerId (цифры) или b2b-memberId")
        return ["supplier", "inspect", t]
    if mode == "orders":
        st = a.get("status") if a.get("status") in _ORDER_STATUS else "all"
        return ["order", "list", "--status", st]
    oid = _text(a.get("order_id"), 30)
    if not _ID_RE.match(oid):
        raise One688Error("order_id — числовой id заказа")
    return ["order", "logistics" if a.get("logistics") else "get", oid]


# ----------------------------------------------------------------- форматирование

def _g(d: Any, *path: str, default: Any = "") -> Any:
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def _offer_line(i: int, o: dict) -> str:
    price = _g(o, "price", "text") or _g(o, "priceRange")
    sup = _g(o, "supplier", "name")
    years = _g(o, "supplier", "years")
    loc = " ".join(x for x in (_g(o, "location", "province"), _g(o, "location", "city")) if x)
    ver = _g(o, "verified")
    badges = [n for n, k in (("суперфабрика", "superFactory"), ("фабрика", "factory")) if _g(ver, k)][:1]
    demand = _g(o, "demand", "orderCountText")
    rep = _g(o, "demand", "repurchaseRateText")
    parts = [f"{i}. {_g(o, 'title')[:90]}", f"{price} ¥" if price else "",
             f"{sup}" + (f" ({years} г.)" if years else "") + (f", {loc}" if loc else ""),
             ", ".join(badges), f"заказов {demand}" if demand else "", f"повтор {rep}" if rep else "",
             f"id {_g(o, 'offerId')}", _g(o, "url")]
    return " | ".join(p for p in parts if p)


def _fmt_offers(d: dict) -> str:
    offers = d.get("offers") or []
    head = f"[1688: {d.get('total', len(offers))} шт." + (
        f" по запросу «{d['keyword']}»" if d.get("keyword") else " по фото") + "]"
    return "\n".join([head] + [_offer_line(i, o) for i, o in enumerate(offers, 1)]) if offers \
        else head + " ничего не найдено"


def _fmt_offer(o: dict) -> str:
    tiers = "; ".join(f"от {t.get('minQty')} шт — {t.get('price')} ¥" for t in o.get("priceTiers") or [])
    attrs = "; ".join(f"{a.get('name')}: {a.get('value')}" for a in (o.get("attributes") or [])[:15])
    pack = "; ".join(str(x) for x in (o.get("packageInfo") or [])[:4])
    lines = [
        f"## {o.get('title', '')} (id {o.get('offerId', '')})",
        f"Цена: {o.get('priceRange') or o.get('priceMin', '')} ¥ за {o.get('unitName') or 'шт'}; "
        f"MOQ {o.get('minOrderQty', '?')}" + (f"; продано {o['saledCount']}" if o.get("saledCount") else ""),
        f"Ступени: {tiers}" if tiers else "",
        f"Поставщик: {_g(o, 'supplier', 'name')} (memberId {_g(o, 'supplier', 'memberId')})",
        f"Характеристики: {attrs}" if attrs else "",
        f"Упаковка: {pack}" if pack else "",
        f"SKU: {len(o.get('skus') or [])}" if o.get("skus") else "",
        f"URL: {o.get('url') or o.get('detailUrl', '')}",
        f"Фото: {o.get('mainImage')}" if o.get("mainImage") else "",
    ]
    return "\n".join(x for x in lines if x)


def _fmt_supplier_item(i: int, it: dict) -> str:
    s = it.get("supplier") or {}
    loc = s.get("location")
    loc = " ".join(str(v) for v in loc.values() if v) if isinstance(loc, dict) else (loc or "")
    prev = "; ".join(_g(p, "title")[:40] for p in (s.get("offersPreview") or [])[:2])
    parts = [f"{i}. {s.get('companyName') or s.get('name', '')}", loc,
             f"на 1688 {s['tp']} г." if s.get("tp") else "", f"оценка {it.get('score')}" if it.get("score") else "",
             f"товары: {prev}" if prev else "", s.get("shopUrl", "")]
    return " | ".join(str(p) for p in parts if p)


def _fmt_inspect(d: dict) -> str:
    s, f, t = d.get("supplier") or {}, d.get("factory") or {}, d.get("trust") or {}
    scores = "; ".join(f"{x.get('label')}: {x.get('score')}" for x in t.get("serviceScores") or [])
    lines = [
        f"## Поставщик: {s.get('name', '')} (memberId {s.get('memberId', '')})",
        f"Фабрика: {'да' if f.get('isFactory') else 'нет'}" + (" · суперфабрика" if f.get("superFactory") else "")
        + (f" · на 1688 {f['tpYears']} г." if f.get("tpYears") else ""),
        f"Основана: {f.get('establishedAtText')}" if f.get("establishedAtText") else "",
        f"Адрес: {f.get('address') or f.get('location', '')}" if (f.get("address") or f.get("location")) else "",
        f"Штат: {f.get('employeeScale') or f.get('workerCount')}" if (f.get("employeeScale") or f.get("workerCount")) else "",
        f"Удержание покупателей: {t.get('retentionRate')}" if t.get("retentionRate") else "",
        f"Сервис: {scores}" if scores else "",
        f"Товаров: {_g(d, 'offers', 'availableCount')}" if _g(d, "offers", "availableCount") else "",
        f"Магазин: {s.get('shopUrl')}" if s.get("shopUrl") else "",
        ("Предупреждения: " + "; ".join(map(str, d["warnings"]))) if d.get("warnings") else "",
    ]
    return "\n".join(x for x in lines if x)


def format_result(mode: str, data: Any, cap: int = 5_500) -> str:
    try:
        if mode in ("search", "image"):
            out = _fmt_offers(data)
        elif mode == "offer":
            out = "\n\n".join(_fmt_offer(o) for o in data.get("offers", [])) if data.get("mode") == "batch" \
                else _fmt_offer(data)
            if data.get("failures"):
                out += "\nНе удалось: " + "; ".join(f"{x.get('offerId')}: {x.get('message')}" for x in data["failures"])
        elif mode == "compare":
            out = "\n".join(
                f"- {it.get('offerId')}: " + (f"оценка {it.get('score')} — {it.get('summary')}" if it.get("ok")
                                             else f"ошибка {_g(it, 'error', 'message')}")
                for it in data.get("items", []))
        elif mode == "suppliers":
            items = data.get("items") or []
            out = f"[1688 поставщики: {data.get('total', len(items))}]\n" + "\n".join(
                _fmt_supplier_item(i, it) for i, it in enumerate(items, 1))
        elif mode == "supplier":
            out = _fmt_inspect(data)
        else:
            out = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (AttributeError, TypeError, KeyError) as exc:
        log.warning("1688 format %s failed (%s), raw json", mode, exc)
        out = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return out if len(out) <= cap else out[:cap] + "\n...[срезано]"


# ----------------------------------------------------------------- запуск

_EXIT_HINTS = {
    3: ("ERROR: сессия 1688 не залогинена. Скажи пользователю: нужен разовый вход — на сервере "
        "бота выполнить `1688 login` и отсканировать QR в приложении 1688. Повторно не вызывай."),
    4: ("ERROR: 1688 показал слайдер-капчу. Скажи пользователю, что нужен ручной проход "
        "(`1688 search … --headed` на сервере). Повторно не вызывай."),
}


async def run(cli: str, a: dict[str, Any], vault_root: Path, *, timeout: float = 180,
              profile: str = "") -> str:
    try:
        args = build_args(a, vault_root)
    except (One688Error, ValueError, TypeError) as exc:
        return f"ERROR: {exc}"
    cmd = shlex.split(cli) + args + ["--json"] + (["--profile", profile] if profile else [])
    mode = str(a.get("mode")).lower()
    async with _lock:
        log.info("1688-cli: %s", " ".join(args))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL)
        except FileNotFoundError:
            return "ERROR: 1688-cli не установлен на сервере (ONE688_CLI)."
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"ERROR: 1688 не ответил за {int(timeout)} с — попробуй позже или уже запрос."
    if proc.returncode in _EXIT_HINTS:
        return _EXIT_HINTS[proc.returncode]
    text = out.decode("utf-8", "replace").strip()
    try:
        data = json.loads(text) if text else None
    except json.JSONDecodeError:
        data = None
    if proc.returncode != 0 or data is None or (isinstance(data, dict) and data.get("ok") is False):
        msg = (data or {}).get("message") if isinstance(data, dict) else None
        tail = err.decode("utf-8", "replace").strip().splitlines()[-3:]
        return f"ERROR: 1688-cli код {proc.returncode}: {msg or ' / '.join(tail) or text[:300]}"
    return format_result(mode, data)
