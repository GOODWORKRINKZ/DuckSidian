"""search_1688: белый список команд (только чтение), валидация, форматирование, запуск CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bot.agent import one688
from bot.agent.one688 import One688Error, build_args, format_result
from bot.agent.tools import ToolExecutor, personal_tool_schemas
from bot.config import settings
from bot.wiki import Wiki

SEARCH_JSON = {
    "keyword": "直流有刷电机 24V 350W", "sort": "relevance", "total": 2, "offers": [
        {"offerId": "123456789", "title": "MY1016Z 24V 350W 直流有刷电机", "price": {"text": "98-120"},
         "supplier": {"name": "永康某电机厂", "years": 7}, "location": {"province": "浙江", "city": "金华"},
         "verified": {"factory": True, "superFactory": False}, "demand": {"orderCountText": "1200+",
                                                                         "repurchaseRateText": "35%"},
         "url": "https://detail.1688.com/offer/123456789.html"},
        {"offerId": "987654321", "title": "电机", "price": {"text": "80"}, "supplier": {"name": "X"},
         "url": "https://detail.1688.com/offer/987654321.html"},
    ]}


# --- build_args: только чтение ------------------------------------------

def test_build_args_search_with_filters(tmp_path: Path) -> None:
    args = build_args({"mode": "search", "query": "直流电机", "max": 50, "sort": "price-asc",
                       "price_max": 150, "verified": "factory"}, tmp_path)
    assert args == ["search", "直流电机", "--max", "10", "--exclude-ads", "--sort", "price-asc",
                    "--verified", "factory", "--price-max", "150.0"]


@pytest.mark.parametrize("a", [
    {"mode": "cart", "query": "x"},
    {"mode": "checkout"},
    {"mode": "seller", "query": "hi"},
    {"mode": "search", "query": "--agent"},
    {"mode": "search", "query": ""},
    {"mode": "offer", "offer_ids": ["123; rm -rf /"]},
    {"mode": "offer", "offer_ids": []},
    {"mode": "compare", "offer_ids": ["1234567"]},
    {"mode": "supplier", "target": "evil --yes"},
    {"mode": "order", "order_id": "abc"},
    {"mode": "image", "image": "../../etc/passwd.jpg"},
    {"mode": "image", "image": "raw/assets/nope.jpg"},
])
def test_build_args_rejects(a: dict, tmp_path: Path) -> None:
    with pytest.raises(One688Error):
        build_args(a, tmp_path)


def test_build_args_never_emits_write_commands(tmp_path: Path) -> None:
    img = tmp_path / "raw" / "assets" / "m.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8")
    calls = [
        {"mode": "search", "query": "a"}, {"mode": "suppliers", "query": "a", "factory_only": True},
        {"mode": "image", "image": "raw/assets/m.jpg"}, {"mode": "image", "image": "https://x.cn/a.jpg"},
        {"mode": "offer", "offer_ids": ["1234567"]}, {"mode": "compare", "offer_ids": ["1234567", "7654321"]},
        {"mode": "supplier", "target": "b2b-22abc"}, {"mode": "orders", "status": "success"},
        {"mode": "order", "order_id": "1234567", "logistics": True},
    ]
    firsts = set()
    for c in calls:
        args = build_args(c, tmp_path)
        firsts.add(" ".join(args[:2]) if args[0] in ("supplier", "order") else args[0])
        assert not {"cart", "checkout", "inquire", "chat", "messages", "--agent", "login", "logout",
                    "feedback", "--submit", "--yes"} & set(args)
    assert firsts == {"search", "supplier search", "image-search", "offer", "compare",
                      "supplier inspect", "order list", "order logistics"}
    assert build_args({"mode": "image", "image": "raw/assets/m.jpg"}, tmp_path)[1] == str(img.resolve())


# --- форматирование ------------------------------------------------------

def test_format_search() -> None:
    out = format_result("search", SEARCH_JSON)
    lines = out.splitlines()
    assert lines[0] == "[1688: 2 шт. по запросу «直流有刷电机 24V 350W»]"
    assert "98-120 ¥" in lines[1] and "永康某电机厂 (7 г.), 浙江 金华" in lines[1]
    assert "фабрика" in lines[1] and "заказов 1200+" in lines[1] and "id 123456789" in lines[1]


def test_format_offer_and_inspect_and_cap() -> None:
    offer = {"offerId": "123456789", "title": "Мотор", "priceRange": "98-120", "unitName": "个",
             "minOrderQty": 2, "priceTiers": [{"minQty": 2, "price": 120}, {"minQty": 100, "price": 98}],
             "attributes": [{"name": "电压", "value": "24V"}], "supplier": {"name": "厂", "memberId": "b2b-1"},
             "url": "u"}
    out = format_result("offer", offer)
    assert "MOQ 2" in out and "от 100 шт — 98 ¥" in out and "电压: 24V" in out
    insp = {"supplier": {"name": "厂", "memberId": "b2b-1"}, "factory": {"isFactory": True, "tpYears": 9},
            "trust": {"serviceScores": [{"label": "响应", "score": 4.8}]}, "warnings": ["w"]}
    out = format_result("supplier", insp)
    assert "Фабрика: да · на 1688 9 г." in out and "响应: 4.8" in out
    big = {"offers": [SEARCH_JSON["offers"][0]] * 200, "total": 200, "keyword": "k"}
    assert format_result("search", big).endswith("[срезано]")
    assert format_result("search", {"weird": 1}).startswith("[1688")  # неожиданная форма не падает


# --- запуск CLI ----------------------------------------------------------

FAKE = '''
import json, sys
args = sys.argv[1:]
open(sys.argv[0] + ".log", "a", encoding="utf-8").write(json.dumps(args, ensure_ascii=False) + "\\n")
q = args[1] if len(args) > 1 else ""
if q == "LOGIN": sys.exit(3)
if args[0] == "login":
    import os
    png = os.path.join(os.path.dirname(sys.argv[0]), "qr.png")
    open(png, "wb").write(b"png")
    print("Scan with your 1688 or Taobao app:")
    print("QR saved as PNG: " + png.replace(os.sep, "/"), flush=True)
    print("Waiting for scan + confirmation...")
    sys.exit(0 if os.environ.get("FAKE_LOGIN_FAIL") is None else 1)
if q == "SLIDER": sys.exit(4)
if q == "SLOW":
    import time; time.sleep(5)
if q == "BROKEN":
    print('{"ok": false, "code": "X", "message": "сломалось"}'); sys.exit(1)
print(json.dumps(json.loads(%r), ensure_ascii=False))
''' % json.dumps(SEARCH_JSON, ensure_ascii=False)


@pytest.fixture(autouse=True)
def _reset_login_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(one688, "_login_blocked_until", 0.0)


@pytest.fixture
def fake_cli(tmp_path: Path) -> tuple[str, Path]:
    script = tmp_path / "fake1688.py"
    script.write_text(FAKE, encoding="utf-8")
    return f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}"', Path(str(script) + ".log")


@pytest.mark.asyncio
async def test_run_ok_and_exit_codes(fake_cli, tmp_path: Path) -> None:
    cli, log = fake_cli
    out = await one688.run(cli, {"mode": "search", "query": "电机"}, tmp_path)
    assert out.startswith("[1688: 2 шт.")
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[0])[-1] == "--json"
    assert "/login1688" in await one688.run(cli, {"mode": "search", "query": "LOGIN"}, tmp_path)
    assert not one688.available()                       # скрыт после «не залогинен»
    await one688.run(cli, {"mode": "search", "query": "電機"}, tmp_path)
    assert one688.available()                           # успешный вызов снимает блок
    assert "слайдер" in await one688.run(cli, {"mode": "search", "query": "SLIDER"}, tmp_path)
    assert "сломалось" in await one688.run(cli, {"mode": "search", "query": "BROKEN"}, tmp_path)
    out = await one688.run(cli, {"mode": "search", "query": "SLOW"}, tmp_path, timeout=0.5)
    assert out.startswith("ERROR") and "не ответил" in out
    assert (await one688.run(cli, {"mode": "cart"}, tmp_path)).startswith("ERROR: mode")
    assert "не установлен" in await one688.run("/nonexistent/1688", {"mode": "search", "query": "a"}, tmp_path)


@pytest.mark.asyncio
async def test_tool_gated_by_config(fake_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def ask(_q, _o):
        return ""
    ex = ToolExecutor(Wiki(tmp_path), ask)
    monkeypatch.setattr(settings, "one688_cli", "")
    assert "search_1688" not in {t["function"]["name"] for t in personal_tool_schemas()}
    assert "не настроен" in await ex.call("search_1688", {"mode": "search", "query": "a"})
    monkeypatch.setattr(settings, "one688_cli", fake_cli[0])
    assert "search_1688" in {t["function"]["name"] for t in personal_tool_schemas()}
    ex = ToolExecutor(Wiki(tmp_path), ask)       # новый ход — без памяти о прошлых вызовах
    assert (await ex.call("search_1688", {"mode": "search", "query": "a"})).startswith("[1688: 2 шт.")


@pytest.mark.asyncio
async def test_run_cancel_kills_cli(fake_cli, tmp_path: Path) -> None:
    import asyncio
    cli, _ = fake_cli
    task = asyncio.create_task(one688.run(cli, {"mode": "search", "query": "SLOW"}, tmp_path, timeout=30))
    await asyncio.sleep(1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not one688._lock.locked()


@pytest.mark.asyncio
async def test_login_flow_sends_qr_and_unblocks(fake_cli, tmp_path: Path, monkeypatch) -> None:
    cli, _ = fake_cli
    monkeypatch.setattr(one688, "_login_blocked_until", 1e18)
    got: list[str] = []

    async def on_qr(path: str) -> None:
        got.append(path)
    ok, info = await one688.login_flow(cli, on_qr, timeout=30)
    assert ok and info == "вход выполнен" and got and got[0].endswith("qr.png")
    assert one688.available()
    monkeypatch.setenv("FAKE_LOGIN_FAIL", "1")
    ok, info = await one688.login_flow(cli, on_qr, timeout=30)
    assert not ok and "Waiting for scan" in info


@pytest.mark.asyncio
async def test_repeat_calls_are_not_executed_twice(tmp_path: Path, monkeypatch) -> None:
    calls: list = []

    async def fake_impl(self, name, arguments):
        calls.append((name, arguments))
        return "[1688 не отдал карточку]"
    monkeypatch.setattr(ToolExecutor, "_call_impl", fake_impl)

    async def ask(_q, _o):
        return ""
    ex = ToolExecutor(Wiki(tmp_path), ask)
    a = await ex.call("fetch_url", {"url": "https://detail.1688.com/offer/804827869776.html?share_token=x"})
    b = await ex.call("fetch_url", {"url": "https://m.1688.com/offer/804827869776.html"})
    c = await ex.call("fetch_url", {"url": "https://detail.1688.com/offer/566634192031.html"})
    await ex.call("list_dir", {"path": "wiki"})
    await ex.call("list_dir", {"path": "wiki"})                  # локальные — без дедупа
    assert a == "[1688 не отдал карточку]" and b.startswith("[ПОВТОР") and "не отдал" in b
    assert c == "[1688 не отдал карточку]" and ex.repeat_hits == 1
    assert [n for n, _ in calls] == ["fetch_url", "fetch_url", "list_dir", "list_dir"]
