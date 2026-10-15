"""baidu_search: разбор выдачи Baidu, распознавание капчи, схема и промпты."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from bot.agent import prompts, tools
from bot.agent.tools import TOOL_SCHEMAS, ToolExecutor, personal_tool_schemas

SAMPLE = """
<html><body>
<div class="result-op c-container xpath-log new-pmd" id="1" tpl="image_grid_san"
     mu="http://image.baidu.com/search/index?tn=baiduimage&amp;word=x">
  <h3><a href="https://image.baidu.com/search/x">大陆智源 ANDI 机器人 - 百度图片</a></h3>
  <span class="c-showurl">百度图片</span>
</div>
<div class="result c-container xpath-log new-pmd" id="2" tpl="www_index"
     mu="https://www.dalurobot.com/andi.html">
  <h3 class="c-title"><a href="http://www.baidu.com/link?url=abc">ANDI-安防巡检机器人 | 大陆智源</a></h3>
  <div class="c-abstract">ANDI是一款自主巡检机器人，采用步甲底盘，支持模块化负载。</div>
  <div class="c-showurl">www.dalurobot.com</div>
</div>
<div class="result c-container xpath-log new-pmd" id="3" tpl="www_index">
  <h3><a href="http://www.baidu.com/link?url=onlyredirect">без mu — только редирект</a></h3>
</div>
<div class="result c-container xpath-log new-pmd" id="4" tpl="www_index"
     mu="https://mp.weixin.qq.com/s?__biz=1">
  <h3><a href="http://www.baidu.com/link?url=def">喜报! 大陆智源机器人 荣获一等奖</a></h3>
  <span class="content-right_8Zs40">近日，大陆智源机器人在2025智能机器人大赛中获得一等奖。</span>
  <span class="c-color-gray">微信公众平台</span>
</div>
<div class="result c-container xpath-log new-pmd" id="5" tpl="www_index"
     mu="https://www.dalurobot.com/andi.html">
  <h3><a href="http://www.baidu.com/link?url=dup">дубль того же URL</a></h3>
</div>
</body></html>
"""


def test_parse_baidu_html_takes_real_urls_and_skips_aggregates() -> None:
    res = ToolExecutor._parse_baidu_html(SAMPLE, max_results=10)
    hrefs = [r["href"] for r in res]
    assert hrefs == ["https://www.dalurobot.com/andi.html", "https://mp.weixin.qq.com/s?__biz=1"]
    assert res[0]["title"] == "ANDI-安防巡检机器人 | 大陆智源"
    assert "步甲底盘" in res[0]["body"]
    assert res[0]["source"] == "www.dalurobot.com"
    assert res[1]["source"] == "微信公众平台"
    assert "一等奖" in res[1]["body"]
    # сниппет не содержит заголовка и источника
    assert not res[1]["body"].startswith("喜报")
    assert not res[1]["body"].endswith("微信公众平台")


def test_parse_baidu_html_respects_max_results() -> None:
    assert len(ToolExecutor._parse_baidu_html(SAMPLE, max_results=1)) == 1


def test_baidu_blocked_detection() -> None:
    assert ToolExecutor._baidu_blocked("", "")
    assert ToolExecutor._baidu_blocked("<html>百度安全验证</html>", "https://www.baidu.com/s")
    assert ToolExecutor._baidu_blocked("<html>c-container</html>", "https://wappass.baidu.com/static/captcha")
    assert ToolExecutor._baidu_blocked("<html>nothing here</html>", "https://www.baidu.com/s")
    assert not ToolExecutor._baidu_blocked(SAMPLE, "https://www.baidu.com/s?wd=x")


@pytest.mark.asyncio
async def test_baidu_search_falls_back_to_ddg_on_captcha(tmp_path) -> None:
    ex = ToolExecutor.__new__(ToolExecutor)

    class _Resp:
        text = "<html>百度安全验证</html>"
        url = "https://wappass.baidu.com/x"

    class _Client:
        def __init__(self, **kw):
            pass

        def get(self, *a, **kw):
            return _Resp()

    class _DDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, region=None, max_results=5):
            assert region == "cn-zh"
            return [{"title": "ddg", "href": "https://example.cn", "body": query}]

    import sys
    import types
    fake_primp = types.ModuleType("primp")
    fake_primp.Client = _Client
    fake_ddgs = types.ModuleType("ddgs")
    fake_ddgs.DDGS = _DDGS
    with patch.dict(sys.modules, {"primp": fake_primp, "ddgs": fake_ddgs}):
        out = await ex._baidu_search("大陆智源", 3)
    assert out.startswith("[Baidu недоступен")
    assert json.loads(out.split("\n", 1)[1])[0]["href"] == "https://example.cn"


@pytest.mark.asyncio
async def test_baidu_search_returns_parsed_json() -> None:
    ex = ToolExecutor.__new__(ToolExecutor)

    class _Resp:
        text = SAMPLE
        url = "https://www.baidu.com/s?wd=x"

    class _Client:
        def __init__(self, **kw):
            pass

        def get(self, *a, **kw):
            return _Resp()

    import sys
    import types
    fake_primp = types.ModuleType("primp")
    fake_primp.Client = _Client
    with patch.dict(sys.modules, {"primp": fake_primp}):
        out = await ex._baidu_search("大陆智源", 5)
    data = json.loads(out)
    assert data[0]["href"] == "https://www.dalurobot.com/andi.html"


def test_schema_and_prompts_mention_baidu() -> None:
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert "baidu_search" in names
    assert names.index("baidu_search") == names.index("web_search") + 1
    assert "baidu_search" in [t["function"]["name"] for t in personal_tool_schemas()]
    assert "baidu_search" not in tools.PERSONAL_WRITE_TOOLS      # поиск — не запись
    assert "baidu_search" in prompts.PERSONAL_SYSTEM
    assert "иероглиф" in prompts.PERSONAL_SYSTEM.lower()
