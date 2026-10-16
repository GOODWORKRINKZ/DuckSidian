"""Цикл агента: chat → tool_calls → results → chat → ... → finish."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..media_refs import find_voice_marker, has_images_block, mentioned_media, resolve_in_vault
from .deepseek import LLMClient, extract_usage
from .tools import TOOL_SCHEMAS, ToolExecutor

log = logging.getLogger(__name__)


@dataclass
class AgentRun:
    summary: str = ""
    steps: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_cached: int = 0
    tools_called: list[str] = field(default_factory=list)
    # Инструменты, вернувшие успех (не «ERROR: …»). Guard записи смотрит сюда: упавший
    # edit_file — не запись, ответ «записал» после него — выдумка.
    tools_ok: list[str] = field(default_factory=list)
    # Реально изменённые файлы vault (успешные write-инструменты): путь → действие.
    writes: dict[str, str] = field(default_factory=dict)
    # Ответ утверждал запись, а write-инструменты не вызывались: nudged — агенту
    # один раз досылали подсказку; unconfirmed_write — так и не записал.
    write_nudged: bool = False
    unconfirmed_write: bool = False
    # То же для медиа: ответ с блоком IMAGES / «посмотрел картинку», а файл не
    # скачивали и не открывали (fetch_image / describe_media / generate_image).
    media_nudged: bool = False
    unconfirmed_media: bool = False
    # Произвольная проверка ответа (answer_check вызывающего) один раз вернула подсказку.
    check_nudged: bool = False


# «Записал / добавила / сохранили / готово / обновлено» — но не «не записал».
_WRITE_CLAIM_RE = re.compile(
    r"(?<![Нн]е )(?<!\w)(?:"
    r"(?:за|до)писал[аио]?|добавил[аи]?|сохранил[аи]?|обновил[аи]?|создал[аи]?"
    r"|дополнил[аи]?|внёс|внес(?:ла|ли)?|занёс|занес(?:ла|ли)?"
    r"|(?:за|до)писано|добавлено|сохранено|обновлено|создано|дополнено|внесено"
    r"|готово"
    r")(?!\w)",
    re.IGNORECASE,
)

# Служебное сообщение (role=system): пользователь его не видит, и первый вариант
# ответа ему тоже не отправлен — финальный ответ не должен на них ссылаться.
WRITE_NUDGE = (
    "[Служебная проверка бота — пользователь её не видит.] Твой ответ утверждает запись "
    "(«записал/добавил/готово»), но в этом ходе не было ни одного вызова инструмента "
    "записи (edit_file / write_file / append_file …) — в vault ничего не изменилось. "
    "Этот вариант ответа пользователю НЕ отправлен. Сделай запись инструментами и "
    "заверши finish(), а если записывать нечего — ответь без слов «записал/добавил/готово». "
    "Пиши финальный ответ так, будто это твой первый ответ: не упоминай эту проверку, "
    "прошлый вариант ответа, «теперь по-настоящему», «в прошлом ходу» и т.п."
)


# finish(summary=«Ответ готов для озвучки» / «Прикольный ответ на подборку: …») — модель
# описала ответ вместо ответа, а сам текст написала рядом с вызовом (content).
_STUB_SUMMARY_RE = re.compile(
    r"(?:ответ|текст|шутка|отчёт|пересказ)[^.!?\n]{0,60}?(?:готов|составлен|сформирован|ниже|выше|для озвучки|для пересылки)"
    r"|^(?:прикольный|готовый|короткий|краткий|мой)\s+(?:ответ|текст|отчёт)\b",
    re.IGNORECASE,
)


def prefer_body(body: str, summary: str) -> bool:
    """Текст рядом с finish() — настоящий ответ, а summary — заглушка/описание?"""
    body = (body or '').strip()
    summary = (summary or '').strip()
    if len(body) < 80:
        return False
    if not summary:
        return True
    if find_voice_marker(body) and not find_voice_marker(summary):
        return True
    if _STUB_SUMMARY_RE.search(summary) and len(body) > len(summary):
        return True
    return len(body) > 2 * len(summary) and len(summary) < 400


def claims_write(text: str) -> bool:
    """Текст ответа утверждает, что что-то записано/сохранено."""
    return bool(text) and bool(_WRITE_CLAIM_RE.search(text))


_MEDIA_NOUN = r"(?:картинк|фотк|фото|изображени|снимк|снимок|скриншот|скрин|иллюстраци)\w*"
_MEDIA_VERB = (
    r"(?<![Нн]е )(?<!\w)(?:"
    r"(?:по|про|рас|за)?смотрел[аи]?|глянул[аи]?|увидел[аи]?|вижу|разглядел[аи]?"
    r"|скачал[аи]?|загрузил[аи]?|нашёл|нашел|нашл[аи]|достал[аи]?"
    r"|прикрепил[аи]?|прикрепляю|прикреплю|отправляю|пришлю|присылаю|высылаю|скидываю|кидаю"
    r"|сгенерировал[аи]?|нарисовал[аи]?"
    r")(?!\w)"
)
# «посмотрел/скачал/прикрепляю … картинку», «фото прикрепил», «на картинке видно»,
# «что увидел по картинке», «вот фото …» — но не «не нашёл картинку».
_MEDIA_CLAIM_RE = re.compile(
    rf"{_MEDIA_VERB}[^.!?\n]{{0,40}}?{_MEDIA_NOUN}"
    rf"|(?<!\w){_MEDIA_NOUN}[^.!?\n]{{0,25}}?{_MEDIA_VERB}"
    rf"|(?<!\w)(?:на|по)\s+(?:этой\s+|той\s+|самой\s+)?{_MEDIA_NOUN}[^.!?\n]{{0,40}}?"
    r"(?<!\w)(?:видн\w*|изображ\w*|запечатл\w*|показан\w*|различим\w*)"
    rf"|(?<!\w)вот\s+(?:\w+\s+)?{_MEDIA_NOUN}",
    re.IGNORECASE,
)

# Инструменты, после которых ответ может законно показывать/описывать картинку.
MEDIA_TOOLS = frozenset({"fetch_image", "describe_media", "generate_image"})

MEDIA_NUDGE = (
    "[Служебная проверка бота — пользователь её не видит.] Твой ответ показывает или "
    "описывает картинку (блок IMAGES, «посмотрел/скачал/прикрепляю», «на фото видно»), но "
    "в этом ходе ты не вызывал fetch_image / describe_media / generate_image, а файлов из "
    "блока IMAGES в vault нет — ты не скачивал и не смотрел файл. Этот вариант ответа "
    "пользователю НЕ отправлен. Вызови fetch_image(url) (картинку найди через web_search / "
    "fetch_url) и при необходимости describe_media(path), либо честно скажи, что картинку не "
    "нашёл. Не описывай то, чего не видел, и не указывай в IMAGES несуществующие пути. "
    "Пиши финальный ответ так, будто это твой первый ответ: не упоминай эту проверку и "
    "прошлый вариант ответа."
)

UNCONFIRMED_MEDIA_NOTE = "⚠️ картинку не скачал и не смотрел — описание выше не подтверждено"


def claims_media(text: str) -> bool:
    """Текст ответа утверждает, что картинка получена/просмотрена/прикреплена."""
    return bool(text) and (has_images_block(text) or bool(_MEDIA_CLAIM_RE.search(text)))


def _ctx_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


async def _emit(on_step: Any, name: str, args: dict[str, Any]) -> None:
    """Служебное событие прогресса (`__thinking__`, `__note__`); ошибки колбэка не роняют прогон."""
    if on_step is None:
        return
    try:
        await on_step(name, args)
    except Exception as exc:  # noqa: BLE001
        log.debug("on_step failed: %s", exc)


async def _compress_old_tool_results(
    client: LLMClient,
    messages: list[dict[str, Any]],
    *,
    keep_last: int = 4,
    min_len: int = 400,
) -> bool:
    """Суммаризировать старые tool-ответы через LLM, чтобы ужать контекст.

    tool_call_id связки не трогаем (DeepSeek требует совпадения), только
    подменяем `content` на короткое summary. Возвращает True если что-то сжали.
    """
    tool_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" and len(str(m.get("content") or "")) >= min_len
    ]
    if len(tool_idxs) <= keep_last:
        return False
    targets = tool_idxs[:-keep_last]
    parts = []
    for n, i in enumerate(targets, 1):
        name = messages[i].get("name") or "tool"
        content = str(messages[i].get("content") or "")
        parts.append(f"### [{n}] tool={name}\n{content}")
    joined = "\n\n".join(parts)
    sys = (
        "Ты — компрессор контекста. Получишь несколько результатов tool-вызовов. "
        "Для КАЖДОГО верни ОДНУ строку формата `[N] краткое summary` "
        "(2-3 предложения, ключевые факты/имена/пути/числа). "
        "Без вступлений. По одной строке на блок."
    )
    try:
        resp = await client.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": joined},
            ],
            tools=None,
        )
        text = resp["choices"][0]["message"].get("content") or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("context compression call failed: %s", exc)
        return False
    # Распарсить строки `[N] ...`.
    summaries: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            n_str, rest = line[1:].split("]", 1)
            n = int(n_str.strip())
            summaries[n] = rest.strip()
        except (ValueError, IndexError):
            continue
    if not summaries:
        log.warning("compression produced no parseable lines, head=%r", text[:200])
        return False
    for n, i in enumerate(targets, 1):
        s = summaries.get(n)
        if s:
            messages[i]["content"] = f"[compressed] {s}"
    return True


async def run_agent(
    *,
    client: LLMClient,
    executor: ToolExecutor,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 30,
    tools: list[dict] | None = None,
    history: list[dict[str, Any]] | None = None,
    force_finish_hint: str | None = None,
    on_step: Any = None,
    interject: Any = None,
    tool_result_cap: int = 6_000,
    ctx_char_budget: int = 400_000,
    compress_ctx_threshold: int = 80_000,
    write_tools: frozenset[str] | set[str] | None = None,
    media_tools: frozenset[str] | set[str] | None = None,
    answer_check: Any = None,
) -> AgentRun:
    """Один прогон агента до вызова finish() или исчерпания шагов.

    write_tools: если задан и финальный ответ утверждает запись («записал»,
    «готово»…), а ни один из этих инструментов не вызывался, — один раз
    возвращаем агента в цикл с WRITE_NUDGE; если и потом записи нет —
    run.unconfirmed_write=True (решение, что показать пользователю, — у вызывающего).

    media_tools: то же для картинок — ответ с блоком IMAGES или «посмотрел/прикрепляю
    картинку», а ни один из этих инструментов не вызывался и ни один упомянутый
    медиафайл не существует в vault → MEDIA_NUDGE, потом run.unconfirmed_media.

    Подсказка за прогон одна (общая для обоих guard'ов): если модель после неё
    перефразировала выдумку, второй подсказки нет, но флаг unconfirmed_* ставится.

    answer_check(text) -> str | None: проверка вызывающего (например, «ответ спрашивает
    автора репоста, хотя он указан»); вернула текст — он досылается агенту как system
    и агент делает ещё шаг (тоже один раз за прогон, run.check_nudged=True).
    """
    vault_root: Path | None = getattr(getattr(executor, "wiki", None), "root", None)

    def _fake_write(text: str) -> bool:
        return (
            write_tools is not None
            and not any(t in write_tools for t in run.tools_ok)
            and claims_write(text)
        )

    def _fake_media(text: str) -> bool:
        if media_tools is None or any(t in media_tools for t in run.tools_called):
            return False
        if not claims_media(text):
            return False
        # Ссылается на реальный файл vault (из list_dir / прошлых ходов) — не выдумка.
        if vault_root is not None and any(
            resolve_in_vault(rel, vault_root) for rel in mentioned_media(text)
        ):
            return False
        return True

    def _nudge(text: str, step: int, where: str) -> bool:
        """Досылает подсказку(и), если ответ выдуман и подсказок ещё не было."""
        if run.write_nudged or run.media_nudged or run.check_nudged:
            return False
        fake_write, fake_media = _fake_write(text), _fake_media(text)
        extra = answer_check(text) if (answer_check is not None and text) else None
        if not (fake_write or fake_media or extra):
            return False
        if extra and not (fake_write or fake_media):
            run.check_nudged = True
            log.warning("agent step %d: %s failed answer_check, nudging. head=%r", step, where, text[:160])
            messages.append({"role": "system", "content": extra})
            return True
        if fake_write:
            run.write_nudged = True
            log.warning("agent step %d: %s claims a write without write tools, nudging. head=%r",
                        step, where, text[:160])
            messages.append({"role": "system", "content": WRITE_NUDGE})
        if fake_media:
            run.media_nudged = True
            log.warning("agent step %d: %s shows/describes media it never fetched, nudging. head=%r",
                        step, where, text[:160])
            messages.append({"role": "system", "content": MEDIA_NUDGE})
        return True

    tool_schemas = tools if tools is not None else TOOL_SCHEMAS
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    # История диалога (для личного режима): plain user/assistant реплики без tool_calls.
    for h in history or []:
        role = h.get("role")
        content = str(h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    run = AgentRun(transcript=messages)

    for step in range(max_steps):
        run.steps = step + 1
        # Лог размера контекста — чтобы ловить раздувание (DeepSeek рвёт большие тела).
        ctx_chars = _ctx_chars(messages)
        if ctx_chars > compress_ctx_threshold and step > 0:
            compressed = await _compress_old_tool_results(client, messages, keep_last=3)
            if compressed:
                ctx_chars = _ctx_chars(messages)
                log.info("agent step %d: context compressed, new ctx_chars=%d", step, ctx_chars)
        if ctx_chars > ctx_char_budget:
            log.warning(
                "agent step %d: ctx %d > budget %d",
                step, ctx_chars, ctx_char_budget,
            )
        log.info("agent step %d: ctx_msgs=%d ctx_chars=%d", step, len(messages), ctx_chars)
        # На предпоследнем и последнем шаге форсируем finish, чтобы не выйти
        # из цикла с пустым summary.
        steps_left = max_steps - step
        if steps_left <= 2:
            messages.append({
                "role": "system",
                "content": (
                    f"⚠️ Осталось шагов: {steps_left}. БОЛЬШЕ НЕ ИЩИ — "
                    "вызови finish(summary=...) сейчас с тем разбором, "
                    "что у тебя уже есть. Лучше неполный отчёт, чем пустой."
                ),
            })
        # Реплики пользователя, пришедшие во время хода (диалог с ask_choice) — в контекст.
        if interject is not None:
            try:
                extra = interject() or []
            except Exception as exc:  # noqa: BLE001
                log.debug("interject failed: %s", exc)
                extra = []
            for text in extra:
                log.info("agent step %d: user interjection %r", step, str(text)[:120])
                messages.append({"role": "user", "content": (
                    "[Пользователь написал, пока ты работал — учти это прямо сейчас, "
                    f"а не как новую задачу]: {text}")})
        await _emit(on_step, "__thinking__", {
            "step": step + 1, "max_steps": max_steps, "ctx_chars": ctx_chars,
            "ctx_msgs": len(messages),
            "tokens": run.tokens_prompt + run.tokens_completion,
            "tokens_cached": run.tokens_cached,
        })
        resp = await client.chat(messages=messages, tools=tool_schemas)
        _p, _c, _ca = extract_usage(resp)
        run.tokens_prompt += _p
        run.tokens_completion += _c
        run.tokens_cached += _ca
        try:
            choice = resp["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError):
            log.error("Bad LLM response: %s", resp)
            break

        # Сохраняем assistant-сообщение как есть (для tool_call_id связки).
        raw_content = msg.get("content") or ""
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": raw_content,
        }
        # DeepSeek thinking mode: reasoning_content нужно передавать обратно
        # в следующих запросах, иначе API вернёт 400.
        reasoning = msg.get("reasoning_content")
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            # Пояснение модели рядом с tool-call'ами («сейчас проверю X») — в прогресс.
            note = _strip_think(raw_content).strip()
            if note:
                await _emit(on_step, "__note__", {"text": note})
        messages.append(assistant_msg)

        # DeepSeek иногда пишет tool-calls прямо в content как XML-разметку
        # (с fullwidth || или ASCII ||) вместо структурированного tool_calls.
        # Детектируем по наличию 'DSML' или '<invoke ' в тексте.
        _is_dsml = bool(raw_content) and (
            "DSML" in raw_content or "<invoke " in raw_content
        )
        if _is_dsml:
            log.warning(
                "agent step %d: DeepSeek returned XML tool-call in content "
                "(tool_calls=%d), asking to reformat. head=%r",
                step, len(tool_calls), raw_content[:120],
            )
            messages.append({
                "role": "user",
                "content": (
                    "Используй tool_calls API, не пиши XML вручную. "
                    "Повтори то же действие через tool call."
                ),
            })
            continue

        if not tool_calls:
            # Модель решила ответить без tool-call → считаем что закончила.
            run.summary = _strip_think(raw_content)
            if _nudge(run.summary, step, "answer"):
                continue
            run.finished = True
            break

        finish_called = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                summary = str(args.get("summary", ""))
                body = _strip_think(raw_content).strip()
                if prefer_body(body, summary):
                    # Модель написала ответ рядом с вызовом, а в summary — описание
                    # («ответ готов для озвучки»): пользователю нужен сам ответ.
                    log.info("agent step %d: finish(): summary is a stub (%d chars), using assistant text (%d chars)",
                             step, len(summary), len(body))
                    summary = body
                run.summary = summary
                run.finished = True
                finish_called = True
                log.info("agent step %d: finish(summary=%r)", step, run.summary[:120])
                if len(run.summary) > 120:
                    log.info("agent step %d: finish(summary_tail=%r)", step, run.summary[-300:])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "content": "OK",
                    }
                )
                continue

            run.tools_called.append(name)
            log.info("agent step %d: tool=%s args=%s", step, name, str(args)[:200])
            if on_step is not None:
                try:
                    await on_step(name, args)
                except Exception as exc:  # noqa: BLE001
                    log.debug("on_step failed: %s", exc)
            result = await executor.call(name, args)
            if not result.lstrip().startswith("ERROR"):
                run.tools_ok.append(name)
                target = args.get("dst") or args.get("path") or args.get("file_path")
                if write_tools is not None and name in write_tools and target:
                    run.writes[str(target).lstrip("/")] = name
            log.info("agent step %d: tool=%s result_len=%d head=%r",
                     step, name, len(result), result[:120])
            # Trim too-large tool outputs to keep context window sane.
            if len(result) > tool_result_cap:
                result = result[:tool_result_cap] + "\n...[truncated]"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": result,
                }
            )

        if finish_called:
            if _nudge(run.summary, step, "finish()"):
                run.finished = False
                continue
            break

    if not run.finished:
        # Форс-финал: модель не вызвала finish() за max_steps. Делаем один
        # запрос БЕЗ tools — пусть просто выдаст текст разбора, его и берём.
        log.warning("agent didn't finish in %d steps, forcing plain-text summary", max_steps)
        try:
            messages.append({
                "role": "system",
                "content": force_finish_hint or (
                    "СТОП. Шаги кончились. Никаких tool-вызовов — их у тебя "
                    "больше нет. Ответь обычным текстом итоговым результатом по тому, "
                    "что уже собрано: кратко, по делу, на русском."
                ),
            })
            final = await client.chat(messages=messages, tools=None)
            _p, _c, _ca = extract_usage(final)
            run.tokens_prompt += _p
            run.tokens_completion += _c
            run.tokens_cached += _ca
            forced_text = (
                final["choices"][0]["message"].get("content")
                or run.summary
                or "(agent stopped without finish())"
            )
            # Forced call also sometimes returns DSML XML — sanitize immediately.
            if "DSML" in forced_text:
                log.warning("forced summary contains DSML, sanitizing")
                forced_text = _sanitize_summary(forced_text)
            run.summary = forced_text
            run.finished = True
        except Exception as exc:  # noqa: BLE001
            log.error("forced summary call failed: %s", exc)
            run.summary = run.summary or "(agent stopped without finish())"
    run.summary = _sanitize_summary(run.summary)
    run.unconfirmed_write = _fake_write(run.summary)
    run.unconfirmed_media = _fake_media(run.summary)
    log.info(
        "agent done: steps=%d tokens_prompt=%d tokens_completion=%d "
        "tokens_total=%d tokens_cached=%d",
        run.steps,
        run.tokens_prompt,
        run.tokens_completion,
        run.tokens_prompt + run.tokens_completion,
        run.tokens_cached,
    )
    return run


# DeepSeek иногда возвращает в content свой внутренний tool-call формат как plain text:
#   <｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="..."><｜｜DSML｜｜parameter ...>
# Количество fullwidth-pipes варьируется (｜ или ｜｜).
# Чистим: находим первое вхождение DSML, откатываемся до '<', обрезаем.
import re as _re


_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL | _re.IGNORECASE)


def _strip_think(s: str) -> str:
    """Убрать блоки рассуждений `<think>…</think>` (MiniMax M2, Qwen3 и др.
    отдают их прямо в content). Незакрытый <think> — отрезаем до конца."""
    if not s or "<think" not in s.lower():
        return s
    s = _THINK_RE.sub("", s)
    low = s.lower()
    idx = low.find("<think>")
    if idx != -1:
        s = s[:idx]
    return s.strip()


def _sanitize_summary(s: str) -> str:
    if not s:
        return ""
    s = _strip_think(s)
    if "DSML" not in s:
        return s.strip()
    idx = s.find("DSML")
    lt = s.rfind("<", 0, idx)
    if lt != -1:
        s = s[:lt].rstrip()
    return s.strip()
