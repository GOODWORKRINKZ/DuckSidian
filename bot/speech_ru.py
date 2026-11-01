"""Числа для русского TTS: даты, суммы, проценты и числа с разрядами — словами.

MiniMax читает «01.11» как «одна целая одиннадцать сотых», а «1 049,11 ₽» — кусками.
normalize_numbers() переписывает такие места так, как их произносят вслух;
остальной текст не трогает.
"""
from __future__ import annotations

import re

_UNITS_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_UNITS_F = ["", "одна", "две"] + _UNITS_M[3:]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
          "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]
# (одна, две–четыре, пять+, женский род)
_SCALES = [
    (("тысяча", "тысячи", "тысяч"), True),
    (("миллион", "миллиона", "миллионов"), False),
    (("миллиард", "миллиарда", "миллиардов"), False),
]

_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
           "сентября", "октября", "ноября", "декабря"]
# Порядковые: средний род, им. падеж («первое ноября») и муж. род, род. падеж («…шестого года»)
_ORD_N = ["", "первое", "второе", "третье", "четвёртое", "пятое", "шестое", "седьмое", "восьмое",
          "девятое", "десятое", "одиннадцатое", "двенадцатое", "тринадцатое", "четырнадцатое",
          "пятнадцатое", "шестнадцатое", "семнадцатое", "восемнадцатое", "девятнадцатое"]
_ORD_N_TENS = {20: "двадцатое", 30: "тридцатое"}
_ORD_G = ["", "первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого",
          "восьмого", "девятого", "десятого", "одиннадцатого", "двенадцатого", "тринадцатого",
          "четырнадцатого", "пятнадцатого", "шестнадцатого", "семнадцатого", "восемнадцатого",
          "девятнадцатого"]
_ORD_G_TENS = ["", "", "двадцатого", "тридцатого", "сорокового", "пятидесятого", "шестидесятого",
               "семидесятого", "восьмидесятого", "девяностого"]

# Валюта: (формы основной единицы, формы разменной, разменная женского рода)
_CURRENCIES = {
    "rub": (("рубль", "рубля", "рублей"), ("копейка", "копейки", "копеек"), True),
    "usd": (("доллар", "доллара", "долларов"), ("цент", "цента", "центов"), False),
    "eur": (("евро", "евро", "евро"), ("цент", "цента", "центов"), False),
}

_SP = "[   ]"
_INT = rf"\d{{1,3}}(?:{_SP}\d{{3}})+|\d+"
_AMOUNT = rf"(?<![\d.,])({_INT})(?:,(\d{{1,2}})|\.(\d{{2}}))?(?![\d])"
_CUR_AFTER = re.compile(
    _AMOUNT + rf"{_SP}?(₽|руб(?:лей|ля|ль|\.)?(?![а-яё])|р\.|\$|USD\b|€|EUR\b)", re.IGNORECASE)
_CUR_BEFORE = re.compile(rf"(\$|€){_SP}?" + _AMOUNT)
_DATE = re.compile(r"(?<![\d.,])(\d{1,2})\.(\d{2})(?:\.(\d{4}))?(?:\s?г\.)?(?![\d]|[.,]\d)")
_PERCENT = re.compile(rf"(?<![\d.,])({_INT}){_SP}?%")
_GROUPED = re.compile(rf"(?<![\d.,])(\d{{1,3}}(?:{_SP}\d{{3}})+)(?![\d]|[.,]\d)")
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Форма слова после числа n: (1 рубль, 2 рубля, 5 рублей)."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _triad(n: int, feminine: bool) -> list[str]:
    units = _UNITS_F if feminine else _UNITS_M
    out = [_HUNDREDS[n // 100]]
    rest = n % 100
    if 10 <= rest <= 19:
        out.append(_TEENS[rest - 10])
    else:
        out += [_TENS[rest // 10], units[rest % 10]]
    return [w for w in out if w]


def number_words(n: int, feminine: bool = False) -> str:
    """Целое число словами: 1049 → «одна тысяча сорок девять» (feminine — «одна/две»)."""
    if n == 0:
        return "ноль"
    if n < 0:
        return "минус " + number_words(-n, feminine)
    words: list[str] = []
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if not g:
            continue
        if i == 0:
            words += _triad(g, feminine)
        else:
            forms, fem = _SCALES[i - 1] if i - 1 < len(_SCALES) else _SCALES[-1]
            # «тысяча пятьсот», а не «одна тысяча пятьсот» — как говорят вслух
            words += ([] if g == 1 else _triad(g, fem)) + [plural(g, forms)]
    return " ".join(words)


def _day_words(d: int) -> str:
    if d < 20:
        return _ORD_N[d]
    tens, unit = divmod(d, 10)
    return _ORD_N_TENS[tens * 10] if not unit else f"{_TENS[tens]} {_ORD_N[unit]}"


def _year_words(y: int) -> str:
    """2026 → «две тысячи двадцать шестого» (порядковое, род. падеж)."""
    rest = y % 100
    if rest == 0:
        head = number_words(y - y % 1000) if y % 1000 else ""
        hundreds = (y % 1000) // 100
        if hundreds:
            last = {1: "сотого", 2: "двухсотого", 3: "трёхсотого", 4: "четырёхсотого", 5: "пятисотого",
                    6: "шестисотого", 7: "семисотого", 8: "восьмисотого", 9: "девятисотого"}[hundreds]
            return f"{head} {last}".strip()
        thousands = y // 1000
        return {1: "тысячного", 2: "двухтысячного", 3: "трёхтысячного"}.get(thousands, number_words(y) + "-го")
    head = number_words(y - rest) if y - rest else ""
    if rest < 20:
        tail = _ORD_G[rest]
    elif rest % 10 == 0:
        tail = _ORD_G_TENS[rest // 10]
    else:
        tail = f"{_TENS[rest // 10]} {_ORD_G[rest % 10]}"
    return f"{head} {tail}".strip()


def _to_int(s: str) -> int:
    return int(re.sub(_SP, "", s))


def _money(whole: str, frac: str | None, cur: str) -> str:
    major, minor, minor_fem = _CURRENCIES[cur]
    n = _to_int(whole)
    out = f"{number_words(n)} {plural(n, major)}"
    if frac:
        cents = int(frac.ljust(2, "0"))
        if cents:
            out += f" {number_words(cents, minor_fem)} {plural(cents, minor)}"
    return out


def _cur_key(sym: str) -> str:
    s = sym.lower()
    if s in ("$", "usd"):
        return "usd"
    if s in ("€", "eur"):
        return "eur"
    return "rub"


def _date_sub(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    out = f"{_day_words(d)} {_MONTHS[mo - 1]}"
    if y:
        out += f" {_year_words(int(y))} года"
    return out


def normalize_numbers(text: str) -> str:
    """Даты (01.11, 01.11.2026), суммы (1 049,11 ₽, $12), проценты и числа с разрядами
    (18 633) — словами. Только для русского текста: без кириллицы возвращает как есть."""
    if not text or not _CYRILLIC.search(text):
        return text
    t = _CUR_AFTER.sub(lambda m: _money(m.group(1), m.group(2) or m.group(3), _cur_key(m.group(4))), text)
    t = _CUR_BEFORE.sub(lambda m: _money(m.group(2), m.group(3) or m.group(4), _cur_key(m.group(1))), t)
    t = _DATE.sub(_date_sub, t)
    t = _PERCENT.sub(lambda m: f"{number_words(_to_int(m.group(1)))} "
                               f"{plural(_to_int(m.group(1)), ('процент', 'процента', 'процентов'))}", t)
    t = _GROUPED.sub(lambda m: number_words(_to_int(m.group(1))), t)
    return t
