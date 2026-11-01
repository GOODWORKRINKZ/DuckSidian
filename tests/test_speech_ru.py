"""Числа для русского TTS: даты, суммы, проценты — словами (bot.speech_ru)."""
from __future__ import annotations

from bot.media_ai import EXTRA_RUSSIAN_VOICES
from bot.speech_ru import normalize_numbers, number_words, plural


def test_number_words_and_plural() -> None:
    assert number_words(0) == "ноль"
    assert number_words(1049) == "тысяча сорок девять"
    assert number_words(84385) == "восемьдесят четыре тысячи триста восемьдесят пять"
    assert number_words(2_000_000) == "два миллиона"
    assert number_words(21, feminine=True) == "двадцать одна"
    forms = ("рубль", "рубля", "рублей")
    assert [plural(n, forms) for n in (1, 2, 5, 11, 21, 112)] == \
        ["рубль", "рубля", "рублей", "рублей", "рубль", "рублей"]


def test_dates_read_as_dates_not_decimals() -> None:
    # Регресс: «01.11» звучало как «одна целая одиннадцать сотых».
    assert normalize_numbers("Итог за 01.11") == "Итог за первое ноября"
    assert normalize_numbers("Записала: 31.10 · спортзал") == "Записала: тридцать первое октября · спортзал"
    assert normalize_numbers("срок 11.11.2026 г.") == "срок одиннадцатое ноября две тысячи двадцать шестого года"
    assert normalize_numbers("на 05.01.2020") == "на пятое января две тысячи двадцатого года"


def test_money_in_words_with_kopecks() -> None:
    assert normalize_numbers("Итог: 1 049,11 ₽") == "Итог: тысяча сорок девять рублей одиннадцать копеек"
    assert normalize_numbers("абонемент 15 825,5 ₽") == \
        "абонемент пятнадцать тысяч восемьсот двадцать пять рублей пятьдесят копеек"
    assert normalize_numbers("Таня 0 ₽, Денис 300 руб.") == "Таня ноль рублей, Денис триста рублей"
    assert normalize_numbers("оплатил $12.50 и 3 €") == "оплатил двенадцать долларов пятьдесят центов и три евро"


def test_percent_and_grouped_numbers() -> None:
    assert normalize_numbers("Маша 64%, Кирилл 21%") == "Маша шестьдесят четыре процента, Кирилл двадцать один процент"
    assert normalize_numbers("всего 18 633 шагов") == "всего восемнадцать тысяч шестьсот тридцать три шагов"


def test_leaves_other_text_alone() -> None:
    for s in ("версия 1.2.3, коэффициент 2,5", "пиньинь 1/2, 二 0/2", "3 траты", "01.11 hello", "我的朋友很好"):
        assert normalize_numbers(s) == s


def test_jovialman_in_russian_voice_list() -> None:
    assert "English_Jovialman" in EXTRA_RUSSIAN_VOICES
