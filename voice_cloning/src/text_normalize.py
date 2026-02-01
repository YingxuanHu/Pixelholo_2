import re
from typing import Callable

from num2words import num2words

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer  # type: ignore
except Exception:
    Normalizer = None


_CURRENCY_RE = re.compile(r"(?P<sign>[$£€])\s?(?P<amount>\d[\d,]*)(?:\.(?P<cents>\d{1,2}))?")
_NUMBER_RE = re.compile(r"\d[\d,]*")

_ACRONYMS = {
    "LLM": "L L M",
    "API": "A P I",
    "AWS": "A W S",
    "GPU": "G P U",
    "AI": "A I",
    "SQL": "Sequel",
}

_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Dr.": "Doctor",
    "St.": "Street",
    "vs.": "versus",
}

_nemo_normalizer = None


def _get_nemo_normalizer():
    global _nemo_normalizer
    if _nemo_normalizer is None and Normalizer is not None:
        _nemo_normalizer = Normalizer(input_case="cased", lang="en")
    return _nemo_normalizer


def warmup_text_normalizer() -> None:
    _get_nemo_normalizer()


def _strip_commas(value: str) -> str:
    return value.replace(",", "")


def _to_words(value: str, *, to_year: bool = False) -> str:
    number = int(_strip_commas(value))
    if to_year:
        return num2words(number, to="year")
    return num2words(number)


def _replace_currency(match: re.Match) -> str:
    sign = match.group("sign")
    amount = match.group("amount") or "0"
    cents = match.group("cents")
    currency_word = {
        "$": "dollars",
        "£": "pounds",
        "€": "euros",
    }.get(sign, "dollars")
    main = _to_words(amount)
    if cents:
        cent_value = int(cents.ljust(2, "0"))
        cents_words = num2words(cent_value)
        return f"{main} {currency_word} and {cents_words} cents"
    return f"{main} {currency_word}"


def _replace_number(match: re.Match, year_hint: Callable[[int], bool]) -> str:
    raw = match.group()
    number = int(_strip_commas(raw))
    return _to_words(raw, to_year=year_hint(number))


def clean_text_for_tts(text: str) -> str:
    """
    Normalizes numbers to words so TTS pronounces them naturally.
    - $100 -> "one hundred dollars"
    - 1995 -> "nineteen ninety-five" (year style)
    """
    if not text:
        return text

    # Strip markdown bullets/asterisks so TTS doesn't read them aloud.
    text = re.sub(r"(^|\n)\s*[*+-]\s+", r"\1", text)
    text = text.replace("*", " ")

    # Address-style "St" -> "Street" (avoid "Saint" in addresses)
    text = re.sub(r"\b(\d+)\s+St\.?\b", r"\1 Street", text)
    text = re.sub(r"\bSt\.?(?=\s*,|\s*$)", " Street", text)

    # 1) Tech jargon first (so the normalizer doesn't alter them)
    for key, value in _ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(key)}\b", value, text)

    # 2) If NeMo is available, let it handle general TN/ITN
    normalizer = _get_nemo_normalizer()
    if normalizer is not None:
        return normalizer.normalize(text)

    # 3) Fallback: lightweight rules
    for key, value in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(key)}\b", value, text)

    text = (
        text.replace("%", " percent")
        .replace("&", " and ")
        .replace("+", " plus ")
        .replace("@", " at ")
        .replace("#", " number ")
    )

    # Decimals: "1.3" -> "1 point 3" (before currency/number rules)
    text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)

    text = _CURRENCY_RE.sub(_replace_currency, text)

    def year_hint(value: int) -> bool:
        return 1000 <= value <= 2099

    return _NUMBER_RE.sub(lambda m: _replace_number(m, year_hint), text)
