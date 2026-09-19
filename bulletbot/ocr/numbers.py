from __future__ import annotations

import re


_GROUPED_INTEGER_PATTERN = re.compile(r"\d{1,3}(?:[,.]\d{3})+")
_INTEGER_PAIR_PATTERN = re.compile(
    r"(?<![\d,.])(\d+(?:\s*[,.]\s*\d+)*)\s*/\s*"
    r"(\d+(?:\s*[,.]\s*\d+)*)(?![\d,.])"
)


def normalized_ocr_integer_text(text: str | None) -> str:
    if not text:
        return ""
    return "".join(
        character
        for character in text.strip()
        if character.isdigit() or character in ",."
    )


def parse_ocr_integer(text: str | None) -> int | None:
    """Parse an integer whose OCR thousands separators may be dots or commas."""

    numeric = normalized_ocr_integer_text(text)
    if not numeric:
        return None
    if numeric.isdigit():
        return int(numeric)
    if not _GROUPED_INTEGER_PATTERN.fullmatch(numeric):
        return None
    return int(numeric.replace(",", "").replace(".", ""))


def has_valid_ocr_thousands_grouping(text: str | None) -> bool:
    numeric = normalized_ocr_integer_text(text)
    return bool(_GROUPED_INTEGER_PATTERN.fullmatch(numeric))


def parse_ocr_integer_pair(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    match = _INTEGER_PAIR_PATTERN.search(text)
    if match is None:
        return None
    first = parse_ocr_integer(match.group(1))
    second = parse_ocr_integer(match.group(2))
    if first is None or second is None:
        return None
    return first, second
