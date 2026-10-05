"""Light text normalization before handing target text to synthesis: numbers to words
(GPT-SoVITS's own phonemizer handles digits inconsistently, so normalizing first is
cheap insurance) and common abbreviation expansion. English only (confirmed project
scope) - not a general-purpose NLP normalizer, deliberately kept small.
"""

import re

_ABBREVIATIONS = {
    "mr.": "mister",
    "mrs.": "missus",
    "ms.": "miz",
    "dr.": "doctor",
    "st.": "street",
    "vs.": "versus",
    "etc.": "et cetera",
}

_ABBREVIATION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)),
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_text(text: str) -> str:
    text = _expand_abbreviations(text)
    text = _expand_numbers(text)
    text = _collapse_whitespace(text)
    return text.strip()


def _expand_abbreviations(text: str) -> str:
    return _ABBREVIATION_RE.sub(lambda m: _ABBREVIATIONS[m.group(0).lower()], text)


def _expand_numbers(text: str) -> str:
    from num2words import num2words

    def replace(match: re.Match) -> str:
        value = match.group(0).replace(",", "")
        try:
            return num2words(float(value)) if "." in value else num2words(int(value))
        except ValueError:
            return match.group(0)

    return _NUMBER_RE.sub(replace, text)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text)
