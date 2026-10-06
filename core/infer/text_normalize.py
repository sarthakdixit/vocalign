"""Light text normalization before handing target text to synthesis: numbers to words
(GPT-SoVITS's own phonemizer handles digits inconsistently, so normalizing first is
cheap insurance), common abbreviation expansion, and markdown stripping. English only
(confirmed project scope) - not a general-purpose NLP normalizer, deliberately kept
small.
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

# Paired markdown syntax, unwrapped to its inner text (order matters: bold/strikethrough
# before italic, so "**x**" isn't first misread as italic's "*...*").
_MARKDOWN_PAIR_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # **bold**
    (re.compile(r"__(.+?)__"), r"\1"),  # __bold__
    (re.compile(r"~~(.+?)~~"), r"\1"),  # ~~strikethrough~~
    (re.compile(r"`(.+?)`"), r"\1"),  # `code`
    (re.compile(r"\*(.+?)\*"), r"\1"),  # *italic*
    (re.compile(r"(?<!\w)_(.+?)_(?!\w)"), r"\1"),  # _italic_ (not snake_case words)
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),  # # Heading
    (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"),  # [text](url)
]

# Confirmed via a real run: leftover markdown symbols (e.g. pasted from a doc/chat
# with formatting intact) reach GPT-SoVITS's text frontend as literal characters
# never seen in any training data, which degraded one chunk's synthesis badly enough
# to fail the WER gate outright. Catches anything the paired patterns above didn't
# (an unpaired/malformed marker) after real pairs have already been unwrapped.
_LEFTOVER_MARKDOWN_RE = re.compile(r"[*_~`#]")


def normalize_text(text: str) -> str:
    text = _strip_markdown(text)
    text = _expand_abbreviations(text)
    text = _expand_numbers(text)
    text = _collapse_whitespace(text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    for pattern, replacement in _MARKDOWN_PAIR_PATTERNS:
        text = pattern.sub(replacement, text)
    return _LEFTOVER_MARKDOWN_RE.sub("", text)


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
