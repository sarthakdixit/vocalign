"""Splits target text into sentence-level chunks for independent per-chunk candidate
synthesis + ranking (core/infer/ranking.py) - NOT a workaround for a model input-length
limit. Confirmed against the real repo: GPT-SoVITS's own TTS.run() has no hard max and
splits/concatenates internally if needed. We chunk anyway so a bad candidate in one
sentence doesn't force discarding (or silently keeping) an otherwise-good whole-text
generation - each sentence gets its own gate-and-rank pass.

Known limitation: this is a simple heuristic splitter, not a real sentence tokenizer -
it can't distinguish an abbreviation ending in "." (e.g. "Dr.") from a real sentence
boundary before a capital letter. Mitigated (not eliminated) by running
text_normalize.normalize_text() first, which expands the common abbreviations it knows
about before this ever sees the text.
"""

import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_into_chunks(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = _SENTENCE_BOUNDARY_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]
