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

Very short chunks are merged into a neighbor rather than left standalone: confirmed via
a real multi-sentence generation that an isolated short sentence (a standalone "Thank
you." closing line, in that case) is a known weak spot for both the AR decoder's
stop-prediction and Whisper's WER-gate transcription on very short/sparse audio - it
failed the WER gate outright (WER=1.00) while 32 of the other 33, longer chunks in the
same real run passed cleanly. MIN_CHUNK_WORDS is a reasonable heuristic chosen to catch
cases like that, not a research-backed optimum.
"""

import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

MIN_CHUNK_WORDS = 3


def split_into_chunks(text: str, min_chunk_words: int = MIN_CHUNK_WORDS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = _SENTENCE_BOUNDARY_RE.split(text)
    chunks = [p.strip() for p in pieces if p.strip()]
    return _merge_short_chunks(chunks, min_chunk_words)


def _merge_short_chunks(chunks: list[str], min_words: int) -> list[str]:
    if len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1].split()) < min_words:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)

    # A short TRAILING chunk never gets a "next" chunk to merge forward into in the
    # loop above - without this, it would be left standalone, exactly the case this
    # whole function exists to avoid. Merge it backward into the previous one instead.
    if len(merged) > 1 and len(merged[-1].split()) < min_words:
        merged[-2] = f"{merged[-2]} {merged[-1]}"
        merged.pop()
    return merged
