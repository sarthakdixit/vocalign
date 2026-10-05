"""Stitches independently-synthesized audio chunks into one continuous clip with a
short linear crossfade at each join, so chunk boundaries aren't audibly abrupt clicks.
"""

import numpy as np

DEFAULT_CROSSFADE_MS = 30.0


def stitch(chunks: list[np.ndarray], sample_rate: int, crossfade_ms: float = DEFAULT_CROSSFADE_MS) -> np.ndarray:
    chunks = [c for c in chunks if c.size > 0]
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = chunks[0].astype(np.float32)
    for chunk in chunks[1:]:
        chunk = chunk.astype(np.float32)
        fade_len = min(crossfade_samples, len(result), len(chunk))
        if fade_len <= 0:
            result = np.concatenate([result, chunk])
            continue
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        overlapped = result[-fade_len:] * fade_out + chunk[:fade_len] * fade_in
        result = np.concatenate([result[:-fade_len], overlapped, chunk[fade_len:]])
    return result
