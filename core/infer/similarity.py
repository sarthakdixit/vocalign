"""Speaker-embedding cosine similarity (SECS) via Resemblyzer - the lighter-weight
option named in RESEARCH.md Finding 4 (vs. ECAPA-TDNN/WavLM-SV). Compares a candidate's
embedding against the centroid of the reference clip's own embedding(s), which is more
stable than comparing against a single reference segment when several exist.
"""

import numpy as np


def compute_embedding(samples: np.ndarray, sample_rate: int, encoder=None, preprocess_fn=None) -> np.ndarray:
    encoder = encoder or _default_encoder()
    preprocess_fn = preprocess_fn or _preprocess
    wav = preprocess_fn(samples, sample_rate)
    return encoder.embed_utterance(wav)


def reference_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    centroid = np.stack(embeddings).mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _preprocess(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    # Resemblyzer expects 16kHz audio; verify this against the real package on first
    # use (preprocess_wav's exact call signature is not independently confirmed the
    # way the GPT-SoVITS integration was - low blast radius if wrong, isolated here).
    from resemblyzer import preprocess_wav

    return preprocess_wav(samples, source_sr=sample_rate)


def _default_encoder():
    from resemblyzer import VoiceEncoder

    return VoiceEncoder()
