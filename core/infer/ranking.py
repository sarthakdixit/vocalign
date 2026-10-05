"""Two-stage candidate ranking for generated audio (RESEARCH.md Finding 4): a
Whisper-WER hard gate discards outright failures (garbled/dropped/repeated words)
that similarity scores alone can miss, then survivors are ranked by a weighted
SECS (speaker similarity, weighted higher) + UTMOS (naturalness, weighted lower)
score. If every candidate fails the gate, the least-bad one (lowest WER) is still
returned rather than nothing, since the pipeline needs *some* output.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.align.whisper_align import normalize_for_comparison, transcribe_and_segment
from core.audio.io import write_wav

WER_REJECT_THRESHOLD = 0.3
SECS_WEIGHT = 0.7
UTMOS_WEIGHT = 0.3
UTMOS_MAX = 5.0


@dataclass(frozen=True)
class Candidate:
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    wer: float
    secs: float | None
    utmos: float | None
    rejected: bool

    @property
    def combined_score(self) -> float:
        secs = self.secs if self.secs is not None else 0.0
        utmos_normalized = (self.utmos / UTMOS_MAX) if self.utmos is not None else 0.0
        return SECS_WEIGHT * secs + UTMOS_WEIGHT * utmos_normalized


def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = normalize_for_comparison(reference).split()
    hyp_words = normalize_for_comparison(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def transcribe_candidate(candidate: Candidate, whisper_model=None) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "candidate.wav"
        write_wav(tmp_path, candidate.samples, candidate.sample_rate)
        segments = transcribe_and_segment(tmp_path, model=whisper_model)
        return " ".join(segment.text for segment in segments)


def score_candidate(
    candidate: Candidate,
    reference_text: str,
    *,
    whisper_model=None,
    secs_fn=None,
    utmos_fn=None,
    wer_reject_threshold: float = WER_REJECT_THRESHOLD,
) -> ScoredCandidate:
    heard = transcribe_candidate(candidate, whisper_model=whisper_model)
    wer = compute_wer(heard, reference_text)
    rejected = wer > wer_reject_threshold

    secs = secs_fn(candidate) if (secs_fn is not None and not rejected) else None
    utmos = utmos_fn(candidate) if (utmos_fn is not None and not rejected) else None

    return ScoredCandidate(candidate=candidate, wer=wer, secs=secs, utmos=utmos, rejected=rejected)


def pick_best(scored_candidates: list[ScoredCandidate]) -> ScoredCandidate | None:
    if not scored_candidates:
        return None
    survivors = [c for c in scored_candidates if not c.rejected]
    if survivors:
        return max(survivors, key=lambda c: c.combined_score)
    return min(scored_candidates, key=lambda c: c.wer)


def default_secs_fn(reference_embedding: np.ndarray, encoder=None):
    from core.infer.similarity import compute_embedding, cosine_similarity

    def _fn(candidate: Candidate) -> float:
        embedding = compute_embedding(candidate.samples, candidate.sample_rate, encoder=encoder)
        return cosine_similarity(embedding, reference_embedding)

    return _fn


def default_utmos_fn(scorer=None):
    from core.infer.naturalness import predict_mos

    def _fn(candidate: Candidate) -> float:
        return predict_mos(candidate.samples, candidate.sample_rate, scorer=scorer)

    return _fn


def _levenshtein(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(b) + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[len(b)]
