"""Whisper-based transcript verification and segmentation (not forced alignment - see DESIGN.md S3).

Whisper's own transcription becomes the per-chunk training text, since it's what's
actually time-aligned; the user's reference text is only used as a sanity check
that the uploaded audio/text pair actually match (compare_to_reference).
"""

import difflib
import re
from dataclasses import dataclass

LOW_CONFIDENCE_AVG_LOGPROB = -1.0
LOW_CONFIDENCE_NO_SPEECH_PROB = 0.6
LIKELY_MISMATCHED_BELOW_RATIO = 0.5


@dataclass(frozen=True)
class WhisperSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True)
class MatchReport:
    similarity_ratio: float
    asr_text: str
    reference_text: str

    @property
    def likely_mismatched(self) -> bool:
        return self.similarity_ratio < LIKELY_MISMATCHED_BELOW_RATIO


def transcribe_and_segment(audio_path, model=None, **transcribe_kwargs) -> list[WhisperSegment]:
    model = model or _default_model()
    segments, _info = model.transcribe(str(audio_path), word_timestamps=False, **transcribe_kwargs)
    return [
        WhisperSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            avg_logprob=segment.avg_logprob,
            no_speech_prob=segment.no_speech_prob,
        )
        for segment in segments
    ]


def is_low_confidence(segment: WhisperSegment) -> bool:
    return (
        segment.avg_logprob < LOW_CONFIDENCE_AVG_LOGPROB
        or segment.no_speech_prob > LOW_CONFIDENCE_NO_SPEECH_PROB
    )


def compare_to_reference(segments: list[WhisperSegment], reference_text: str) -> MatchReport:
    asr_text = _normalize_text(" ".join(segment.text for segment in segments))
    reference = _normalize_text(reference_text)
    ratio = difflib.SequenceMatcher(a=asr_text, b=reference).ratio() if reference else 0.0
    return MatchReport(similarity_ratio=ratio, asr_text=asr_text, reference_text=reference)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    stripped = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def _default_model():
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="auto", compute_type="auto")
