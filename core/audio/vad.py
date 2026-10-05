"""Speech-segment detection (silero-vad), silence trimming, and audio quality gating."""

from dataclasses import dataclass, field

import numpy as np

HARD_MIN_SPEECH_SECONDS = 3.0
WARN_MIN_SPEECH_SECONDS = 6.0
WARN_MIN_SPEECH_RATIO = 0.60


@dataclass(frozen=True)
class QualityReport:
    total_speech_seconds: float
    speech_ratio: float
    blocking: bool
    warnings: list[str] = field(default_factory=list)


def detect_speech_segments(samples: np.ndarray, sample_rate: int, detector=None) -> list[tuple[float, float]]:
    detector = detector or _default_detector
    return detector(samples, sample_rate)


def trim_silence(samples: np.ndarray, sample_rate: int, segments: list[tuple[float, float]]) -> np.ndarray:
    if not segments:
        return samples[:0]
    start_sample = int(segments[0][0] * sample_rate)
    end_sample = int(segments[-1][1] * sample_rate)
    return samples[start_sample:end_sample]


def total_speech_seconds(segments: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in segments)


def speech_ratio(segments: list[tuple[float, float]], total_duration_seconds: float) -> float:
    if total_duration_seconds <= 0:
        return 0.0
    return total_speech_seconds(segments) / total_duration_seconds


def evaluate_quality(segments: list[tuple[float, float]], total_duration_seconds: float) -> QualityReport:
    speech_seconds = total_speech_seconds(segments)
    ratio = speech_ratio(segments, total_duration_seconds)
    warnings: list[str] = []

    blocking = speech_seconds < HARD_MIN_SPEECH_SECONDS
    if blocking:
        warnings.append(
            f"Only {speech_seconds:.1f}s of detected speech - below the {HARD_MIN_SPEECH_SECONDS:.0f}s minimum."
        )
    elif speech_seconds < WARN_MIN_SPEECH_SECONDS:
        warnings.append(
            f"Only {speech_seconds:.1f}s of detected speech - below the "
            f"{WARN_MIN_SPEECH_SECONDS:.0f}s recommended minimum."
        )

    if ratio < WARN_MIN_SPEECH_RATIO:
        warnings.append(
            f"Speech makes up only {ratio:.0%} of the clip - below the "
            f"{WARN_MIN_SPEECH_RATIO:.0%} recommended ratio."
        )

    return QualityReport(
        total_speech_seconds=speech_seconds,
        speech_ratio=ratio,
        blocking=blocking,
        warnings=warnings,
    )


def _default_detector(samples: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    timestamps = get_speech_timestamps(torch.from_numpy(samples), model, sampling_rate=sample_rate)
    return [(t["start"] / sample_rate, t["end"] / sample_rate) for t in timestamps]
