"""Exports generated audio to disk with a generation-metadata JSON sidecar, and makes
a best-effort attempt to tag the audio file itself as AI-generated in its own comment
field. This is the near-zero-cost half of the watermarking decision (RESEARCH.md
Finding 7: full watermarking deferred, revisit only if this tool is ever shared or
distributed) - tagging costs nothing now, so it happens regardless.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

AI_GENERATED_COMMENT = "AI-generated voice clone output (clone-voice)"

_FORMAT_BY_SUFFIX = {".wav": "WAV", ".mp3": "MP3"}


@dataclass(frozen=True)
class GenerationMetadata:
    text: str
    secs_score: float | None = None
    utmos_score: float | None = None
    low_confidence: bool = False
    extra: dict = field(default_factory=dict)


def export_audio(path: Path, samples: np.ndarray, sample_rate: int, metadata: GenerationMetadata) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_with_comment(path, samples, sample_rate)

    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps(_metadata_to_dict(metadata), indent=2))
    return path, metadata_path


def _write_with_comment(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    fmt = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    try:
        with sf.SoundFile(str(path), mode="w", samplerate=sample_rate, channels=channels, format=fmt) as f:
            try:
                f.comment = AI_GENERATED_COMMENT
            except Exception:
                pass
            f.write(samples)
    except Exception:
        # The export itself must never fail just because comment-tagging isn't
        # supported for this format/library build - fall back to the plain writer.
        sf.write(str(path), samples, sample_rate)


def _metadata_to_dict(metadata: GenerationMetadata) -> dict:
    return {
        "text": metadata.text,
        "secs_score": metadata.secs_score,
        "utmos_score": metadata.utmos_score,
        "low_confidence": metadata.low_confidence,
        **metadata.extra,
    }
