"""Selects which short clip from the project's own training chunks to use as
GPT-SoVITS's "prompt audio" at inference time (DESIGN.md S8.3 step 1). Confirmed via a
real OSError on the first real generation attempt: TTS_infer_pack enforces a hard 3-10
second range on ref_audio_path and rejects anything else outright - the user's full
reference recording (any length, per the original requirements) is never valid there on
its own. One of the ~2-10s chunks preprocessing already produces (core/audio/chunk.py)
almost always has at least one candidate in range.
"""

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

MIN_PROMPT_SECONDS = 3.0
MAX_PROMPT_SECONDS = 10.0


@dataclass(frozen=True)
class PromptCandidate:
    audio_path: Path
    text: str
    duration: float


def select_prompt_clip(candidates: list[PromptCandidate]) -> PromptCandidate:
    in_range = [c for c in candidates if MIN_PROMPT_SECONDS <= c.duration <= MAX_PROMPT_SECONDS]
    if not in_range:
        raise ValueError(
            f"No candidate clip falls within GPT-SoVITS's required "
            f"{MIN_PROMPT_SECONDS:.0f}-{MAX_PROMPT_SECONDS:.0f}s prompt-audio range "
            "(confirmed via a real OSError at inference time, not a guess - see "
            "DESIGN.md S13). Re-run preprocessing on a reference clip that has at "
            "least one usable chunk in that range."
        )
    # Longest in-range clip: more audio likely means more speaker-identity signal for
    # zero-shot conditioning, up to the limit GPT-SoVITS itself enforces. A reasonable
    # default, not a research-backed optimum - revisit if real output quality suggests
    # otherwise.
    return max(in_range, key=lambda c: c.duration)


def load_candidates_from_dataset_list(list_path: Path, duration_fn=None) -> list[PromptCandidate]:
    """Parses GPT-SoVITS's own dataset.list format (core/train/dataset.py:
    `audio_path|speaker_name|language|text`) and measures each clip's real duration."""
    duration_fn = duration_fn or _audio_duration
    candidates = []
    for line in list_path.read_text().splitlines():
        if not line.strip():
            continue
        audio_path_str, _speaker, _language, text = line.split("|", 3)
        audio_path = Path(audio_path_str)
        candidates.append(PromptCandidate(audio_path=audio_path, text=text, duration=duration_fn(audio_path)))
    return candidates


def _audio_duration(audio_path: Path) -> float:
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate
