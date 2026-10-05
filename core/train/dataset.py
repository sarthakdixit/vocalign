"""Builds GPT-SoVITS's own dataset list file from our training chunks. Format confirmed
directly against the real repo (1-get-text.py: `wav_name, spk_name, language, text =
line.split("|")`): one line per clip, `audio_path|speaker_name|language|text`.
"""

from pathlib import Path

from core.audio.chunk import TrainingChunk

LANGUAGE_CODES = {"en", "zh", "ja", "ko", "yue"}


def write_dataset_list(
    list_path: Path,
    chunk_audio_paths: list[tuple[TrainingChunk, Path]],
    speaker_name: str,
    language: str = "en",
    include_low_confidence: bool = False,
) -> tuple[Path, int]:
    if language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language {language!r} - GPT-SoVITS supports {sorted(LANGUAGE_CODES)}")

    usable = [
        (chunk, audio_path)
        for chunk, audio_path in chunk_audio_paths
        if include_low_confidence or not chunk.low_confidence
    ]
    skipped = len(chunk_audio_paths) - len(usable)

    lines = [f"{audio_path}|{speaker_name}|{language}|{_escape(chunk.text)}" for chunk, audio_path in usable]
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return list_path, skipped


def _escape(text: str) -> str:
    return text.replace("|", " ").replace("\n", " ").strip()
