"""New Project pipeline (DESIGN.md S8.1 / S9): transcode/normalize/VAD the uploaded
reference clip, verify it against the reference transcript with Whisper, chunk it for
training, and write GPT-SoVITS's own dataset.list - moving the project
CREATED -> PREPROCESSING -> PREPROCESSED (or -> ERROR). Exposed as one plain function
so the Gradio callback wiring in gui/app.py stays a thin adapter, testable here without
a running Gradio server.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.align.whisper_align import compare_to_reference, transcribe_and_segment
from core.audio.chunk import build_training_chunks, slice_audio
from core.audio.io import load_and_transcode, write_wav
from core.audio.normalize import normalize_loudness
from core.audio.vad import detect_speech_segments, evaluate_quality, trim_silence
from core.projects import manager, storage
from core.projects.models import ProjectState
from core.train.dataset import write_dataset_list
from core.train.recipe import select_recipe


@dataclass(frozen=True)
class PreprocessingOutcome:
    project_id: str
    ok: bool
    warnings: list[str] = field(default_factory=list)
    chunk_count: int = 0
    total_speech_seconds: float = 0.0
    transcript_match_ratio: float | None = None
    recipe_tier: str | None = None
    error: str | None = None


def create_and_preprocess_project(
    projects_root,
    name: str,
    reference_audio_path,
    reference_text: str,
    *,
    language: str = "en",
    ffmpeg_run=None,
    vad_detector=None,
    whisper_model=None,
) -> PreprocessingOutcome:
    project = manager.create_project(projects_root, name, language=language)
    try:
        return _preprocess(
            projects_root, project.id, reference_audio_path, reference_text,
            ffmpeg_run=ffmpeg_run, vad_detector=vad_detector, whisper_model=whisper_model,
        )
    except Exception as exc:
        manager.transition_project(projects_root, project.id, ProjectState.ERROR, error_message=str(exc))
        return PreprocessingOutcome(project_id=project.id, ok=False, error=str(exc))


def _preprocess(
    projects_root, project_id: str, reference_audio_path, reference_text: str,
    *, ffmpeg_run=None, vad_detector=None, whisper_model=None,
) -> PreprocessingOutcome:
    layout = storage.ensure_project_layout(projects_root, project_id)
    manager.transition_project(projects_root, project_id, ProjectState.PREPROCESSING)

    raw_path = layout["raw"] / f"reference{Path(reference_audio_path).suffix or '.wav'}"
    shutil.copy(reference_audio_path, raw_path)

    samples, sample_rate = load_and_transcode(raw_path, run=ffmpeg_run)
    samples = normalize_loudness(samples, sample_rate)

    speech_segments = detect_speech_segments(samples, sample_rate, detector=vad_detector)
    quality = evaluate_quality(speech_segments, len(samples) / sample_rate)
    if quality.blocking:
        error = quality.warnings[0] if quality.warnings else "Not enough usable speech detected."
        manager.transition_project(projects_root, project_id, ProjectState.ERROR, error_message=error)
        return PreprocessingOutcome(
            project_id=project_id, ok=False, warnings=quality.warnings,
            total_speech_seconds=quality.total_speech_seconds, error=error,
        )

    trimmed = trim_silence(samples, sample_rate, speech_segments)
    trimmed_path = layout["processed"] / "trimmed.wav"
    write_wav(trimmed_path, trimmed, sample_rate)

    whisper_segments = transcribe_and_segment(trimmed_path, model=whisper_model)
    match = compare_to_reference(whisper_segments, reference_text)
    warnings = list(quality.warnings)
    if match.likely_mismatched:
        warnings.append(
            "The reference text doesn't look like it matches what was heard in the "
            "audio - double-check both before training."
        )

    chunks = build_training_chunks(whisper_segments)
    chunk_audio_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = layout["segments"] / f"chunk_{i:03d}.wav"
        write_wav(chunk_path, slice_audio(trimmed, sample_rate, chunk), sample_rate)
        chunk_audio_paths.append((chunk, chunk_path))

    dataset_list_path = layout["processed"] / "dataset.list"
    dataset_list_path, skipped = write_dataset_list(dataset_list_path, chunk_audio_paths, project_id)
    if skipped:
        warnings.append(f"{skipped} low-confidence chunk(s) were excluded from training data.")

    total_speech_seconds = sum(c.duration for c in chunks)
    recipe = select_recipe(total_speech_seconds)

    manager.update_config(
        projects_root,
        project_id,
        {
            "reference_text": reference_text,
            "raw_audio_path": str(raw_path),
            "dataset_list_path": str(dataset_list_path),
            "total_speech_seconds": total_speech_seconds,
            "transcript_match_ratio": match.similarity_ratio,
            "recipe_tier": recipe.tier,
        },
    )
    manager.transition_project(projects_root, project_id, ProjectState.PREPROCESSED)

    return PreprocessingOutcome(
        project_id=project_id,
        ok=True,
        warnings=warnings,
        chunk_count=len(chunks),
        total_speech_seconds=total_speech_seconds,
        transcript_match_ratio=match.similarity_ratio,
        recipe_tier=recipe.tier,
    )
