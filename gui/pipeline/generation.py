"""Generate page pipeline (DESIGN.md S8.3 / S9): resolves a trained project's
checkpoint + auto-selected prompt clip, loads (and caches) the TTS pipeline, and runs
one generate_and_export() call - writing output + metadata under the project's own
output/ dir.

Caches the loaded TTS pipeline + SECS reference embedding per project, since reloading
real models on every single generation would be wasteful - but evicts the previous
entry (and frees CUDA memory) whenever a DIFFERENT project is requested, since this
hardware's VRAM is tight enough that holding two projects' models loaded at once isn't
realistic (DESIGN.md S13). Caching logic (load_inference_context) and real model
construction (_build_inference_context) are split apart specifically so the caching
behavior is testable with a trivial fake builder, without needing real models.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core.infer.generate import generate_and_export
from core.projects import manager, storage


@dataclass(frozen=True)
class GenerationOutcome:
    ok: bool
    audio_path: str | None = None
    metadata_path: str | None = None
    secs_score: float | None = None
    utmos_score: float | None = None
    low_confidence: bool = False
    error: str | None = None


@dataclass
class _ProjectInferenceContext:
    project_id: str
    tts_instance: object
    prompt: object  # a core.infer.prompt_selection.PromptCandidate, kept loosely typed
    # here to avoid this module needing the real class just for a type hint in tests.
    secs_fn: object
    utmos_fn: object


_cache: _ProjectInferenceContext | None = None


def _evict_cache() -> None:
    global _cache
    if _cache is None:
        return
    _cache = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_inference_context(projects_root, project_id: str, *, build_fn=None) -> _ProjectInferenceContext:
    global _cache
    build_fn = build_fn or _build_inference_context
    if _cache is not None and _cache.project_id == project_id:
        return _cache
    _evict_cache()
    _cache = build_fn(projects_root, project_id)
    return _cache


def _build_inference_context(projects_root, project_id: str) -> _ProjectInferenceContext:
    """Real construction: not exercised directly by tests (injected via build_fn
    instead) - this is straightline wiring of already-individually-tested pieces
    (tts_adapter, prompt_selection, similarity, ranking), the same kind of code
    scripts/smoke_test_batch4.py already proved out for real."""
    from core import config
    from core.audio.io import load_and_transcode
    from core.infer.prompt_selection import load_candidates_from_dataset_list, select_prompt_clip
    from core.infer.ranking import default_secs_fn, default_utmos_fn
    from core.infer.similarity import compute_embedding, reference_centroid
    from core.infer.tts_adapter import TtsCheckpoints, load_tts
    from core.train.data_prep import default_bert_dir, default_cnhubert_dir
    from core.train.gpt_sovits_adapter import chdir_to_repo_root, default_paths, ensure_on_sys_path

    project = manager.get_project(projects_root, project_id)
    dataset_list_path = Path(project.config["dataset_list_path"])
    prompt = select_prompt_clip(load_candidates_from_dataset_list(dataset_list_path))

    version = project.config.get("gpt_sovits_version", config.GPT_SOVITS_VERSION)
    paths = default_paths(config.VENDOR_DIR, sys.executable, version)
    checkpoint_path = project.config.get("checkpoint_path")
    t2s_weights_path = Path(checkpoint_path) if checkpoint_path else paths.pretrained_s1

    checkpoints = TtsCheckpoints(
        version=version,
        t2s_weights_path=t2s_weights_path,
        vits_weights_path=paths.pretrained_s2_g,
        bert_base_path=default_bert_dir(config.VENDOR_DIR),
        cnhubert_base_path=default_cnhubert_dir(config.VENDOR_DIR),
    )
    ensure_on_sys_path(paths)
    with chdir_to_repo_root(paths):
        tts_instance = load_tts(checkpoints)

    ref_samples, ref_sr = load_and_transcode(Path(project.config["raw_audio_path"]))
    ref_embedding = compute_embedding(ref_samples, ref_sr)

    return _ProjectInferenceContext(
        project_id=project_id,
        tts_instance=tts_instance,
        prompt=prompt,
        secs_fn=default_secs_fn(reference_centroid([ref_embedding])),
        utmos_fn=default_utmos_fn(),
    )


def run_generation(
    projects_root, project_id: str, target_text: str, *,
    on_progress=None, load_context_fn=None, whisper_model=None,
) -> GenerationOutcome:
    load_context_fn = load_context_fn or load_inference_context
    try:
        context = load_context_fn(projects_root, project_id)
    except Exception as exc:
        return GenerationOutcome(ok=False, error=str(exc))

    layout = storage.ensure_project_layout(projects_root, project_id)
    # monotonic_ns(), not time.time(): millisecond resolution can collide between two
    # generations fired in quick succession, silently overwriting the first one's file.
    output_path = layout["output"] / f"gen_{time.monotonic_ns()}.wav"

    try:
        _audio_path, metadata_path = generate_and_export(
            target_text,
            output_path,
            tts_instance=context.tts_instance,
            text_lang="en",
            ref_audio_path=context.prompt.audio_path,
            prompt_text=context.prompt.text,
            prompt_lang="en",
            secs_fn=context.secs_fn,
            utmos_fn=context.utmos_fn,
            whisper_model=whisper_model,
            on_progress=on_progress or (lambda message: None),
        )
    except Exception as exc:
        return GenerationOutcome(ok=False, error=str(exc))

    return _outcome_from_metadata_file(metadata_path)


def list_generation_history(projects_root, project_id: str) -> list[GenerationOutcome]:
    layout = storage.ensure_project_layout(projects_root, project_id)
    metadata_paths = sorted(layout["output"].glob("*.wav.json"), reverse=True)
    return [_outcome_from_metadata_file(path) for path in metadata_paths]


def _outcome_from_metadata_file(metadata_path: Path) -> GenerationOutcome:
    metadata_path = Path(metadata_path)
    audio_path = metadata_path.with_suffix("")  # "x.wav.json" -> "x.wav"
    metadata = json.loads(metadata_path.read_text())
    return GenerationOutcome(
        ok=True,
        audio_path=str(audio_path),
        metadata_path=str(metadata_path),
        secs_score=metadata.get("secs_score"),
        utmos_score=metadata.get("utmos_score"),
        low_confidence=metadata.get("low_confidence", False),
    )
