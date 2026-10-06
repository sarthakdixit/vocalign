"""Project Detail "Start Training" pipeline (DESIGN.md S8.2 / S9): selects a recipe
tier, runs GPT-SoVITS's own data-prep scripts, fine-tunes the GPT stage (or skips
training entirely at the zero_shot_only tier, where there's nothing to fine-tune),
then runs the post-training quality signal - moving the project
PREPROCESSED/ERROR -> TRAINING -> TRAINED (or -> ERROR).

The subprocess-heavy steps (data prep, the GPT stage) accept the same `popen_factory`
injection core.train.runner already supports, and the quality-signal step is injectable
as a whole (`quality_signal_fn`) - real model loading is exactly what Batch 4's smoke
testing already covered, so these tests can fake it without paying that cost again.

TrainingSession runs run_training() on a background thread so gui/app.py's Gradio
generator can poll for progress and yield UI updates without blocking the server;
cancellation is cooperative via a threading.Event, matching run_stage()'s own
cancel_check contract (checked between subprocess output lines - a step producing no
output for a long stretch won't respond to Cancel until its next line or completion).
"""

import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from core import config
from core.infer.quality_signal import QualityBand
from core.projects import manager, storage
from core.projects.models import ProjectState
from core.train.data_prep import DataPrepJob, default_bert_dir, default_cnhubert_dir, run_all_prep_steps
from core.train.gpt_sovits_adapter import FineTuneJob, default_paths, find_latest_gpt_checkpoint, run_gpt_stage
from core.train.recipe import select_recipe
from core.train.runner import RunStatus


@dataclass(frozen=True)
class TrainingOutcome:
    project_id: str
    ok: bool
    skipped_zero_shot: bool = False
    checkpoint_path: str | None = None
    quality_band: QualityBand | None = None
    error: str | None = None
    cancelled: bool = False


class TrainingSession:
    """Owns one project's training run on a background thread. gui/app.py creates one
    per "Start Training" click, polls `.drain_log()`/`.result` to stream progress, and
    calls `.cancel()` from a separate "Cancel" button's own click handler."""

    def __init__(self, projects_root, project_id: str, **run_training_kwargs):
        self.projects_root = projects_root
        self.project_id = project_id
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._cancel_event = threading.Event()
        self.result: TrainingOutcome | None = None
        self._run_training_kwargs = run_training_kwargs
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "TrainingSession":
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def done(self) -> bool:
        return self.result is not None

    def drain_log(self) -> list[str]:
        lines = []
        while True:
            try:
                lines.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def _run(self) -> None:
        self.result = run_training(
            self.projects_root,
            self.project_id,
            on_progress=self._log_queue.put,
            cancel_check=self._cancel_event.is_set,
            **self._run_training_kwargs,
        )


def run_training(
    projects_root,
    project_id: str,
    *,
    on_progress=None,
    cancel_check=None,
    popen_factory=None,
    run_data_prep_fn=None,
    run_gpt_stage_fn=None,
    quality_signal_fn=None,
    gpt_epochs_override=None,
) -> TrainingOutcome:
    on_progress = on_progress or (lambda line: None)
    cancel_check = cancel_check or (lambda: False)

    project = manager.get_project(projects_root, project_id)
    try:
        manager.transition_project(projects_root, project_id, ProjectState.TRAINING)
        return _train(
            projects_root, project, on_progress, cancel_check, popen_factory, run_data_prep_fn,
            run_gpt_stage_fn, quality_signal_fn, gpt_epochs_override,
        )
    except Exception as exc:
        on_progress(f"ERROR: {exc}")
        manager.transition_project(projects_root, project_id, ProjectState.ERROR, error_message=str(exc))
        return TrainingOutcome(project_id=project_id, ok=False, error=str(exc))


def _default_run_data_prep(paths, prep_job, *, on_progress, cancel_check, popen_factory):
    return run_all_prep_steps(paths, prep_job, on_progress=on_progress, cancel_check=cancel_check, popen_factory=popen_factory)


def _default_run_gpt_stage(paths, job, *, on_progress, cancel_check, popen_factory):
    return run_gpt_stage(paths, job, on_progress=on_progress, cancel_check=cancel_check, popen_factory=popen_factory)


def _train(
    projects_root, project, on_progress, cancel_check, popen_factory, run_data_prep_fn, run_gpt_stage_fn,
    quality_signal_fn, gpt_epochs_override,
) -> TrainingOutcome:
    run_data_prep_fn = run_data_prep_fn or _default_run_data_prep
    run_gpt_stage_fn = run_gpt_stage_fn or _default_run_gpt_stage
    quality_signal_fn = quality_signal_fn or run_post_training_quality_signal
    layout = storage.ensure_project_layout(projects_root, project.id)
    total_speech_seconds = project.config.get("total_speech_seconds", 0.0)
    recipe = select_recipe(total_speech_seconds)
    # A per-run experiment, not a permanent change to the recipe's own default: the
    # right epoch count for a given project/data size is genuinely unknown (recipe.py's
    # own counts are explicitly placeholders), and more epochs on the same data could
    # help generalization or just as easily overfit - this lets a specific run be
    # tried without committing every future project at this tier to the same value.
    gpt_epochs = gpt_epochs_override if gpt_epochs_override is not None else recipe.gpt_epochs
    override_note = f" (overridden from {recipe.gpt_epochs})" if gpt_epochs_override is not None else ""
    on_progress(
        f"Recipe: tier={recipe.tier} attempt_training={recipe.attempt_training} gpt_epochs={gpt_epochs}{override_note}"
    )

    if not recipe.attempt_training:
        on_progress("Not enough reference audio to fine-tune - using zero-shot conditioning only.")
        manager.update_config(projects_root, project.id, {"checkpoint_path": None})
        manager.transition_project(projects_root, project.id, ProjectState.TRAINED)
        return TrainingOutcome(project_id=project.id, ok=True, skipped_zero_shot=True)

    dataset_list_path = Path(project.config["dataset_list_path"])
    version = config.GPT_SOVITS_VERSION
    paths = default_paths(config.VENDOR_DIR, sys.executable, version)

    prep_job = DataPrepJob(
        dataset_list_path=dataset_list_path,
        audio_dir=layout["segments"],
        experiment_dir=layout["training"],
        bert_pretrained_dir=default_bert_dir(config.VENDOR_DIR),
        cnhubert_pretrained_dir=default_cnhubert_dir(config.VENDOR_DIR),
        version=version,
    )
    on_progress("Running GPT-SoVITS data prep...")
    prep_results = run_data_prep_fn(
        paths, prep_job, on_progress=on_progress, cancel_check=cancel_check, popen_factory=popen_factory
    )
    failed = next((r for r in prep_results if r.status is not RunStatus.COMPLETED), None)
    if failed is not None:
        return _non_completed_outcome(projects_root, project.id, failed, "data prep")

    job = FineTuneJob(
        dataset_list_path=dataset_list_path, experiment_dir=layout["training"], version=version,
        gpt_epochs=gpt_epochs,
    )
    on_progress("Fine-tuning the GPT stage...")
    stage1 = run_gpt_stage_fn(
        paths, job, on_progress=on_progress, cancel_check=cancel_check, popen_factory=popen_factory
    )
    if stage1.status is not RunStatus.COMPLETED:
        return _non_completed_outcome(projects_root, project.id, stage1, "training")

    checkpoint_path = find_latest_gpt_checkpoint(layout["training"] / "s1_weights")
    if checkpoint_path is None:
        raise RuntimeError("Training completed but no checkpoint file was found afterward.")

    # Persisted before the quality signal runs (not after) so quality_signal_fn's real
    # implementation - which loads the project's inference context the same way
    # Generate does - picks up this checkpoint rather than a stale/missing one.
    manager.update_config(
        projects_root, project.id, {"checkpoint_path": str(checkpoint_path), "gpt_sovits_version": version},
    )

    on_progress("Running post-training quality signal...")
    try:
        quality_band = quality_signal_fn(projects_root, project.id)
    except Exception as exc:
        # The checkpoint itself is already valid at this point - a failure in this
        # diagnostic-only step (e.g. a real CUDA OOM loading the model a second time)
        # must not undo a successful training run.
        on_progress(f"Quality signal failed (non-fatal): {exc}")
        quality_band = None
    quality_dict = (
        {"label": quality_band.label, "average_secs": quality_band.average_secs, "average_utmos": quality_band.average_utmos}
        if quality_band is not None
        else None
    )

    manager.update_config(projects_root, project.id, {"quality_band": quality_dict})
    manager.transition_project(projects_root, project.id, ProjectState.TRAINED)
    on_progress(f"Done. Quality signal: {quality_band.label if quality_band else 'unknown'}")
    return TrainingOutcome(
        project_id=project.id, ok=True, checkpoint_path=str(checkpoint_path), quality_band=quality_band
    )


def _non_completed_outcome(projects_root, project_id: str, result, step_name: str) -> TrainingOutcome:
    if result.status is RunStatus.CANCELLED:
        manager.transition_project(
            projects_root, project_id, ProjectState.ERROR, error_message=f"Cancelled during {step_name}"
        )
        return TrainingOutcome(project_id=project_id, ok=False, cancelled=True, error=f"Cancelled during {step_name}")
    raise RuntimeError(f"{step_name} failed: {result.error}")


def run_post_training_quality_signal(projects_root, project_id: str):
    """Real implementation - not called by tests directly (injected as
    quality_signal_fn instead, which is now called as quality_signal_fn(projects_root,
    project_id)).

    Reuses gui.pipeline.generation's cached inference context rather than loading its
    own separate TTS pipeline: confirmed via a real CUDA OOM on a real run that loading
    two independent copies (one here, one for the very next Generate click) doesn't fit
    on this hardware. Loading it here instead *pre-warms* that cache, so the first
    Generate click right after training reuses this same load rather than paying for a
    second one - this also makes the quality signal's own SECS reference embedding use
    the full reference clip (via the cached context), consistent with how Generate
    itself scores, rather than just the short 3-10s prompt clip as before.
    """
    from core.infer.quality_signal import run_quality_signal
    from gui.pipeline.generation import load_inference_context

    try:
        context = load_inference_context(projects_root, project_id)
    except ValueError:
        return None  # no chunk in the required 3-10s range - not fatal to training itself

    return run_quality_signal(
        tts_instance=context.tts_instance,
        text_lang="en",
        ref_audio_path=context.prompt.audio_path,
        prompt_text=context.prompt.text,
        prompt_lang="en",
        secs_fn=context.secs_fn,
        utmos_fn=context.utmos_fn,
    )
