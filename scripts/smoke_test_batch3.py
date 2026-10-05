#!/usr/bin/env python3
"""One-off real smoke test for Batch 3: preprocess a real clip end-to-end through our
own pipeline, drive GPT-SoVITS's data-prep scripts, then attempt a tiny fine-tune.

Needs vendor/GPT-SoVITS/ set up first (bash scripts/vendor_gpt_sovits.sh) and a short
real reference audio clip + its transcript text.

Usage:
  python scripts/smoke_test_batch3.py path/to/clip.wav "reference text" [--version v2Pro]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.align.whisper_align import compare_to_reference, transcribe_and_segment  # noqa: E402
from core.audio.chunk import build_training_chunks, slice_audio  # noqa: E402
from core.audio.io import load_and_transcode, write_wav  # noqa: E402
from core.audio.normalize import normalize_loudness  # noqa: E402
from core.audio.vad import detect_speech_segments, evaluate_quality, trim_silence  # noqa: E402
from core.train.data_prep import (  # noqa: E402
    DataPrepJob,
    default_bert_dir,
    default_cnhubert_dir,
    run_all_prep_steps,
)
from core.train.dataset import write_dataset_list  # noqa: E402
from core.train.gpt_sovits_adapter import FineTuneJob, default_paths, run_fine_tune  # noqa: E402
from core.train.recipe import select_recipe  # noqa: E402
from core.train.runner import RunStatus  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("reference_text")
    parser.add_argument("--version", default="v2Pro")
    parser.add_argument(
        "--segment-size",
        type=int,
        default=None,
        help="Override the SoVITS stage's train.segment_size (samples/step) - lower uses less VRAM "
        "per step at some cost to training quality/stability. Must be a multiple of the model's "
        "hop_length (640) or the model breaks internally with a shape-mismatch error. "
        "Template default is 20480; try 10240 for roughly half the memory.",
    )
    parser.add_argument("--vendor-dir", type=Path, default=REPO_ROOT / "vendor" / "GPT-SoVITS")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "smoke_test_output")
    args = parser.parse_args()

    if not args.vendor_dir.exists():
        print(f"[smoke_test] {args.vendor_dir} doesn't exist - run scripts/vendor_gpt_sovits.sh first", file=sys.stderr)
        return 1

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("[smoke_test] 1/7 Transcoding + normalizing audio")
    samples, sample_rate = load_and_transcode(args.audio_path, sample_rate=16000)
    samples = normalize_loudness(samples, sample_rate)

    print("[smoke_test] 2/7 Running VAD + quality gate")
    speech_segments = detect_speech_segments(samples, sample_rate)
    quality = evaluate_quality(speech_segments, len(samples) / sample_rate)
    for warning in quality.warnings:
        print(f"[smoke_test]     WARNING: {warning}")
    if quality.blocking:
        print("[smoke_test] Audio below the hard minimum - stopping.", file=sys.stderr)
        return 1
    trimmed = trim_silence(samples, sample_rate, speech_segments)

    trimmed_path = work_dir / "trimmed.wav"
    write_wav(trimmed_path, trimmed, sample_rate)

    print("[smoke_test] 3/7 Transcribing + segmenting with Whisper")
    whisper_segments = transcribe_and_segment(trimmed_path)
    match = compare_to_reference(whisper_segments, args.reference_text)
    print(f"[smoke_test]     transcript match ratio: {match.similarity_ratio:.2f}")
    if match.likely_mismatched:
        print("[smoke_test]     WARNING: transcription doesn't look like it matches your reference text")
        print(f"[smoke_test]     heard: {match.asr_text!r}")

    chunks = build_training_chunks(whisper_segments)
    total_seconds = sum(c.duration for c in chunks)
    print(f"[smoke_test]     {len(chunks)} chunk(s), {total_seconds:.1f}s total")

    print("[smoke_test] 4/7 Writing chunk audio + dataset list")
    audio_dir = work_dir / "chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    chunk_audio_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = audio_dir / f"chunk_{i:03d}.wav"
        write_wav(chunk_path, slice_audio(trimmed, sample_rate, chunk), sample_rate)
        chunk_audio_paths.append((chunk, chunk_path))

    dataset_list_path, skipped = write_dataset_list(work_dir / "dataset.list", chunk_audio_paths, "smoketest")
    print(f"[smoke_test]     dataset list: {dataset_list_path} ({skipped} low-confidence chunk(s) skipped)")

    print("[smoke_test] 5/7 Selecting recipe")
    recipe = select_recipe(total_seconds, version=args.version)
    print(
        f"[smoke_test]     tier={recipe.tier} attempt_training={recipe.attempt_training} "
        f"lora_rank={recipe.lora_rank} gpt_epochs={recipe.gpt_epochs} sovits_epochs={recipe.sovits_epochs}"
    )
    if not recipe.attempt_training:
        print("[smoke_test] Not enough audio to attempt training (zero-shot tier) - stopping here.")
        return 0

    paths = default_paths(args.vendor_dir, sys.executable, args.version)
    experiment_dir = work_dir / "experiment"
    job = FineTuneJob(
        dataset_list_path=dataset_list_path,
        experiment_dir=experiment_dir,
        version=args.version,
        gpt_epochs=recipe.gpt_epochs,
        sovits_epochs=recipe.sovits_epochs,
        lora_rank=recipe.lora_rank,
        segment_size=args.segment_size,
    )

    print("[smoke_test] 6/7 Running GPT-SoVITS's own data-prep pipeline (this needs the")
    print("[smoke_test]     pretrained checkpoints from scripts/vendor_gpt_sovits.sh)")
    prep_job = DataPrepJob(
        dataset_list_path=dataset_list_path,
        audio_dir=audio_dir,
        experiment_dir=experiment_dir,
        bert_pretrained_dir=default_bert_dir(args.vendor_dir),
        cnhubert_pretrained_dir=default_cnhubert_dir(args.vendor_dir),
        version=args.version,
    )
    prep_results = run_all_prep_steps(paths, prep_job, on_progress=print)
    if any(r.status is not RunStatus.COMPLETED for r in prep_results):
        failed = next(r for r in prep_results if r.status is not RunStatus.COMPLETED)
        print(f"[smoke_test] Data prep failed: {failed.error}", file=sys.stderr)
        return 1

    print("[smoke_test] 7/7 Running fine-tune (this is the real test)")
    stage1, stage2 = run_fine_tune(paths, job, on_progress=print)

    print(f"[smoke_test] GPT stage: {stage1.status.value}" + (f" ({stage1.error})" if stage1.error else ""))
    if stage2 is not None:
        print(f"[smoke_test] SoVITS stage: {stage2.status.value}" + (f" ({stage2.error})" if stage2.error else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
