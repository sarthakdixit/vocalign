#!/usr/bin/env python3
"""One-off real smoke test for Batch 4: load the real fine-tuned GPT checkpoint
produced by scripts/smoke_test_batch3.py alongside the base pretrained SoVITS
checkpoint (never fine-tuned - DESIGN.md S3 "Training scope"), and run one real
end-to-end generation through core/infer/generate.py's full pipeline (text
normalize -> chunk -> synthesize candidates -> Whisper-WER gate -> SECS+UTMOS
rank -> stitch -> export).

Needs scripts/smoke_test_batch3.py to have already produced a fine-tuned GPT
checkpoint under <experiment-dir>/s1_weights/*.ckpt.

Usage:
  python scripts/smoke_test_batch4.py path/to/clip.wav "reference text" "text to synthesize" [--version v2Pro]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.audio.io import load_and_transcode  # noqa: E402
from core.infer.generate import generate_and_export  # noqa: E402
from core.infer.ranking import default_secs_fn, default_utmos_fn  # noqa: E402
from core.infer.similarity import compute_embedding, reference_centroid  # noqa: E402
from core.infer.tts_adapter import TtsCheckpoints, load_tts  # noqa: E402
from core.train.data_prep import default_bert_dir, default_cnhubert_dir  # noqa: E402
from core.train.gpt_sovits_adapter import default_paths, ensure_on_sys_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_latest_checkpoint(weights_dir: Path) -> Path:
    checkpoints = sorted(weights_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            f"No .ckpt files found in {weights_dir} - run scripts/smoke_test_batch3.py first"
        )
    return checkpoints[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("reference_text")
    parser.add_argument("target_text")
    parser.add_argument("--version", default="v2Pro")
    parser.add_argument("--vendor-dir", type=Path, default=REPO_ROOT / "vendor" / "GPT-SoVITS")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "smoke_test_output")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.vendor_dir.exists():
        print(f"[smoke_test] {args.vendor_dir} doesn't exist - run scripts/vendor_gpt_sovits.sh first", file=sys.stderr)
        return 1

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = args.experiment_dir or (work_dir / "experiment")

    print("[smoke_test] 1/5 Locating the fine-tuned GPT checkpoint from Batch 3")
    checkpoint_path = _find_latest_checkpoint(experiment_dir / "s1_weights")
    print(f"[smoke_test]     using {checkpoint_path}")

    print("[smoke_test] 2/5 Loading TTS pipeline (fine-tuned GPT + base pretrained SoVITS)")
    paths = default_paths(args.vendor_dir, sys.executable, args.version)
    ensure_on_sys_path(paths)
    checkpoints = TtsCheckpoints(
        version=args.version,
        t2s_weights_path=checkpoint_path,
        vits_weights_path=paths.pretrained_s2_g,
        bert_base_path=default_bert_dir(args.vendor_dir),
        cnhubert_base_path=default_cnhubert_dir(args.vendor_dir),
        device=args.device,
    )
    tts_instance = load_tts(checkpoints)

    print("[smoke_test] 3/5 Building reference-speaker embedding for SECS scoring")
    ref_samples, ref_sr = load_and_transcode(args.audio_path, sample_rate=16000)
    ref_embedding = compute_embedding(ref_samples, ref_sr)
    secs_fn = default_secs_fn(reference_centroid([ref_embedding]))
    utmos_fn = default_utmos_fn()

    print(f"[smoke_test] 4/5 Generating: {args.target_text!r}")
    start = time.monotonic()
    output_path = work_dir / "generated_output.wav"
    audio_path, metadata_path = generate_and_export(
        args.target_text,
        output_path,
        tts_instance=tts_instance,
        text_lang="en",
        ref_audio_path=args.audio_path,
        prompt_text=args.reference_text,
        prompt_lang="en",
        secs_fn=secs_fn,
        utmos_fn=utmos_fn,
        on_progress=lambda message: print(f"[smoke_test]     {message}"),
    )
    elapsed = time.monotonic() - start

    print("[smoke_test] 5/5 Done")
    samples, sample_rate = sf.read(str(audio_path))
    duration = len(samples) / sample_rate
    peak_amplitude = float(max(abs(samples.min()), abs(samples.max())))
    metadata = json.loads(metadata_path.read_text())

    print(f"[smoke_test]     audio: {audio_path} ({duration:.2f}s, peak amplitude {peak_amplitude:.3f})")
    print(f"[smoke_test]     metadata: {metadata_path}")
    print(f"[smoke_test]     SECS={metadata['secs_score']} UTMOS={metadata['utmos_score']} low_confidence={metadata['low_confidence']}")
    print(f"[smoke_test]     wall time: {elapsed:.1f}s")
    print("[smoke_test]     Listen to the output file and judge voice similarity/naturalness yourself -")
    print("[smoke_test]     SECS/UTMOS are directional signals, not a substitute for your own ears.")

    if duration <= 0 or peak_amplitude == 0.0:
        print("[smoke_test] Output is empty or silent - treat as a failure.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
