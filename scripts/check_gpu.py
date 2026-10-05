#!/usr/bin/env python3
"""Reports the compute device clone-voice will use. Run after scripts/setup_env.*"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models.device import detect_device  # noqa: E402


def main() -> int:
    try:
        info = detect_device()
    except RuntimeError as exc:
        print(f"[check_gpu] {exc}", file=sys.stderr)
        return 1

    print(f"torch version: {info.torch_version}")
    print(f"device:        {info.kind}")
    print(f"device name:   {info.name}")
    if info.kind == "cuda":
        print(f"CUDA version:  {info.cuda_version}")
        print(f"VRAM total:    {info.vram_total_mb} MB")
        print(f"VRAM free:     {info.vram_free_mb} MB")
    else:
        print("No CUDA GPU detected - training will fall back to CPU")
        print("(fine for the smallest adaptive tier only; see DESIGN.md section 8.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
