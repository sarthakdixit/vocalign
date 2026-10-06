#!/usr/bin/env python3
"""Checks whether this machine's environment is ready to run clone-voice: Python
version, ffmpeg, GPU/device, required packages importable, and the vendored
GPT-SoVITS checkout's key files present and not obviously truncated. Downloads
nothing - run scripts/setup_env.* and scripts/vendor_gpt_sovits.* first if anything
here is reported missing.

Usage:
  python scripts/verify_setup.py [--version v2Pro] [--vendor-dir path/to/GPT-SoVITS]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config  # noqa: E402
from core.setup_check import run_all_checks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=config.GPT_SOVITS_VERSION)
    parser.add_argument("--vendor-dir", type=Path, default=config.VENDOR_DIR)
    args = parser.parse_args()

    report = run_all_checks(args.vendor_dir, args.version)

    for result in report.results:
        status = "OK  " if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    if report.all_ok:
        print("\nEverything checked out - ready to run scripts/run.py.")
        return 0

    print(f"\n{len(report.failures)} check(s) failed - see FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
