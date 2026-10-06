"""Environment/setup diagnostics (DESIGN.md Batch 6, revised scope - see §10/§13): a
single place to check "is my environment ready" instead of discovering gaps one real
error at a time, the way missing ffmpeg, partial downloads from real network flakiness,
and mismatched dependency versions were each discovered separately across Batches 3-5.

Downloads nothing itself: every model this project needs is fetched by someone else's
own mechanism now (GPT-SoVITS's own installer, faster-whisper/resemblyzer/utmos-pytorch's
own auto-download) - this only checks what should already be there.
"""

import importlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Real checkpoints are hundreds of MB+; anything smaller strongly suggests a
# truncated/partial download (confirmed real risk - see the repeated real network
# flakiness logged in DESIGN.md S13), not a genuine file.
MIN_CHECKPOINT_SIZE_BYTES = 1_000_000

SUPPORTED_PYTHON_VERSIONS = {(3, 10), (3, 11)}

# Import names, not pip package names, where they differ (pyyaml -> yaml,
# pywebview -> webview) - confirmed against how this project's own code imports them.
REQUIRED_PACKAGES = (
    "torch", "gradio", "webview", "faster_whisper", "resemblyzer", "utmos_pytorch",
    "num2words", "soundfile", "pyloudnorm", "silero_vad", "yaml",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class SetupReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok]


def check_python_version(version_info=None) -> CheckResult:
    version_info = version_info or sys.version_info
    ok = (version_info.major, version_info.minor) in SUPPORTED_PYTHON_VERSIONS
    detail = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    if not ok:
        detail += " - expected 3.10 or 3.11 (see DESIGN.md Tech Stack)"
    return CheckResult("Python version", ok, detail)


def check_ffmpeg(which_fn=None) -> CheckResult:
    which_fn = which_fn or shutil.which
    path = which_fn("ffmpeg")
    detail = path if path else "not found on PATH - required by core/audio/io.py for transcoding"
    return CheckResult("ffmpeg", path is not None, detail)


def check_packages(packages=REQUIRED_PACKAGES, import_fn=None) -> list[CheckResult]:
    import_fn = import_fn or importlib.import_module
    results = []
    for package in packages:
        try:
            import_fn(package)
            results.append(CheckResult(f"package: {package}", True, "importable"))
        except ImportError as exc:
            results.append(CheckResult(f"package: {package}", False, str(exc)))
    return results


def check_device(detect_fn=None) -> CheckResult:
    from core.models.device import detect_device

    detect_fn = detect_fn or detect_device
    try:
        info = detect_fn()
    except RuntimeError as exc:
        return CheckResult("GPU/device", False, str(exc))
    if info.kind == "cuda":
        detail = f"{info.name}, {info.vram_total_mb}MB VRAM, CUDA {info.cuda_version}"
    else:
        detail = "CPU only - training will be slow and limited to the smallest recipe tier"
    return CheckResult("GPU/device", True, detail)


def check_vendored_gpt_sovits(vendor_dir: Path, version: str) -> list[CheckResult]:
    from core.train.data_prep import default_bert_dir, default_cnhubert_dir
    from core.train.gpt_sovits_adapter import default_paths

    vendor_dir = Path(vendor_dir)
    if not vendor_dir.exists():
        return [
            CheckResult(
                "vendored GPT-SoVITS", False,
                f"{vendor_dir} does not exist - run scripts/vendor_gpt_sovits first",
            )
        ]

    try:
        paths = default_paths(vendor_dir, "python", version)
    except ValueError as exc:
        return [CheckResult("vendored GPT-SoVITS", False, str(exc))]

    targets = {
        f"{version} GPT checkpoint": paths.pretrained_s1,
        f"{version} SoVITS checkpoint": paths.pretrained_s2_g,
        "BERT pretrained dir": default_bert_dir(vendor_dir),
        "CNHuBERT pretrained dir": default_cnhubert_dir(vendor_dir),
    }
    return [_check_vendored_path(label, path) for label, path in targets.items()]


def _check_vendored_path(label: str, path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(label, False, f"{path} not found")
    if path.is_dir():
        return CheckResult(label, True, f"{path} (directory present)")
    size = path.stat().st_size
    if size < MIN_CHECKPOINT_SIZE_BYTES:
        return CheckResult(label, False, f"{path} is only {size} bytes - looks truncated or partial")
    return CheckResult(label, True, f"{path} ({size / 1_000_000:.1f}MB)")


def run_all_checks(vendor_dir: Path, version: str) -> SetupReport:
    results = [check_python_version(), check_ffmpeg(), check_device()]
    results.extend(check_packages())
    results.extend(check_vendored_gpt_sovits(vendor_dir, version))
    return SetupReport(results=results)
