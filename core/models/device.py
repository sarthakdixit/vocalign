"""GPU/CPU device detection. CUDA + CPU only - see DESIGN.md's GPU scope decision."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # "cuda" or "cpu"
    name: str
    torch_version: str
    cuda_version: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None


def detect_device(torch_module=None) -> DeviceInfo:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed in this environment. Run "
                "scripts/setup_env.sh (Linux/macOS) or scripts/setup_env.ps1 (Windows) first."
            ) from exc

    torch_version = str(torch_module.__version__)

    if torch_module.cuda.is_available():
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
        return DeviceInfo(
            kind="cuda",
            name=torch_module.cuda.get_device_name(0),
            torch_version=torch_version,
            cuda_version=str(torch_module.version.cuda),
            vram_total_mb=total_bytes // (1024 * 1024),
            vram_free_mb=free_bytes // (1024 * 1024),
        )

    return DeviceInfo(kind="cpu", name="CPU", torch_version=torch_version)
