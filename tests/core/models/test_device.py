import sys

import pytest

from core.models.device import DeviceInfo, detect_device


class _FakeCuda:
    def __init__(
        self,
        available,
        name="NVIDIA Fake GPU",
        free_bytes=2 * 1024**3,
        total_bytes=8 * 1024**3,
    ):
        self._available = available
        self._name = name
        self._free_bytes = free_bytes
        self._total_bytes = total_bytes

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        assert index == 0
        return self._name

    def mem_get_info(self, index):
        assert index == 0
        return self._free_bytes, self._total_bytes


class _FakeVersion:
    def __init__(self, cuda_version):
        self.cuda = cuda_version


class _FakeTorch:
    def __init__(self, cuda_available, cuda_version="12.4", **cuda_kwargs):
        self.__version__ = "2.4.0+fake"
        self.cuda = _FakeCuda(cuda_available, **cuda_kwargs)
        self.version = _FakeVersion(cuda_version)


def test_detect_device_reports_cpu_when_no_cuda():
    fake_torch = _FakeTorch(cuda_available=False)

    info = detect_device(torch_module=fake_torch)

    assert info == DeviceInfo(kind="cpu", name="CPU", torch_version="2.4.0+fake")


def test_detect_device_reports_cuda_details_when_available():
    fake_torch = _FakeTorch(
        cuda_available=True,
        cuda_version="12.4",
        name="NVIDIA RTX 4090",
        free_bytes=4 * 1024**3,
        total_bytes=24 * 1024**3,
    )

    info = detect_device(torch_module=fake_torch)

    assert info.kind == "cuda"
    assert info.name == "NVIDIA RTX 4090"
    assert info.cuda_version == "12.4"
    assert info.vram_total_mb == 24 * 1024
    assert info.vram_free_mb == 4 * 1024
    assert info.torch_version == "2.4.0+fake"


def test_detect_device_raises_clear_error_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        detect_device()


def test_device_info_is_frozen():
    info = DeviceInfo(kind="cpu", name="CPU", torch_version="2.4.0")
    with pytest.raises(AttributeError):
        info.kind = "cuda"  # type: ignore[misc]
