from types import SimpleNamespace

from core import setup_check
from core.train.gpt_sovits_adapter import default_paths


def _version(major, minor, micro=0):
    return SimpleNamespace(major=major, minor=minor, micro=micro)


def test_check_python_version_accepts_310():
    result = setup_check.check_python_version(version_info=_version(3, 10, 5))

    assert result.ok is True
    assert result.detail == "3.10.5"


def test_check_python_version_accepts_311():
    result = setup_check.check_python_version(version_info=_version(3, 11, 0))

    assert result.ok is True


def test_check_python_version_rejects_unsupported_version():
    result = setup_check.check_python_version(version_info=_version(3, 9, 0))

    assert result.ok is False
    assert "expected 3.10 or 3.11" in result.detail


def test_check_ffmpeg_ok_when_found(tmp_path):
    fake_path = str(tmp_path / "ffmpeg")

    result = setup_check.check_ffmpeg(which_fn=lambda name: fake_path)

    assert result.ok is True
    assert result.detail == fake_path


def test_check_ffmpeg_fails_when_not_found():
    result = setup_check.check_ffmpeg(which_fn=lambda name: None)

    assert result.ok is False
    assert "not found" in result.detail


def test_check_packages_reports_each_importable_package():
    results = setup_check.check_packages(packages=("foo", "bar"), import_fn=lambda name: object())

    assert [r.ok for r in results] == [True, True]
    assert [r.name for r in results] == ["package: foo", "package: bar"]


def test_check_packages_reports_missing_package_as_failed():
    def import_fn(name):
        if name == "missing_pkg":
            raise ImportError("No module named 'missing_pkg'")
        return object()

    results = setup_check.check_packages(packages=("ok_pkg", "missing_pkg"), import_fn=import_fn)

    assert results[0].ok is True
    assert results[1].ok is False
    assert "missing_pkg" in results[1].detail


def test_check_device_reports_cuda_details():
    fake_info = SimpleNamespace(kind="cuda", name="RTX 3050", vram_total_mb=3768, vram_free_mb=3000, cuda_version="12.6")

    result = setup_check.check_device(detect_fn=lambda: fake_info)

    assert result.ok is True
    assert "RTX 3050" in result.detail
    assert "3768" in result.detail


def test_check_device_reports_cpu_without_failing():
    fake_info = SimpleNamespace(kind="cpu", name="CPU", vram_total_mb=None, vram_free_mb=None, cuda_version=None)

    result = setup_check.check_device(detect_fn=lambda: fake_info)

    assert result.ok is True
    assert "CPU" in result.detail


def test_check_device_fails_when_torch_is_not_installed():
    def raising_detect():
        raise RuntimeError("PyTorch is not installed in this environment.")

    result = setup_check.check_device(detect_fn=raising_detect)

    assert result.ok is False
    assert "PyTorch is not installed" in result.detail


def test_check_vendored_gpt_sovits_fails_when_vendor_dir_missing(tmp_path):
    results = setup_check.check_vendored_gpt_sovits(tmp_path / "does_not_exist", "v2Pro")

    assert len(results) == 1
    assert results[0].ok is False
    assert "vendor_gpt_sovits" in results[0].detail


def test_check_vendored_gpt_sovits_fails_for_unknown_version(tmp_path):
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()

    results = setup_check.check_vendored_gpt_sovits(vendor_dir, "v5")

    assert len(results) == 1
    assert results[0].ok is False


def _write_checkpoint(path, size_bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size_bytes)


def test_check_vendored_gpt_sovits_all_ok_with_real_sized_files(tmp_path):
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    paths = default_paths(vendor_dir, "python", "v2Pro")
    _write_checkpoint(paths.pretrained_s1, setup_check.MIN_CHECKPOINT_SIZE_BYTES + 1)
    _write_checkpoint(paths.pretrained_s2_g, setup_check.MIN_CHECKPOINT_SIZE_BYTES + 1)
    (vendor_dir / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large").mkdir(parents=True)
    (vendor_dir / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base").mkdir(parents=True)

    results = setup_check.check_vendored_gpt_sovits(vendor_dir, "v2Pro")

    assert all(r.ok for r in results)


def test_check_vendored_gpt_sovits_flags_a_truncated_checkpoint(tmp_path):
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    paths = default_paths(vendor_dir, "python", "v2Pro")
    _write_checkpoint(paths.pretrained_s1, 100)  # far below MIN_CHECKPOINT_SIZE_BYTES
    _write_checkpoint(paths.pretrained_s2_g, setup_check.MIN_CHECKPOINT_SIZE_BYTES + 1)
    (vendor_dir / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large").mkdir(parents=True)
    (vendor_dir / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base").mkdir(parents=True)

    results = setup_check.check_vendored_gpt_sovits(vendor_dir, "v2Pro")

    gpt_result = next(r for r in results if "GPT checkpoint" in r.name)
    assert gpt_result.ok is False
    assert "truncated" in gpt_result.detail


def test_setup_report_all_ok_true_when_everything_passes():
    report = setup_check.SetupReport(results=[setup_check.CheckResult("a", True, ""), setup_check.CheckResult("b", True, "")])

    assert report.all_ok is True
    assert report.failures == []


def test_setup_report_all_ok_false_when_anything_fails():
    failing = setup_check.CheckResult("b", False, "nope")
    report = setup_check.SetupReport(results=[setup_check.CheckResult("a", True, ""), failing])

    assert report.all_ok is False
    assert report.failures == [failing]
