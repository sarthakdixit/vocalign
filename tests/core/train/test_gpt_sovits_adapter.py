import copy
import json
import os

import pytest
import yaml

from core.train import gpt_sovits_adapter as adapter
from core.train.runner import RunStatus

# Minimal but structurally real fakes - same top-level shape as the actual
# s1longer-v2.yaml / s2.json templates (confirmed verbatim against the real repo),
# just trimmed. Used so build_s1_config/build_s2_config tests exercise the real
# "load template, overlay specific keys" code path without needing a real checkout.
FAKE_S1_TEMPLATE = {
    "train": {
        "seed": 1234,
        "epochs": 20,
        "batch_size": 8,
        "save_every_n_epoch": 1,
        "precision": "16-mixed",
    },
    "data": {"max_sec": 54, "num_workers": 4},
    "model": {"vocab_size": 1025, "phoneme_vocab_size": 732},
}

FAKE_S2_TEMPLATE = {
    "train": {"batch_size": 32, "epochs": 100, "fp16_run": True, "segment_size": 20480},
    "data": {"sampling_rate": 32000, "filter_length": 2048, "hop_length": 640, "win_length": 2048},
    "model": {"inter_channels": 192, "gin_channels": 1024},
    "s2_ckpt_dir": "logs/s2/big2k1",
    "content_module": "cnhubert",
}


def _paths(tmp_path):
    return adapter.GptSoVitsPaths(
        repo_root=tmp_path / "vendor" / "GPT-SoVITS",
        python_executable="python",
        pretrained_s1=tmp_path / "pretrained" / "s1.ckpt",
        pretrained_s2_g=tmp_path / "pretrained" / "s2G.pth",
    )


def _job(tmp_path, **overrides):
    defaults = dict(
        dataset_list_path=tmp_path / "dataset.list",
        experiment_dir=tmp_path / "experiment",
        version="v2Pro",
        gpt_epochs=2,
        sovits_epochs=2,
        lora_rank=None,
    )
    defaults.update(overrides)
    return adapter.FineTuneJob(**defaults)


def _write_s1_template_file(paths, version):
    template_path = paths.repo_root / "GPT_SoVITS" / "configs" / adapter.S1_TEMPLATE_BY_VERSION[version]
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(yaml.safe_dump(FAKE_S1_TEMPLATE))


def _write_s2_template_file(paths, version):
    template_path = adapter.s2_template_path(paths.repo_root, version)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps(FAKE_S2_TEMPLATE))


# --- build_s1_config / build_s2_config: overlay logic, given an injected template ---


def test_build_s1_config_overlays_job_fields(tmp_path):
    config = adapter.build_s1_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S1_TEMPLATE)

    assert config["train"]["epochs"] == 2
    assert config["pretrained_s1"] == str(tmp_path / "pretrained" / "s1.ckpt")


def test_build_s1_config_preserves_template_fields_it_does_not_override(tmp_path):
    # Regression check for the real bug: a hand-built config omitted real template
    # fields entirely, causing an AttributeError deep in GPT-SoVITS at runtime.
    config = adapter.build_s1_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S1_TEMPLATE)

    assert config["data"]["max_sec"] == 54
    assert config["model"]["phoneme_vocab_size"] == 732


def test_build_s1_config_does_not_mutate_the_passed_in_template(tmp_path):
    before = copy.deepcopy(FAKE_S1_TEMPLATE)

    adapter.build_s1_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S1_TEMPLATE)

    assert FAKE_S1_TEMPLATE == before


def test_build_s2_config_omits_lora_rank_when_not_set(tmp_path):
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path, lora_rank=None), template=FAKE_S2_TEMPLATE)

    assert "lora_rank" not in config["train"]


def test_build_s2_config_includes_lora_rank_when_set(tmp_path):
    config = adapter.build_s2_config(
        _paths(tmp_path), _job(tmp_path, version="v3", lora_rank=16), template=FAKE_S2_TEMPLATE
    )

    assert config["train"]["lora_rank"] == 16
    assert config["model"]["version"] == "v3"


def test_build_s2_config_includes_pretrained_s2d_only_when_given(tmp_path):
    paths = _paths(tmp_path)
    without_d = adapter.build_s2_config(paths, _job(tmp_path), template=FAKE_S2_TEMPLATE)
    assert "pretrained_s2D" not in without_d["train"]

    paths_with_d = adapter.GptSoVitsPaths(
        repo_root=paths.repo_root,
        python_executable=paths.python_executable,
        pretrained_s1=paths.pretrained_s1,
        pretrained_s2_g=paths.pretrained_s2_g,
        pretrained_s2_d=tmp_path / "pretrained" / "s2D.pth",
    )
    with_d = adapter.build_s2_config(paths_with_d, _job(tmp_path), template=FAKE_S2_TEMPLATE)
    assert with_d["train"]["pretrained_s2D"] == str(tmp_path / "pretrained" / "s2D.pth")


def test_build_s2_config_sets_gpu_numbers(tmp_path):
    # s2_train.py reads hps.train.gpu_numbers at module import time with no fallback -
    # confirmed with a real AttributeError on a real run when this was missing.
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S2_TEMPLATE)

    assert config["train"]["gpu_numbers"] == "0"


def test_build_s2_config_grad_ckpt_defaults_to_true(tmp_path):
    # On by default: confirmed with a real CUDA OOM mid-epoch that this GPU needs it.
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S2_TEMPLATE)

    assert config["train"]["grad_ckpt"] is True


def test_build_s2_config_grad_ckpt_can_be_disabled(tmp_path):
    config = adapter.build_s2_config(
        _paths(tmp_path), _job(tmp_path, grad_ckpt=False), template=FAKE_S2_TEMPLATE
    )

    assert config["train"]["grad_ckpt"] is False


def test_build_s2_config_segment_size_defaults_to_template_value(tmp_path):
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S2_TEMPLATE)

    assert config["train"]["segment_size"] == FAKE_S2_TEMPLATE["train"]["segment_size"]


def test_build_s2_config_segment_size_can_be_overridden(tmp_path):
    config = adapter.build_s2_config(
        _paths(tmp_path), _job(tmp_path, segment_size=10240), template=FAKE_S2_TEMPLATE
    )

    assert config["train"]["segment_size"] == 10240


def test_build_s2_config_rejects_segment_size_not_a_multiple_of_hop_length(tmp_path):
    # Confirmed the hard way with a real shape-mismatch RuntimeError on a real run:
    # 8192 isn't a multiple of hop_length (640), and that broke the model internally.
    with pytest.raises(ValueError, match="multiple of"):
        adapter.build_s2_config(_paths(tmp_path), _job(tmp_path, segment_size=8192), template=FAKE_S2_TEMPLATE)


def test_build_s2_config_segment_size_error_suggests_a_valid_value(tmp_path):
    # round(8192 / 640) * 640 = 13 * 640 = 8320 - the nearest multiple of hop_length.
    with pytest.raises(ValueError, match="8320"):
        adapter.build_s2_config(_paths(tmp_path), _job(tmp_path, segment_size=8192), template=FAKE_S2_TEMPLATE)


def test_build_s2_config_preserves_template_fields_it_does_not_override(tmp_path):
    # filter_length specifically - this is the exact field the real run crashed on.
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S2_TEMPLATE)

    assert config["data"]["filter_length"] == 2048
    assert config["data"]["sampling_rate"] == 32000


def test_build_s2_config_does_not_mutate_the_passed_in_template(tmp_path):
    before = copy.deepcopy(FAKE_S2_TEMPLATE)

    adapter.build_s2_config(_paths(tmp_path), _job(tmp_path), template=FAKE_S2_TEMPLATE)

    assert FAKE_S2_TEMPLATE == before


# --- build_env (PYTHONPATH for any subprocess call into the vendored checkout) ---


def test_build_env_includes_pythonpath_for_repo_root_and_gpt_sovits_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    paths = _paths(tmp_path)

    env = adapter.build_env(paths)

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert str(paths.repo_root) in entries
    assert str(paths.repo_root / "GPT_SoVITS") in entries


def test_build_env_preserves_existing_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/other/path")
    paths = _paths(tmp_path)

    env = adapter.build_env(paths)

    assert "/some/other/path" in env["PYTHONPATH"].split(os.pathsep)


def test_build_env_merges_extra_vars_without_overwriting_pythonpath(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    paths = _paths(tmp_path)

    env = adapter.build_env(paths, extra={"FOO": "bar"})

    assert env["FOO"] == "bar"
    assert "PYTHONPATH" in env


# --- template path/loading helpers ---


def test_s2_template_path_uses_version_specific_file_for_pro_versions(tmp_path):
    paths = _paths(tmp_path)
    assert adapter.s2_template_path(paths.repo_root, "v2Pro").name == "s2v2Pro.json"
    assert adapter.s2_template_path(paths.repo_root, "v2ProPlus").name == "s2v2ProPlus.json"


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_s2_template_path_uses_plain_s2_json_for_other_versions(tmp_path, version):
    paths = _paths(tmp_path)
    assert adapter.s2_template_path(paths.repo_root, version).name == "s2.json"


def test_s2_template_path_rejects_unknown_version(tmp_path):
    with pytest.raises(ValueError):
        adapter.s2_template_path(_paths(tmp_path).repo_root, "v5")


def test_load_s1_template_reads_a_real_file(tmp_path):
    paths = _paths(tmp_path)
    _write_s1_template_file(paths, "v2Pro")

    loaded = adapter.load_s1_template(paths.repo_root, "v2Pro")

    assert loaded == FAKE_S1_TEMPLATE


def test_load_s2_template_reads_a_real_file(tmp_path):
    paths = _paths(tmp_path)
    _write_s2_template_file(paths, "v3")

    loaded = adapter.load_s2_template(paths.repo_root, "v3")

    assert loaded == FAKE_S2_TEMPLATE


# --- run_gpt_stage / run_sovits_stage / run_fine_tune ---


def test_run_gpt_stage_writes_yaml_config_and_invokes_s1_script(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=["epoch 1/2"], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)

    result = adapter.run_gpt_stage(paths, job, popen_factory=factory)

    assert result.status == RunStatus.COMPLETED
    call = factory.calls[0]
    assert call["cwd"] == str(paths.repo_root)
    assert call["command"][0] == "python"
    assert "GPT_SoVITS/s1_train.py" in call["command"]
    assert "--config_file" in call["command"]
    assert str(paths.repo_root / "GPT_SoVITS") in call["env"]["PYTHONPATH"].split(os.pathsep)

    config_path = job.experiment_dir / "s1_config.yaml"
    assert config_path.exists()
    written = yaml.safe_load(config_path.read_text())
    assert written["train"]["epochs"] == 2
    assert written["data"]["max_sec"] == 54  # preserved from the template


def test_run_gpt_stage_creates_half_weights_save_dir_and_output_dir(tmp_path, fake_popen_factory):
    # Regression test: s1_train.py shutil.move()s checkpoints into this directory
    # without creating it - confirmed with a real FileNotFoundError on a real run.
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)

    adapter.run_gpt_stage(paths, job, popen_factory=factory)

    assert (job.experiment_dir / "s1_weights").is_dir()
    assert (job.experiment_dir / "s1").is_dir()


def test_run_gpt_stage_clears_stale_checkpoint_dir_before_starting(tmp_path, fake_popen_factory):
    # pytorch_lightning auto-resumes from any checkpoint already in output_dir, which
    # is incompatible with PyTorch 2.6+'s weights_only=True default on torch.load -
    # confirmed with a real UnpicklingError on a real run. Each run should start fresh.
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)
    stale_ckpt_dir = job.experiment_dir / "s1" / "ckpt"
    stale_ckpt_dir.mkdir(parents=True)
    (stale_ckpt_dir / "epoch=1-step=46.ckpt").write_bytes(b"stale")

    adapter.run_gpt_stage(paths, job, popen_factory=factory)

    assert not (stale_ckpt_dir / "epoch=1-step=46.ckpt").exists()
    assert (job.experiment_dir / "s1").is_dir()


def test_run_sovits_stage_creates_save_weight_dir(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s2_template_file(paths, job.version)

    adapter.run_sovits_stage(paths, job, popen_factory=factory)

    assert (job.experiment_dir / "s2_weights").is_dir()


def test_run_gpt_stage_retries_clearing_checkpoint_dir_on_transient_oserror(
    tmp_path, fake_popen_factory, monkeypatch
):
    # Confirmed the hard way on a real run: a mounted/external filesystem reported
    # "Directory not empty" mid-rmtree (sync lag, not a real conflict). Retrying
    # should recover rather than letting a transient hiccup fail the whole run.
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)
    (job.experiment_dir / "s1").mkdir(parents=True)

    real_rmtree = adapter.shutil.rmtree
    calls = {"n": 0}

    def flaky_rmtree(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(39, "Directory not empty")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(adapter.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)

    adapter.run_gpt_stage(paths, job, popen_factory=factory)

    assert calls["n"] == 2
    assert (job.experiment_dir / "s1").is_dir()


def test_run_gpt_stage_reraises_if_clearing_checkpoint_dir_keeps_failing(
    tmp_path, fake_popen_factory, monkeypatch
):
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)
    (job.experiment_dir / "s1").mkdir(parents=True)

    def always_fails(path, *args, **kwargs):
        raise OSError(39, "Directory not empty")

    monkeypatch.setattr(adapter.shutil, "rmtree", always_fails)
    monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)

    with pytest.raises(OSError):
        adapter.run_gpt_stage(paths, job, popen_factory=factory)


def test_run_sovits_stage_sets_expandable_segments_alloc_conf(tmp_path, fake_popen_factory):
    # Suggested directly by a real CUDA OOM's own error message (fragmentation, not
    # just raw usage) - cheap to try given how tight this stage's VRAM budget is.
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s2_template_file(paths, job.version)

    adapter.run_sovits_stage(paths, job, popen_factory=factory)

    assert factory.calls[0]["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_run_sovits_stage_uses_lora_script_for_v3_and_v4(tmp_path, fake_popen_factory):
    for version in ("v3", "v4"):
        factory = fake_popen_factory(lines=[], returncode=0)
        paths = _paths(tmp_path)
        job = _job(tmp_path, version=version, lora_rank=16)
        _write_s2_template_file(paths, version)

        adapter.run_sovits_stage(paths, job, popen_factory=factory)

        assert "GPT_SoVITS/s2_train_v3_lora.py" in factory.calls[0]["command"]


@pytest.mark.parametrize("version", ["v2", "v2Pro", "v2ProPlus"])
def test_run_sovits_stage_uses_full_script_for_v2_family(tmp_path, fake_popen_factory, version):
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path, version=version, lora_rank=None)
    _write_s2_template_file(paths, version)

    adapter.run_sovits_stage(paths, job, popen_factory=factory)

    assert "GPT_SoVITS/s2_train.py" in factory.calls[0]["command"]


def test_run_fine_tune_skips_stage2_when_stage1_fails(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=["error"], returncode=1)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)

    stage1, stage2 = adapter.run_fine_tune(paths, job, popen_factory=factory)

    assert stage1.status == RunStatus.FAILED
    assert stage2 is None
    assert len(factory.calls) == 1


def test_run_fine_tune_runs_both_stages_on_success(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)
    _write_s1_template_file(paths, job.version)
    _write_s2_template_file(paths, job.version)

    stage1, stage2 = adapter.run_fine_tune(paths, job, popen_factory=factory)

    assert stage1.status == RunStatus.COMPLETED
    assert stage2.status == RunStatus.COMPLETED
    assert len(factory.calls) == 2


# --- default_paths (pretrained checkpoint locations) ---


def test_default_paths_v2pro_matches_confirmed_filenames(tmp_path):
    paths = adapter.default_paths(tmp_path, "python", "v2Pro")

    assert paths.pretrained_s2_g == tmp_path / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2Pro.pth"
    assert paths.pretrained_s1 == tmp_path / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    assert paths.pretrained_s2_d.name == "s2Dv2Pro.pth"


def test_default_paths_all_known_versions_resolve_with_s2d(tmp_path):
    for version in ("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"):
        paths = adapter.default_paths(tmp_path, "python", version)
        assert paths.pretrained_s2_d is not None
        assert "s2D" in paths.pretrained_s2_d.name


def test_default_paths_rejects_unknown_version(tmp_path):
    with pytest.raises(ValueError):
        adapter.default_paths(tmp_path, "python", "v5")


def test_default_paths_v4_uses_its_own_pretrained_subdir(tmp_path):
    paths = adapter.default_paths(tmp_path, "python", "v4")

    assert "gsv-v4-pretrained" in str(paths.pretrained_s2_g)


def test_default_paths_s1_shared_across_v2pro_family_and_v3_v4(tmp_path):
    s1_paths = {
        version: adapter.default_paths(tmp_path, "python", version).pretrained_s1
        for version in ("v2Pro", "v2ProPlus", "v3", "v4")
    }

    assert len(set(s1_paths.values())) == 1
