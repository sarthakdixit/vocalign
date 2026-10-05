import pytest
import yaml

from core.train import gpt_sovits_adapter as adapter
from core.train.runner import RunStatus


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


def test_build_s1_config_includes_expected_fields(tmp_path):
    config = adapter.build_s1_config(_paths(tmp_path), _job(tmp_path))

    assert config["train"]["epochs"] == 2
    assert config["pretrained_s1"] == str(tmp_path / "pretrained" / "s1.ckpt")


def test_build_s2_config_omits_lora_rank_when_not_set(tmp_path):
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path, lora_rank=None))

    assert "lora_rank" not in config["train"]


def test_build_s2_config_includes_lora_rank_when_set(tmp_path):
    config = adapter.build_s2_config(_paths(tmp_path), _job(tmp_path, version="v3", lora_rank=16))

    assert config["train"]["lora_rank"] == 16
    assert config["model"]["version"] == "v3"


def test_build_s2_config_includes_pretrained_s2d_only_when_given(tmp_path):
    paths = _paths(tmp_path)
    without_d = adapter.build_s2_config(paths, _job(tmp_path))
    assert "pretrained_s2D" not in without_d["train"]

    paths_with_d = adapter.GptSoVitsPaths(
        repo_root=paths.repo_root,
        python_executable=paths.python_executable,
        pretrained_s1=paths.pretrained_s1,
        pretrained_s2_g=paths.pretrained_s2_g,
        pretrained_s2_d=tmp_path / "pretrained" / "s2D.pth",
    )
    with_d = adapter.build_s2_config(paths_with_d, _job(tmp_path))
    assert with_d["train"]["pretrained_s2D"] == str(tmp_path / "pretrained" / "s2D.pth")


def test_run_gpt_stage_writes_yaml_config_and_invokes_s1_script(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=["epoch 1/2"], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)

    result = adapter.run_gpt_stage(paths, job, popen_factory=factory)

    assert result.status == RunStatus.COMPLETED
    call = factory.calls[0]
    assert call["cwd"] == str(paths.repo_root)
    assert call["command"][0] == "python"
    assert "GPT_SoVITS/s1_train.py" in call["command"]
    assert "--config_file" in call["command"]

    config_path = job.experiment_dir / "s1_config.yaml"
    assert config_path.exists()
    assert yaml.safe_load(config_path.read_text())["train"]["epochs"] == 2


def test_run_sovits_stage_uses_lora_script_for_v3_and_v4(tmp_path, fake_popen_factory):
    for version in ("v3", "v4"):
        factory = fake_popen_factory(lines=[], returncode=0)
        job = _job(tmp_path, version=version, lora_rank=16)

        adapter.run_sovits_stage(_paths(tmp_path), job, popen_factory=factory)

        assert "GPT_SoVITS/s2_train_v3_lora.py" in factory.calls[0]["command"]


@pytest.mark.parametrize("version", ["v2", "v2Pro", "v2ProPlus"])
def test_run_sovits_stage_uses_full_script_for_v2_family(tmp_path, fake_popen_factory, version):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path, version=version, lora_rank=None)

    adapter.run_sovits_stage(_paths(tmp_path), job, popen_factory=factory)

    assert "GPT_SoVITS/s2_train.py" in factory.calls[0]["command"]


def test_run_fine_tune_skips_stage2_when_stage1_fails(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=["error"], returncode=1)

    stage1, stage2 = adapter.run_fine_tune(_paths(tmp_path), _job(tmp_path), popen_factory=factory)

    assert stage1.status == RunStatus.FAILED
    assert stage2 is None
    assert len(factory.calls) == 1


def test_run_fine_tune_runs_both_stages_on_success(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)

    stage1, stage2 = adapter.run_fine_tune(_paths(tmp_path), _job(tmp_path), popen_factory=factory)

    assert stage1.status == RunStatus.COMPLETED
    assert stage2.status == RunStatus.COMPLETED
    assert len(factory.calls) == 2


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
