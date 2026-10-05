import os

import pytest

from core.train import data_prep
from core.train.gpt_sovits_adapter import GptSoVitsPaths
from core.train.runner import RunStatus


def _paths(tmp_path):
    return GptSoVitsPaths(
        repo_root=tmp_path / "vendor" / "GPT-SoVITS",
        python_executable="python",
        pretrained_s1=tmp_path / "pretrained" / "s1.ckpt",
        pretrained_s2_g=tmp_path / "pretrained" / "s2G.pth",
    )


def _job(tmp_path, **overrides):
    defaults = dict(
        dataset_list_path=tmp_path / "dataset.list",
        audio_dir=tmp_path / "audio",
        experiment_dir=tmp_path / "experiment",
        bert_pretrained_dir=tmp_path / "pretrained" / "bert",
        cnhubert_pretrained_dir=tmp_path / "pretrained" / "hubert",
        version="v2Pro",
    )
    defaults.update(overrides)
    return data_prep.DataPrepJob(**defaults)


def test_run_get_text_sets_expected_env_and_invokes_correct_script(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path)

    data_prep.run_get_text(_paths(tmp_path), job, popen_factory=factory)

    call = factory.calls[0]
    assert "GPT_SoVITS/prepare_datasets/1-get-text.py" in call["command"]
    assert call["cwd"] == str(_paths(tmp_path).repo_root)
    env = call["env"]
    assert env["inp_text"] == str(job.dataset_list_path)
    assert env["inp_wav_dir"] == str(job.audio_dir)
    assert env["bert_pretrained_dir"] == str(job.bert_pretrained_dir)
    assert env["i_part"] == "0"
    assert env["all_parts"] == "1"
    assert "cnhubert_base_dir" not in env
    assert "pretrained_s2G" not in env


def test_run_get_text_merges_part_file_into_final_name(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path)
    job.experiment_dir.mkdir(parents=True)
    (job.experiment_dir / "2-name2text-0.txt").write_text("a.wav\thello\n")

    data_prep.run_get_text(_paths(tmp_path), job, popen_factory=factory)

    assert not (job.experiment_dir / "2-name2text-0.txt").exists()
    assert (job.experiment_dir / "2-name2text.txt").read_text() == "a.wav\thello\n"


def test_run_get_text_does_not_merge_when_the_stage_failed(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=["err"], returncode=1)
    job = _job(tmp_path)
    job.experiment_dir.mkdir(parents=True)
    (job.experiment_dir / "2-name2text-0.txt").write_text("stale")

    data_prep.run_get_text(_paths(tmp_path), job, popen_factory=factory)

    assert (job.experiment_dir / "2-name2text-0.txt").read_text() == "stale"
    assert not (job.experiment_dir / "2-name2text.txt").exists()


def test_run_get_hubert_wav32k_sets_expected_env(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path)

    data_prep.run_get_hubert_wav32k(_paths(tmp_path), job, popen_factory=factory)

    call = factory.calls[0]
    assert "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py" in call["command"]
    env = call["env"]
    assert env["cnhubert_base_dir"] == str(job.cnhubert_pretrained_dir)
    assert env["inp_wav_dir"] == str(job.audio_dir)


def test_run_get_sv_sets_hardcoded_sv_model_path(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path)

    data_prep.run_get_sv(paths, job, popen_factory=factory)

    call = factory.calls[0]
    assert "GPT_SoVITS/prepare_datasets/2-get-sv.py" in call["command"]
    assert call["env"]["sv_path"] == str(paths.repo_root / "GPT_SoVITS" / data_prep.SV_MODEL_RELATIVE_PATH)


def test_run_get_semantic_sets_expected_env_and_no_wav_dir(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)
    job = _job(tmp_path, version="v2Pro")

    data_prep.run_get_semantic(paths, job, popen_factory=factory)

    call = factory.calls[0]
    assert "GPT_SoVITS/prepare_datasets/3-get-semantic.py" in call["command"]
    env = call["env"]
    assert env["pretrained_s2G"] == str(paths.pretrained_s2_g)
    assert env["s2config_path"] == str(paths.repo_root / "GPT_SoVITS" / "configs" / "s2v2Pro.json")
    assert "inp_wav_dir" not in env


def test_run_get_semantic_merges_with_header_prepended(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path)
    job.experiment_dir.mkdir(parents=True)
    (job.experiment_dir / "6-name2semantic-0.tsv").write_text("a.wav\t1 2 3\n")

    data_prep.run_get_semantic(_paths(tmp_path), job, popen_factory=factory)

    merged = job.experiment_dir / "6-name2semantic.tsv"
    assert merged.read_text() == "item_name\tsemantic_audio\na.wav\t1 2 3\n"
    assert not (job.experiment_dir / "6-name2semantic-0.tsv").exists()


@pytest.mark.parametrize(
    "version,expected",
    [("v2Pro", True), ("v2ProPlus", True), ("v2", False), ("v3", False), ("v4", False)],
)
def test_requires_speaker_verification_step(version, expected):
    assert data_prep.requires_speaker_verification_step(version) is expected


def test_run_all_prep_steps_runs_sv_step_for_pro_versions(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path, version="v2Pro")

    results = data_prep.run_all_prep_steps(_paths(tmp_path), job, popen_factory=factory)

    scripts_called = [c["command"][2] for c in factory.calls]
    assert "GPT_SoVITS/prepare_datasets/2-get-sv.py" in scripts_called
    assert len(results) == 4
    assert all(r.status == RunStatus.COMPLETED for r in results)


def test_run_all_prep_steps_skips_sv_step_for_non_pro_versions(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)
    job = _job(tmp_path, version="v3")

    results = data_prep.run_all_prep_steps(_paths(tmp_path), job, popen_factory=factory)

    scripts_called = [c["command"][2] for c in factory.calls]
    assert "GPT_SoVITS/prepare_datasets/2-get-sv.py" not in scripts_called
    assert len(results) == 3


def test_prep_script_env_includes_pythonpath_for_repo_root_and_gpt_sovits_dir(
    tmp_path, fake_popen_factory, monkeypatch
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    factory = fake_popen_factory(lines=[], returncode=0)
    paths = _paths(tmp_path)

    data_prep.run_get_text(paths, _job(tmp_path), popen_factory=factory)

    entries = factory.calls[0]["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(paths.repo_root) in entries
    assert str(paths.repo_root / "GPT_SoVITS") in entries


def test_prep_script_env_preserves_existing_pythonpath(tmp_path, fake_popen_factory, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/other/path")
    factory = fake_popen_factory(lines=[], returncode=0)

    data_prep.run_get_hubert_wav32k(_paths(tmp_path), _job(tmp_path), popen_factory=factory)

    entries = factory.calls[0]["env"]["PYTHONPATH"].split(os.pathsep)
    assert "/some/other/path" in entries


def test_default_bert_dir_and_cnhubert_dir_resolve_under_pretrained_models(tmp_path):
    bert_dir = data_prep.default_bert_dir(tmp_path)
    hubert_dir = data_prep.default_cnhubert_dir(tmp_path)

    assert bert_dir == tmp_path / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
    assert hubert_dir == tmp_path / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"


def test_run_all_prep_steps_stops_early_on_failure(tmp_path, fake_popen_factory):
    factory = fake_popen_factory(lines=["boom"], returncode=1)
    job = _job(tmp_path)

    results = data_prep.run_all_prep_steps(_paths(tmp_path), job, popen_factory=factory)

    assert len(results) == 1
    assert results[0].status == RunStatus.FAILED
    assert len(factory.calls) == 1
