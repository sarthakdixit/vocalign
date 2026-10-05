"""Subprocess adapter driving a vendored GPT-SoVITS checkout. Shape confirmed against
the real repo (commit 48b1a0169a28582a8984402f82cf438d3bfa6aca): two training stages,
each a standalone script invoked as `python -s <script> --config <path>` with cwd set
to the GPT-SoVITS checkout root (matches what GPT-SoVITS's own webui.py does).
Pretrained-checkpoint paths are passed in via GptSoVitsPaths rather than hardcoded here -
the real default locations get wired up once the vendoring/download step is settled.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from core.train.runner import RunResult, RunStatus, run_stage


@dataclass(frozen=True)
class GptSoVitsPaths:
    repo_root: Path
    python_executable: str
    pretrained_s1: Path
    pretrained_s2_g: Path
    pretrained_s2_d: Path | None = None


# Confirmed 2026-10-06 against config.py's pretrained_sovits_name/pretrained_gpt_name
# dicts in the real repo (commit 48b1a0169a28582a8984402f82cf438d3bfa6aca), relative
# to GPT_SoVITS/ inside the vendored checkout. v2Pro/v2ProPlus/v3/v4 all share the same
# s1 (GPT-stage) checkpoint - only the SoVITS-stage (s2G) checkpoint differs per version.
PRETRAINED_S2G_BY_VERSION = {
    "v1": "pretrained_models/s2G488k.pth",
    "v2": "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "v2Pro": "pretrained_models/v2Pro/s2Gv2Pro.pth",
    "v2ProPlus": "pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    "v3": "pretrained_models/s2Gv3.pth",
    "v4": "pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
}

PRETRAINED_S1_BY_VERSION = {
    "v1": "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "v2": "pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "v2Pro": "pretrained_models/s1v3.ckpt",
    "v2ProPlus": "pretrained_models/s1v3.ckpt",
    "v3": "pretrained_models/s1v3.ckpt",
    "v4": "pretrained_models/s1v3.ckpt",
}


def default_paths(repo_root: Path, python_executable: str, version: str) -> GptSoVitsPaths:
    if version not in PRETRAINED_S2G_BY_VERSION:
        raise ValueError(f"Unknown GPT-SoVITS version {version!r}")

    gpt_sovits_dir = Path(repo_root) / "GPT_SoVITS"
    s2g = gpt_sovits_dir / PRETRAINED_S2G_BY_VERSION[version]
    s2d_name = s2g.name.replace("s2G", "s2D")
    return GptSoVitsPaths(
        repo_root=Path(repo_root),
        python_executable=python_executable,
        pretrained_s1=gpt_sovits_dir / PRETRAINED_S1_BY_VERSION[version],
        pretrained_s2_g=s2g,
        pretrained_s2_d=(s2g.parent / s2d_name) if s2d_name != s2g.name else None,
    )


@dataclass(frozen=True)
class FineTuneJob:
    dataset_list_path: Path
    experiment_dir: Path
    version: str = "v2Pro"
    gpt_epochs: int = 8
    sovits_epochs: int = 8
    lora_rank: int | None = None
    batch_size: int = 1
    use_fp16: bool = True


def build_s1_config(paths: GptSoVitsPaths, job: FineTuneJob) -> dict:
    return {
        "train": {
            "batch_size": job.batch_size,
            "epochs": job.gpt_epochs,
            "save_every_n_epoch": max(1, job.gpt_epochs // 2),
            "precision": "fp16" if job.use_fp16 else "fp32",
        },
        "pretrained_s1": str(paths.pretrained_s1),
        "train_semantic_path": str(job.experiment_dir / "6-name2semantic.tsv"),
        "train_phoneme_path": str(job.experiment_dir / "2-name2text.txt"),
        "output_dir": str(job.experiment_dir / "s1"),
    }


def build_s2_config(paths: GptSoVitsPaths, job: FineTuneJob) -> dict:
    config = {
        "train": {
            "batch_size": job.batch_size,
            "epochs": job.sovits_epochs,
            "pretrained_s2G": str(paths.pretrained_s2_g),
        },
        "model": {"version": job.version},
        "data": {"exp_dir": str(job.experiment_dir)},
        "save_weight_dir": str(job.experiment_dir / "s2"),
    }
    if paths.pretrained_s2_d is not None:
        config["train"]["pretrained_s2D"] = str(paths.pretrained_s2_d)
    if job.lora_rank is not None:
        config["train"]["lora_rank"] = job.lora_rank
    return config


def run_gpt_stage(
    paths: GptSoVitsPaths, job: FineTuneJob, *, on_progress=None, popen_factory=None, cancel_check=None
) -> RunResult:
    config_path = job.experiment_dir / "s1_config.yaml"
    write_yaml_config(build_s1_config(paths, job), config_path)
    command = [paths.python_executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(config_path)]
    return run_stage(
        command, cwd=str(paths.repo_root), on_progress=on_progress, popen_factory=popen_factory, cancel_check=cancel_check
    )


def run_sovits_stage(
    paths: GptSoVitsPaths, job: FineTuneJob, *, on_progress=None, popen_factory=None, cancel_check=None
) -> RunResult:
    config_path = job.experiment_dir / "s2_config.json"
    write_json_config(build_s2_config(paths, job), config_path)
    script = "GPT_SoVITS/s2_train_v3_lora.py" if job.version in {"v3", "v4"} else "GPT_SoVITS/s2_train.py"
    command = [paths.python_executable, "-s", script, "--config", str(config_path)]
    return run_stage(
        command, cwd=str(paths.repo_root), on_progress=on_progress, popen_factory=popen_factory, cancel_check=cancel_check
    )


def run_fine_tune(
    paths: GptSoVitsPaths, job: FineTuneJob, *, on_progress=None, popen_factory=None, cancel_check=None
) -> tuple[RunResult, RunResult | None]:
    """Runs both stages in sequence; skips stage 2 entirely if stage 1 didn't complete."""
    stage1 = run_gpt_stage(paths, job, on_progress=on_progress, popen_factory=popen_factory, cancel_check=cancel_check)
    if stage1.status is not RunStatus.COMPLETED:
        return stage1, None
    stage2 = run_sovits_stage(paths, job, on_progress=on_progress, popen_factory=popen_factory, cancel_check=cancel_check)
    return stage1, stage2


def write_yaml_config(config: dict, path: Path) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def write_json_config(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path
