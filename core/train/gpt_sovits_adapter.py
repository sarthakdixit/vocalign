"""Subprocess adapter driving a vendored GPT-SoVITS checkout. Shape confirmed against
the real repo (commit 48b1a0169a28582a8984402f82cf438d3bfa6aca): two training stages,
each a standalone script invoked as `python -s <script> --config <path>` with cwd set
to the GPT-SoVITS checkout root (matches what GPT-SoVITS's own webui.py does).

Config building loads GPT-SoVITS's OWN real template config (s1longer.yaml /
s1longer-v2.yaml for the GPT stage, s2.json / s2v2Pro.json / s2v2ProPlus.json for the
SoVITS stage) and overlays only the keys webui.py's own open1Bb/open1Ba functions
overlay - NOT a config built from scratch, which is what caused a real
`AttributeError: 'HParams' object has no attribute 'filter_length'` on the first
real-hardware run (a hand-built config was missing the real acoustic hyperparameters).
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from core.train.runner import RunResult, RunStatus, run_stage

# Confirmed 2026-10-06 against config.py's pretrained_sovits_name/pretrained_gpt_name
# dicts in the real repo, relative to GPT_SoVITS/ inside the vendored checkout.
# v2Pro/v2ProPlus/v3/v4 all share the same s1 (GPT-stage) checkpoint - only the
# SoVITS-stage (s2G) checkpoint differs per version.
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

# Confirmed 2026-10-06 directly against webui.py's own template-selection lines.
# Only v2Pro/v2ProPlus get their own SoVITS template; v1/v2/v3/v4 all load plain
# s2.json (s2_train_v3_lora.py overrides model.version/train.lora_rank afterward).
S1_TEMPLATE_BY_VERSION = {
    "v1": "s1longer.yaml",
    "v2": "s1longer-v2.yaml",
    "v2Pro": "s1longer-v2.yaml",
    "v2ProPlus": "s1longer-v2.yaml",
    "v3": "s1longer-v2.yaml",
    "v4": "s1longer-v2.yaml",
}

S2_TEMPLATE_BY_VERSION = {
    "v1": "s2.json",
    "v2": "s2.json",
    "v2Pro": "s2v2Pro.json",
    "v2ProPlus": "s2v2ProPlus.json",
    "v3": "s2.json",
    "v4": "s2.json",
}


@dataclass(frozen=True)
class GptSoVitsPaths:
    repo_root: Path
    python_executable: str
    pretrained_s1: Path
    pretrained_s2_g: Path
    pretrained_s2_d: Path | None = None


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


def s2_template_path(repo_root, version: str) -> Path:
    """The raw, unmodified template path. 3-get-semantic.py's s2config_path points
    directly at this per webui.py's open1c - that script does not get a mutated
    temp config at all, unlike the two training stages."""
    if version not in S2_TEMPLATE_BY_VERSION:
        raise ValueError(f"Unknown GPT-SoVITS version {version!r}")
    return Path(repo_root) / "GPT_SoVITS" / "configs" / S2_TEMPLATE_BY_VERSION[version]


def load_s1_template(repo_root, version: str) -> dict:
    import yaml

    if version not in S1_TEMPLATE_BY_VERSION:
        raise ValueError(f"Unknown GPT-SoVITS version {version!r}")
    template_path = Path(repo_root) / "GPT_SoVITS" / "configs" / S1_TEMPLATE_BY_VERSION[version]
    return yaml.full_load(template_path.read_text())


def load_s2_template(repo_root, version: str) -> dict:
    return json.loads(s2_template_path(repo_root, version).read_text())


def build_s1_config(paths: GptSoVitsPaths, job: FineTuneJob, template: dict | None = None) -> dict:
    config = copy.deepcopy(template if template is not None else load_s1_template(paths.repo_root, job.version))
    config["train"]["batch_size"] = job.batch_size
    config["train"]["epochs"] = job.gpt_epochs
    config["train"]["save_every_n_epoch"] = max(1, job.gpt_epochs // 2)
    config["train"]["if_save_every_weights"] = True
    config["train"]["if_save_latest"] = True
    config["train"]["if_dpo"] = False
    config["train"]["half_weights_save_dir"] = str(job.experiment_dir / "s1_weights")
    config["train"]["exp_name"] = job.experiment_dir.name
    if not job.use_fp16:
        config["train"]["precision"] = "32"
    config["pretrained_s1"] = str(paths.pretrained_s1)
    config["train_semantic_path"] = str(job.experiment_dir / "6-name2semantic.tsv")
    config["train_phoneme_path"] = str(job.experiment_dir / "2-name2text.txt")
    config["output_dir"] = str(job.experiment_dir / "s1")
    return config


def build_s2_config(paths: GptSoVitsPaths, job: FineTuneJob, template: dict | None = None) -> dict:
    config = copy.deepcopy(template if template is not None else load_s2_template(paths.repo_root, job.version))
    config["train"]["batch_size"] = job.batch_size
    config["train"]["epochs"] = job.sovits_epochs
    config["train"]["pretrained_s2G"] = str(paths.pretrained_s2_g)
    config["train"]["if_save_latest"] = True
    config["train"]["if_save_every_weights"] = True
    config["train"]["save_every_epoch"] = max(1, job.sovits_epochs // 2)
    config["train"]["grad_ckpt"] = False
    config["model"]["version"] = job.version
    config["data"]["exp_dir"] = str(job.experiment_dir)
    config["s2_ckpt_dir"] = str(job.experiment_dir)
    config["save_weight_dir"] = str(job.experiment_dir / "s2_weights")
    config["name"] = job.experiment_dir.name
    config["version"] = job.version
    if paths.pretrained_s2_d is not None:
        config["train"]["pretrained_s2D"] = str(paths.pretrained_s2_d)
    if job.lora_rank is not None:
        config["train"]["lora_rank"] = job.lora_rank
    return config


def run_gpt_stage(
    paths: GptSoVitsPaths, job: FineTuneJob, *, on_progress=None, popen_factory=None, cancel_check=None
) -> RunResult:
    config = build_s1_config(paths, job)
    # s1_train.py shutil.move()s checkpoints into half_weights_save_dir without ever
    # creating it itself - confirmed the hard way on a real run (FileNotFoundError).
    Path(config["train"]["half_weights_save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    config_path = job.experiment_dir / "s1_config.yaml"
    write_yaml_config(config, config_path)
    command = [paths.python_executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(config_path)]
    return run_stage(
        command, cwd=str(paths.repo_root), on_progress=on_progress, popen_factory=popen_factory, cancel_check=cancel_check
    )


def run_sovits_stage(
    paths: GptSoVitsPaths, job: FineTuneJob, *, on_progress=None, popen_factory=None, cancel_check=None
) -> RunResult:
    config = build_s2_config(paths, job)
    Path(config["save_weight_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["exp_dir"]).mkdir(parents=True, exist_ok=True)
    config_path = job.experiment_dir / "s2_config.json"
    write_json_config(config, config_path)
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
