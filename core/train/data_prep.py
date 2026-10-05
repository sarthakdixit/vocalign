"""Drives GPT-SoVITS's 4 data-prep scripts (GPT_SoVITS/prepare_datasets/). Env-var
interface and merge behavior confirmed directly against webui.py's open1a/open1b/open1c
functions (commit 48b1a0169a28582a8984402f82cf438d3bfa6aca). We always run as a single
part (i_part=0, all_parts=1) since we only ever target one GPU - no multi-process
splitting. Env vars are passed via a merged dict to Popen rather than mutating our own
process's os.environ, unlike webui.py itself.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from core.train.gpt_sovits_adapter import GptSoVitsPaths
from core.train.runner import RunResult, RunStatus, run_stage

# Hardcoded in webui.py itself, inside the same pretrained_models/sv/ directory the
# installers use as their "already downloaded" sentinel. Only needed for v2Pro/v2ProPlus.
SV_MODEL_RELATIVE_PATH = "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

# Shared, version-independent deps (same for every GPT-SoVITS version).
BERT_PRETRAINED_RELATIVE_PATH = "pretrained_models/chinese-roberta-wwm-ext-large"
CNHUBERT_PRETRAINED_RELATIVE_PATH = "pretrained_models/chinese-hubert-base"


def default_bert_dir(repo_root) -> Path:
    return Path(repo_root) / "GPT_SoVITS" / BERT_PRETRAINED_RELATIVE_PATH


def default_cnhubert_dir(repo_root) -> Path:
    return Path(repo_root) / "GPT_SoVITS" / CNHUBERT_PRETRAINED_RELATIVE_PATH


@dataclass(frozen=True)
class DataPrepJob:
    dataset_list_path: Path
    audio_dir: Path
    experiment_dir: Path
    bert_pretrained_dir: Path
    cnhubert_pretrained_dir: Path
    s2_config_path: Path
    version: str = "v2Pro"
    use_fp16: bool = True
    cuda_visible_devices: str = "0"


def requires_speaker_verification_step(version: str) -> bool:
    return "Pro" in version


def run_get_text(paths: GptSoVitsPaths, job: DataPrepJob, **kwargs) -> RunResult:
    env = _base_env(job) | {
        "inp_wav_dir": str(job.audio_dir),
        "bert_pretrained_dir": str(job.bert_pretrained_dir),
    }
    result = _run_prep_script(paths, "1-get-text.py", env, **kwargs)
    if result.status is RunStatus.COMPLETED:
        _merge_part_file(job.experiment_dir / "2-name2text-0.txt", job.experiment_dir / "2-name2text.txt")
    return result


def run_get_hubert_wav32k(paths: GptSoVitsPaths, job: DataPrepJob, **kwargs) -> RunResult:
    env = _base_env(job) | {
        "inp_wav_dir": str(job.audio_dir),
        "cnhubert_base_dir": str(job.cnhubert_pretrained_dir),
    }
    return _run_prep_script(paths, "2-get-hubert-wav32k.py", env, **kwargs)


def run_get_sv(paths: GptSoVitsPaths, job: DataPrepJob, **kwargs) -> RunResult:
    env = _base_env(job) | {
        "inp_wav_dir": str(job.audio_dir),
        "sv_path": str(Path(paths.repo_root) / "GPT_SoVITS" / SV_MODEL_RELATIVE_PATH),
    }
    return _run_prep_script(paths, "2-get-sv.py", env, **kwargs)


def run_get_semantic(paths: GptSoVitsPaths, job: DataPrepJob, **kwargs) -> RunResult:
    env = _base_env(job) | {
        "pretrained_s2G": str(paths.pretrained_s2_g),
        "s2config_path": str(job.s2_config_path),
    }
    result = _run_prep_script(paths, "3-get-semantic.py", env, **kwargs)
    if result.status is RunStatus.COMPLETED:
        _merge_semantic_part_file(
            job.experiment_dir / "6-name2semantic-0.tsv", job.experiment_dir / "6-name2semantic.tsv"
        )
    return result


def run_all_prep_steps(paths: GptSoVitsPaths, job: DataPrepJob, **kwargs) -> list[RunResult]:
    """Runs the prep steps in the confirmed order (text -> hubert/wav32k -> [speaker
    verification, v2Pro/v2ProPlus only] -> semantic), stopping as soon as one fails."""
    results = [run_get_text(paths, job, **kwargs)]
    if results[-1].status is not RunStatus.COMPLETED:
        return results

    results.append(run_get_hubert_wav32k(paths, job, **kwargs))
    if results[-1].status is not RunStatus.COMPLETED:
        return results

    if requires_speaker_verification_step(job.version):
        results.append(run_get_sv(paths, job, **kwargs))
        if results[-1].status is not RunStatus.COMPLETED:
            return results

    results.append(run_get_semantic(paths, job, **kwargs))
    return results


def _base_env(job: DataPrepJob) -> dict:
    return {
        "inp_text": str(job.dataset_list_path),
        "exp_name": job.experiment_dir.name,
        "opt_dir": str(job.experiment_dir),
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": job.cuda_visible_devices,
        "is_half": str(job.use_fp16),
    }


def _run_prep_script(
    paths: GptSoVitsPaths,
    script_name: str,
    script_env: dict,
    *,
    on_progress=None,
    popen_factory=None,
    cancel_check=None,
) -> RunResult:
    merged_env = {**os.environ, **script_env}
    command = [paths.python_executable, "-s", f"GPT_SoVITS/prepare_datasets/{script_name}"]
    return run_stage(
        command,
        cwd=str(paths.repo_root),
        env=merged_env,
        on_progress=on_progress,
        popen_factory=popen_factory,
        cancel_check=cancel_check,
    )


def _merge_part_file(part_path: Path, merged_path: Path) -> None:
    if part_path.exists():
        part_path.replace(merged_path)


def _merge_semantic_part_file(part_path: Path, merged_path: Path) -> None:
    if part_path.exists():
        content = part_path.read_text()
        merged_path.write_text("item_name\tsemantic_audio\n" + content)
        part_path.unlink()
