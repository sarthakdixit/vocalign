"""Adaptive training-tier selection. Boundaries are research-backed (RESEARCH.md
Finding 2); epoch counts are deliberately small placeholders for an initial real
smoke test, not tuned for quality yet - see DESIGN.md S13 "Recipe-tier step counts/timing".
LoRA is only real for GPT-SoVITS's v3/v4 SoVITS stage (confirmed against the actual
repo) - v2-family always gets lora_rank=None, meaning full fine-tune.
"""

from dataclasses import dataclass

ZERO_SHOT_ONLY_BELOW_SECONDS = 15.0
MINIMAL_BELOW_SECONDS = 60.0
LIGHT_BELOW_SECONDS = 300.0
STANDARD_BELOW_SECONDS = 1200.0

LORA_CAPABLE_VERSIONS = {"v3", "v4"}


@dataclass(frozen=True)
class TrainingRecipe:
    tier: str
    attempt_training: bool
    gpt_epochs: int
    sovits_epochs: int
    lora_rank: int | None


def select_recipe(total_speech_seconds: float, version: str = "v2Pro") -> TrainingRecipe:
    supports_lora = version in LORA_CAPABLE_VERSIONS

    if total_speech_seconds < ZERO_SHOT_ONLY_BELOW_SECONDS:
        return TrainingRecipe("zero_shot_only", False, 0, 0, None)
    if total_speech_seconds < MINIMAL_BELOW_SECONDS:
        return TrainingRecipe("minimal", True, 4, 4, 16 if supports_lora else None)
    if total_speech_seconds < LIGHT_BELOW_SECONDS:
        return TrainingRecipe("light", True, 8, 8, 16 if supports_lora else None)
    if total_speech_seconds < STANDARD_BELOW_SECONDS:
        return TrainingRecipe("standard", True, 15, 15, 32 if supports_lora else None)
    return TrainingRecipe("extended", True, 25, 25, 64 if supports_lora else None)
