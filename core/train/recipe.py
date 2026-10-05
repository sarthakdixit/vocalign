"""Adaptive training-tier selection for the GPT (text-to-semantic) stage - the only
stage fine-tuned locally (see DESIGN.md S13, 2026-10-06 decision: the SoVITS/acoustic
stage doesn't fit this hardware's VRAM even at the smallest tier, confirmed with real
training attempts, so it stays at its pretrained checkpoint with zero-shot reference
conditioning at inference time instead). Tier boundaries are research-backed
(RESEARCH.md Finding 2); epoch counts are deliberately small placeholders for an
initial real smoke test, not tuned for quality yet - see DESIGN.md S13.
"""

from dataclasses import dataclass

ZERO_SHOT_ONLY_BELOW_SECONDS = 15.0
MINIMAL_BELOW_SECONDS = 60.0
LIGHT_BELOW_SECONDS = 300.0
STANDARD_BELOW_SECONDS = 1200.0


@dataclass(frozen=True)
class TrainingRecipe:
    tier: str
    attempt_training: bool
    gpt_epochs: int


def select_recipe(total_speech_seconds: float) -> TrainingRecipe:
    if total_speech_seconds < ZERO_SHOT_ONLY_BELOW_SECONDS:
        return TrainingRecipe("zero_shot_only", False, 0)
    if total_speech_seconds < MINIMAL_BELOW_SECONDS:
        return TrainingRecipe("minimal", True, 4)
    if total_speech_seconds < LIGHT_BELOW_SECONDS:
        return TrainingRecipe("light", True, 8)
    if total_speech_seconds < STANDARD_BELOW_SECONDS:
        return TrainingRecipe("standard", True, 15)
    return TrainingRecipe("extended", True, 25)
