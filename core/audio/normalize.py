"""Loudness normalization (EBU R128 / ITU-R BS.1770 via pyloudnorm)."""

import math

import numpy as np

TARGET_LUFS = -23.0
TRUE_PEAK_CEILING_DB = -1.0


def measure_lufs(samples: np.ndarray, sample_rate: int, pyln_module=None) -> float:
    pyln = pyln_module or _import_pyloudnorm()
    meter = pyln.Meter(sample_rate)
    return meter.integrated_loudness(samples)


def normalize_loudness(
    samples: np.ndarray,
    sample_rate: int,
    target_lufs: float = TARGET_LUFS,
    true_peak_ceiling_db: float = TRUE_PEAK_CEILING_DB,
    pyln_module=None,
) -> np.ndarray:
    pyln = pyln_module or _import_pyloudnorm()
    current_lufs = measure_lufs(samples, sample_rate, pyln_module=pyln)
    if math.isinf(current_lufs):
        return samples  # digital silence - nothing to normalize

    gained = pyln.normalize.loudness(samples, current_lufs, target_lufs)
    return _apply_true_peak_ceiling(gained, true_peak_ceiling_db)


def _apply_true_peak_ceiling(samples: np.ndarray, ceiling_db: float) -> np.ndarray:
    ceiling_linear = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak <= ceiling_linear or peak == 0.0:
        return samples
    return samples * (ceiling_linear / peak)


def _import_pyloudnorm():
    import pyloudnorm as pyln

    return pyln
