"""No-reference automatic naturalness/MOS prediction via utmos-pytorch (a fairseq-free
reimplementation - the original `utmos` PyPI package depends on fairseq, a known
problematic install on modern Python/PyTorch, per RESEARCH.md Finding 4's update).
Used only for ranking/reporting, never as a training objective (UTMOS is known to be
adversarially gameable if optimized against directly).
"""

import numpy as np


def predict_mos(samples: np.ndarray, sample_rate: int, scorer=None) -> float:
    scorer = scorer or _default_scorer()
    return float(scorer.score(samples, sample_rate))


def _default_scorer():
    # utmos-pytorch's exact import/API is not independently confirmed the way the
    # GPT-SoVITS integration was - low blast radius if wrong, isolated to this one
    # function; adjust here on first real use if the real package shape differs.
    import utmos_pytorch

    return utmos_pytorch.Scorer()
