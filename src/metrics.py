"""Bit-error counting and binomial confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy import stats


def count_errors(decisions: np.ndarray, truth: np.ndarray) -> int:
    return int(np.count_nonzero(np.sign(decisions) != np.sign(truth)))


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))
