from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass
class BetaFitResult:
    alpha: float | None
    beta: float | None
    method: str
    valid: bool


def clip01(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def fit_beta_moments(values: Iterable[float], eps: float = 1e-4) -> BetaFitResult:
    arr = clip01(np.asarray(list(values), dtype=float), eps=eps)
    if arr.size < 2:
        return BetaFitResult(None, None, "moments", False)
    mean = float(arr.mean())
    var = float(arr.var(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(var) or var <= 0:
        return BetaFitResult(None, None, "moments", False)
    common = mean * (1.0 - mean) / var - 1.0
    if not np.isfinite(common) or common <= 0:
        return BetaFitResult(None, None, "moments", False)
    alpha = max(mean * common, 1e-6)
    beta = max((1.0 - mean) * common, 1e-6)
    return BetaFitResult(alpha, beta, "moments", True)


def fit_beta_mle(values: Iterable[float], eps: float = 1e-4) -> BetaFitResult:
    arr = clip01(np.asarray(list(values), dtype=float), eps=eps)
    if arr.size < 2:
        return BetaFitResult(None, None, "mle", False)
    try:
        alpha, beta, _, _ = stats.beta.fit(arr, floc=0.0, fscale=1.0)
    except Exception:
        return BetaFitResult(None, None, "mle", False)
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        return BetaFitResult(None, None, "mle", False)
    return BetaFitResult(float(alpha), float(beta), "mle", True)


def fit_beta(values: Iterable[float], eps: float = 1e-4, prefer: str = "moments") -> BetaFitResult:
    first = fit_beta_moments(values, eps=eps) if prefer == "moments" else fit_beta_mle(values, eps=eps)
    if first.valid:
        return first
    second = fit_beta_mle(values, eps=eps) if prefer == "moments" else fit_beta_moments(values, eps=eps)
    return second


def beta_cdf(values: Iterable[float], alpha: float, beta: float) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return stats.beta.cdf(arr, alpha, beta)


def bilateral_typicity_from_cdf(cdf_values: Iterable[float]) -> np.ndarray:
    cdf = np.asarray(list(cdf_values), dtype=float)
    return 2.0 * np.minimum(cdf, 1.0 - cdf)


def empirical_percentile_typicity(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    ranks = stats.rankdata(arr, method="average")
    percentiles = ranks / (arr.size + 1.0)
    return 2.0 * np.minimum(percentiles, 1.0 - percentiles)


def normalize_nonnegative(values: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= eps:
        return np.ones_like(arr) / max(arr.size, 1)
    return arr / total


def moving_average(values: Iterable[float], window: int) -> list[float]:
    arr = list(values)
    if window <= 1:
        return arr
    out: list[float] = []
    for idx in range(len(arr)):
        start = max(0, idx - window + 1)
        out.append(float(np.mean(arr[start : idx + 1])))
    return out

