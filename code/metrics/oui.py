from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats

from utils.stats import (
    BetaFitResult,
    bilateral_typicity_from_cdf,
    empirical_percentile_typicity,
    fit_beta,
)


@dataclass
class OUIFitOutput:
    scores: np.ndarray
    alpha: float | None
    beta: float | None
    fit_method: str
    score_mode: str


def compute_oui_from_preactivations(preacts: torch.Tensor) -> float:
    if preacts.ndim != 2:
        preacts = preacts.flatten(start_dim=1)
    batch_size = int(preacts.shape[0])
    if batch_size < 2:
        return 0.0
    masks = (preacts > 0).to(dtype=torch.float32)
    s = masks.sum(dim=0)
    u = torch.minimum(s, torch.tensor(float(batch_size), device=preacts.device) - s)
    denom = max(batch_size // 2, 1)
    oui = (u / float(denom)).mean().item()
    return float(np.clip(oui, 0.0, 1.0))


def compute_model_oui(model: torch.nn.Module, probe_x: torch.Tensor, device: torch.device | str) -> float:
    model.eval()
    probe_x = probe_x.to(device)
    with torch.no_grad():
        _, cache = model.forward_with_cache(probe_x)
    preacts = cache["penultimate_preact"]
    return compute_oui_from_preactivations(preacts.detach().cpu())


def beta_typicity_scores(values: list[float] | np.ndarray, alpha: float, beta: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, 1e-4, 1.0 - 1e-4)
    cdf = stats.beta.cdf(clipped, alpha, beta)
    scores = bilateral_typicity_from_cdf(cdf)
    return np.asarray(scores, dtype=float)


def fit_oui_scores(
    ouis: list[float],
    mode: str = "beta",
    eps: float = 1e-3,
    prefer_beta_method: str = "moments",
) -> OUIFitOutput:
    values = np.asarray(ouis, dtype=float)
    if values.size == 0:
        return OUIFitOutput(np.asarray([]), None, None, "none", "none")
    if mode == "beta":
        fit = fit_beta(values, eps=eps, prefer=prefer_beta_method)
        if fit.valid and fit.alpha is not None and fit.beta is not None:
            scores = beta_typicity_scores(values, fit.alpha, fit.beta)
            if np.any(~np.isfinite(scores)):
                scores = empirical_percentile_typicity(values)
                return OUIFitOutput(scores, None, None, "empirical", "fallback_empirical")
            return OUIFitOutput(scores, fit.alpha, fit.beta, fit.method, "beta")
    scores = empirical_percentile_typicity(values)
    return OUIFitOutput(scores, None, None, "empirical", "percentile")
