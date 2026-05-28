from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from metrics.alignment import positive_alignment
from metrics.oui import beta_typicity_scores, fit_oui_scores
from utils.stats import normalize_nonnegative


@dataclass
class ClientRoundResult:
    client_id: int
    n_k: int
    update_vec: torch.Tensor
    oui: float
    train_loss: float
    update_norm: float


def model_to_vector(model: torch.nn.Module) -> torch.Tensor:
    parts = [param.detach().flatten().cpu() for param in model.parameters()]
    if not parts:
        return torch.empty(0)
    return torch.cat(parts).clone()


def vector_to_model(model: torch.nn.Module, vector: torch.Tensor) -> None:
    pointer = 0
    vector = vector.detach().to(next(model.parameters()).device)
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            chunk = vector[pointer : pointer + numel].view_as(param)
            param.copy_(chunk)
            pointer += numel


def aggregate_weighted_vectors(vectors: list[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
    if not vectors:
        return torch.empty(0)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    out = torch.zeros_like(vectors[0])
    for vec, weight in zip(vectors, weights):
        out = out + vec * float(weight)
    return out


def _base_weights(client_results: list[ClientRoundResult], rho: float) -> np.ndarray:
    sizes = np.asarray([r.n_k for r in client_results], dtype=float)
    return np.power(np.maximum(sizes, 1.0), rho)


def _historical_oui_values(history: list[dict[str, Any]], window_size: int) -> list[float]:
    if window_size <= 1:
        return []
    recent = history[-(window_size - 1) :]
    values: list[float] = []
    for item in recent:
        values.extend(float(x) for x in item.get("ouis", []))
    return values


def score_ouis_for_round(
    current_ouis: list[float],
    history: list[dict[str, Any]],
    mode: str,
    window: str,
    window_size: int,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_values = list(current_ouis)
    if window == "last5":
        fit_values = _historical_oui_values(history, window_size) + fit_values
    result = fit_oui_scores(fit_values, mode=mode, eps=eps)
    if result.alpha is not None and result.beta is not None and result.score_mode == "beta":
        scores = beta_typicity_scores(current_ouis, result.alpha, result.beta)
        if np.any(~np.isfinite(scores)):
            scores = fit_oui_scores(current_ouis, mode="empirical", eps=eps).scores
            result = fit_oui_scores(current_ouis, mode="empirical", eps=eps)
    else:
        scores = fit_oui_scores(current_ouis, mode="empirical", eps=eps).scores
        result = fit_oui_scores(current_ouis, mode="empirical", eps=eps)
    stats = {
        "alpha": result.alpha,
        "beta": result.beta,
        "score_mode": result.score_mode,
        "fit_method": result.fit_method,
    }
    return np.asarray(scores, dtype=float), stats


def aggregate_round(
    method: str,
    client_results: list[ClientRoundResult],
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not client_results:
        raise ValueError("No client results provided")
    method = method.lower()
    rho = float(config.get("rho", 1.0))
    gamma = float(config.get("gamma", 1.0))
    eta = float(config.get("eta", 1.0))
    eps = float(config.get("epsilon", 1e-3))
    oui_mode = str(config.get("beta_mode", "beta")).lower()
    beta_window = str(config.get("beta_window", "current")).lower()
    beta_window_size = int(config.get("beta_window_size", 5))

    deltas = [r.update_vec for r in client_results]
    sizes = np.asarray([r.n_k for r in client_results], dtype=float)
    base = _base_weights(client_results, rho=rho)
    round_ouis = [float(r.oui) for r in client_results]
    round_stats: dict[str, Any] = {
        "ouis": round_ouis,
        "oui_mean": float(np.mean(round_ouis)) if round_ouis else 0.0,
        "oui_var": float(np.var(round_ouis, ddof=1)) if len(round_ouis) > 1 else 0.0,
        "participants": [r.client_id for r in client_results],
    }

    if method == "fedavg" or method == "fedprox":
        weights = normalize_nonnegative(base)
        delta_agg = aggregate_weighted_vectors(deltas, weights)
        round_stats.update({"weights": weights.tolist(), "scores": [1.0] * len(client_results), "alignments": [None] * len(client_results)})
        return delta_agg, round_stats

    alignment_vec = aggregate_weighted_vectors(deltas, normalize_nonnegative(sizes))
    alignments = [positive_alignment(delta, alignment_vec) for delta in deltas]
    align_factor = np.power(np.maximum(np.asarray(alignments, dtype=float), 0.0) + eps, eta)

    scores = np.ones(len(client_results), dtype=float)
    beta_alpha = None
    beta_beta = None
    score_mode = "none"
    fit_method = "none"
    if method in {"fedoui", "fedoui_align"}:
        scores, score_meta = score_ouis_for_round(
            current_ouis=round_ouis,
            history=history,
            mode=oui_mode,
            window=beta_window,
            window_size=beta_window_size,
            eps=eps,
        )
        beta_alpha = score_meta.get("alpha")
        beta_beta = score_meta.get("beta")
        score_mode = score_meta.get("score_mode", "none")
        fit_method = score_meta.get("fit_method", "none")

    if method == "fedalign":
        combined = base * np.power(np.maximum(np.asarray(alignments, dtype=float), 0.0) + eps, eta)
    elif method == "fedoui":
        combined = base * np.power(np.maximum(scores, 0.0) + eps, gamma)
    elif method == "fedoui_align":
        combined = base * np.power(np.maximum(scores, 0.0) + eps, gamma) * align_factor
    else:
        raise ValueError(f"Unsupported method: {method}")

    weights = normalize_nonnegative(combined)
    delta_agg = aggregate_weighted_vectors(deltas, weights)
    round_stats.update(
        {
            "weights": weights.tolist(),
            "scores": scores.tolist(),
            "alignments": [float(v) for v in alignments],
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "score_mode": score_mode,
            "fit_method": fit_method,
        }
    )
    return delta_agg, round_stats
