from __future__ import annotations

import torch


def cosine_alignment(update_vec: torch.Tensor, mean_update_vec: torch.Tensor) -> float:
    if update_vec.numel() == 0 or mean_update_vec.numel() == 0:
        return 0.0
    denom = torch.norm(update_vec) * torch.norm(mean_update_vec)
    if denom.item() == 0:
        return 0.0
    cos = torch.dot(update_vec, mean_update_vec) / denom
    return float(torch.clamp(cos, min=-1.0, max=1.0).item())


def positive_alignment(update_vec: torch.Tensor, mean_update_vec: torch.Tensor) -> float:
    return max(0.0, cosine_alignment(update_vec, mean_update_vec))

