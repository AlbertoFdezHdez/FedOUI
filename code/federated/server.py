from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from federated.aggregation import aggregate_round, model_to_vector, vector_to_model


@dataclass
class FederatedServer:
    model: torch.nn.Module
    method: str
    config: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)

    def apply_round(self, client_results):
        delta, stats = aggregate_round(self.method, client_results, self.config, self.history)
        current_vec = model_to_vector(self.model)
        vector_to_model(self.model, current_vec + delta)
        self.history.append(stats)
        return stats

