from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=False)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.flatten(x)
        hidden1 = self.fc1(x)
        hidden2_preact = self.fc2(self.relu(hidden1))
        hidden2_act = self.relu(hidden2_preact)
        logits = self.fc3(hidden2_act)
        return logits, hidden2_preact, hidden2_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_features(x)
        return logits

    def forward_with_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, hidden2_preact, hidden2_act = self.forward_features(x)
        return logits, {
            "penultimate_preact": hidden2_preact,
            "penultimate_act": hidden2_act,
        }


def build_mlp(input_dim: int = 28 * 28, num_classes: int = 10) -> SimpleMLP:
    return SimpleMLP(input_dim=input_dim, num_classes=num_classes)

