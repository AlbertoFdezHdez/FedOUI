from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        penultimate_preact = self.fc1(x)
        penultimate_act = self.relu(penultimate_preact)
        logits = self.fc2(penultimate_act)
        return logits, penultimate_preact, penultimate_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_features(x)
        return logits

    def forward_with_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, penultimate_preact, penultimate_act = self.forward_features(x)
        return logits, {
            "penultimate_preact": penultimate_preact,
            "penultimate_act": penultimate_act,
        }


def build_small_cnn(num_classes: int = 10) -> SmallCNN:
    return SmallCNN(num_classes=num_classes)

