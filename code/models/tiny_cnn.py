from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.adapt(x)
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


def build_tiny_cnn(input_channels: int = 1, num_classes: int = 10) -> TinyCNN:
    return TinyCNN(input_channels=input_channels, num_classes=num_classes)
