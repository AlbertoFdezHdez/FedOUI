from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.datasets import IndexedDatasetWithLabelMap
from federated.aggregation import ClientRoundResult, model_to_vector
from metrics.oui import compute_model_oui


@dataclass
class Client:
    client_id: int
    train_dataset: torch.utils.data.Dataset
    probe_dataset: torch.utils.data.Dataset
    train_indices: list[int]
    label_map: dict[int, int]
    probe_seed: int = 0
    is_rare: bool = False
    is_noisy: bool = False

    def build_loaders(self, batch_size: int, probe_batch_size: int) -> tuple[DataLoader, torch.Tensor]:
        dataset = IndexedDatasetWithLabelMap(self.train_dataset, self.train_indices, self.label_map)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        probe_indices = self._probe_indices(probe_batch_size)
        probe_dataset = IndexedDatasetWithLabelMap(self.probe_dataset, probe_indices, self.label_map)
        probe_loader = DataLoader(probe_dataset, batch_size=probe_batch_size, shuffle=False, drop_last=False)
        probe_x, _ = next(iter(probe_loader))
        return loader, probe_x

    def _probe_indices(self, probe_batch_size: int) -> list[int]:
        rng = torch.Generator().manual_seed(self.probe_seed + self.client_id * 10007)
        if len(self.train_indices) >= probe_batch_size:
            perm = torch.randperm(len(self.train_indices), generator=rng).tolist()
            return [self.train_indices[i] for i in perm[:probe_batch_size]]
        sampled = torch.randint(0, len(self.train_indices), (probe_batch_size,), generator=rng).tolist()
        return [self.train_indices[i] for i in sampled]

    def train_round(
        self,
        global_model: torch.nn.Module,
        device: torch.device | str,
        batch_size: int,
        probe_batch_size: int,
        local_epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        mu: float,
        method: str,
    ) -> ClientRoundResult:
        device = torch.device(device)
        loader, probe_x = self.build_loaders(batch_size=batch_size, probe_batch_size=probe_batch_size)
        global_model = global_model.to(device)
        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()
        global_vec = model_to_vector(global_model).to(device)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        for _ in range(local_epochs):
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = local_model(x)
                loss = criterion(logits, y)
                if method == "fedprox":
                    prox = 0.0
                    for param, ref in zip(local_model.parameters(), global_model.parameters()):
                        prox = prox + torch.sum((param - ref) ** 2)
                    loss = loss + 0.5 * mu * prox
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(y.size(0))
                total_samples += int(y.size(0))
        update_vec = model_to_vector(local_model).detach().cpu() - global_vec.detach().cpu()
        update_norm = float(torch.norm(update_vec).item())
        oui = compute_model_oui(local_model, probe_x, device=device)
        return ClientRoundResult(
            client_id=self.client_id,
            n_k=len(self.train_indices),
            update_vec=update_vec,
            oui=oui,
            train_loss=total_loss / total_samples if total_samples else 0.0,
            update_norm=update_norm,
        )
