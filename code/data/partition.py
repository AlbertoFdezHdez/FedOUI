from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ClientPartition:
    client_id: int
    train_indices: list[int]
    label_map: dict[int, int]
    is_rare: bool = False
    is_noisy: bool = False


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def iid_partition(labels: Iterable[int], num_clients: int, seed: int) -> list[list[int]]:
    labels = np.asarray(list(labels))
    rng = _rng(seed)
    indices = np.arange(labels.size)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]


def dirichlet_partition(
    labels: Iterable[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
) -> list[list[int]]:
    labels = np.asarray(list(labels))
    n_classes = int(labels.max()) + 1
    rng = _rng(seed)
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_c) - len(ci)) for p, ci in zip(proportions, client_indices)])
            proportions = np.maximum(proportions, 0)
            if proportions.sum() == 0:
                proportions = np.ones(num_clients)
            proportions = proportions / proportions.sum()
            counts = (proportions * len(idx_c)).astype(int)
            diff = len(idx_c) - counts.sum()
            for i in range(diff):
                counts[i % num_clients] += 1
            start = 0
            for client_id, count in enumerate(counts):
                if count > 0:
                    client_indices[client_id].extend(idx_c[start : start + count].tolist())
                start += count
        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_size:
            for lst in client_indices:
                rng.shuffle(lst)
            return client_indices


def make_label_flip_map(labels: list[int] | np.ndarray, indices: list[int], num_classes: int, noise_rate: float, seed: int) -> dict[int, int]:
    rng = _rng(seed)
    labels = np.asarray(labels)
    n_noisy = int(round(len(indices) * noise_rate))
    if n_noisy <= 0:
        return {}
    noisy_indices = rng.choice(indices, size=n_noisy, replace=False)
    label_map: dict[int, int] = {}
    for idx in noisy_indices:
        current = int(labels[idx])
        choices = list(range(num_classes))
        choices.remove(current)
        label_map[int(idx)] = int(rng.choice(choices))
    return label_map


def build_client_partitions(
    labels: Iterable[int],
    num_clients: int,
    seed: int,
    partition: str = "dirichlet",
    alpha: float = 0.3,
    rare_client_fraction: float = 0.2,
    rare_client_size: int = 120,
    noisy_client_fraction: float = 0.2,
    noise_rate: float = 0.4,
    num_classes: int | None = None,
) -> list[ClientPartition]:
    labels_list = list(labels)
    if num_classes is None:
        num_classes = int(np.max(labels_list)) + 1
    if partition == "iid":
        partitions = iid_partition(labels_list, num_clients, seed)
    elif partition == "dirichlet":
        partitions = dirichlet_partition(labels_list, num_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unsupported partition: {partition}")

    rng = _rng(seed)
    client_specs = [
        ClientPartition(client_id=i, train_indices=list(indices), label_map={}, is_rare=False, is_noisy=False)
        for i, indices in enumerate(partitions)
    ]

    if rare_client_fraction > 0:
        num_rare = max(1, int(round(num_clients * rare_client_fraction)))
        rare_ids = set(rng.choice(num_clients, size=num_rare, replace=False).tolist())
        pool: list[int] = []
        for spec in client_specs:
            if spec.client_id in rare_ids and len(spec.train_indices) > rare_client_size:
                pool.extend(spec.train_indices[rare_client_size:])
                spec.train_indices = spec.train_indices[:rare_client_size]
                spec.is_rare = True
        rng.shuffle(pool)
        if pool:
            normal_ids = [spec.client_id for spec in client_specs if spec.client_id not in rare_ids]
            for idx, item in enumerate(pool):
                target = normal_ids[idx % len(normal_ids)]
                client_specs[target].train_indices.append(item)

    if noisy_client_fraction > 0 and noise_rate > 0:
        num_noisy = max(1, int(round(num_clients * noisy_client_fraction)))
        noisy_ids = set(rng.choice(num_clients, size=num_noisy, replace=False).tolist())
        labels_array = np.asarray(labels_list)
        for spec in client_specs:
            if spec.client_id in noisy_ids and spec.train_indices:
                spec.label_map = make_label_flip_map(labels_array, spec.train_indices, num_classes, noise_rate, seed + spec.client_id + 13)
                spec.is_noisy = bool(spec.label_map)

    for spec in client_specs:
        spec.train_indices = sorted(set(int(i) for i in spec.train_indices))
    return client_specs

