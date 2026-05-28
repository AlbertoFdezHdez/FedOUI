from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int
    input_channels: int
    image_size: int
    class_names: list[str]


class IndexedDatasetWithLabelMap(Dataset):
    def __init__(self, base_dataset: Dataset, indices: list[int], label_map: dict[int, int] | None = None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_map = dict(label_map or {})

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        x, y = self.base_dataset[base_idx]
        if base_idx in self.label_map:
            y = self.label_map[base_idx]
        return x, y


def _cifar10_transforms() -> tuple[Any, Any]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, test_tf


def _grayscale_transforms(mean: float, std: float) -> tuple[Any, Any]:
    train_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return train_tf, test_tf


def _emnist_transforms() -> tuple[Any, Any]:
    mean = (0.1736,)
    std = (0.3317,)
    base = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return base, base


def _food101_transforms(image_size: int = 64) -> tuple[Any, Any]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, test_tf


def load_dataset(name: str, root: str | Path, download: bool = True) -> DatasetBundle:
    name = name.lower()
    root = Path(root)
    if name == "cifar10":
        train_tf, test_tf = _cifar10_transforms()
        train = datasets.CIFAR10(root=root, train=True, transform=train_tf, download=download)
        test = datasets.CIFAR10(root=root, train=False, transform=test_tf, download=download)
        return DatasetBundle(train, test, 10, 3, 32, list(train.classes))
    if name in {"mnist", "fashion_mnist", "fashion-mnist"}:
        if name == "mnist":
            train_tf, test_tf = _grayscale_transforms(0.1307, 0.3081)
            train = datasets.MNIST(root=root, train=True, transform=train_tf, download=download)
            test = datasets.MNIST(root=root, train=False, transform=test_tf, download=download)
            return DatasetBundle(train, test, 10, 1, 28, list(train.classes))
        train_tf, test_tf = _grayscale_transforms(0.2860, 0.3530)
        train = datasets.FashionMNIST(root=root, train=True, transform=train_tf, download=download)
        test = datasets.FashionMNIST(root=root, train=False, transform=test_tf, download=download)
        return DatasetBundle(train, test, 10, 1, 28, list(train.classes))
    if name in {"emnist", "emnist_balanced", "emnist-letters", "femnist"}:
        train_tf, test_tf = _emnist_transforms()
        split = "balanced" if name in {"emnist", "emnist_balanced", "femnist"} else "letters"
        train = datasets.EMNIST(root=root, split=split, train=True, transform=train_tf, download=download)
        test = datasets.EMNIST(root=root, split=split, train=False, transform=test_tf, download=download)
        num_classes = 47 if split == "balanced" else 26
        return DatasetBundle(train, test, num_classes, 1, 28, list(getattr(train, "classes", [])))
    if name == "food101":
        train_tf, test_tf = _food101_transforms(image_size=64)
        train = datasets.Food101(root=root, split="train", transform=train_tf, download=download)
        test = datasets.Food101(root=root, split="test", transform=test_tf, download=download)
        return DatasetBundle(train, test, 101, 3, 64, list(getattr(train, "classes", [])))
    raise ValueError(f"Unsupported dataset: {name}")


def select_subset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None:
        return dataset
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[:subset_size]
    return IndexedDatasetWithLabelMap(dataset, indices)
