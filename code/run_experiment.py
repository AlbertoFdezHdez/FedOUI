from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import load_dataset, select_subset
from data.partition import build_client_partitions
from federated.client import Client
from federated.server import FederatedServer
from metrics.evaluation import evaluate
from models.cnn_small import build_small_cnn
from models.mlp import build_mlp
from models.tiny_cnn import build_tiny_cnn
from utils.io import append_jsonl, ensure_dir, write_json, write_yaml
from utils.logging_utils import save_run_metadata, setup_logging
from utils.seed import seed_everything


def _build_model(model_name: str, num_classes: int, input_channels: int, image_size: int):
    model_name = model_name.lower()
    if model_name in {"small_cnn", "cnn_small", "cnn"}:
        model = build_small_cnn(num_classes=num_classes)
        model._init_args = (num_classes,)
        return model
    if model_name in {"tiny_cnn", "adaptive_cnn"}:
        model = build_tiny_cnn(input_channels=input_channels, num_classes=num_classes)
        model._init_args = (input_channels, num_classes)
        return model
    if model_name in {"mlp", "simple_mlp"}:
        input_dim = image_size * image_size * input_channels
        model = build_mlp(input_dim=input_dim, num_classes=num_classes)
        model._init_args = (input_dim, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def _save_round_row(run_dir: Path, row: dict[str, Any]) -> None:
    append_jsonl(run_dir / "rounds.jsonl", row)


def _make_test_loader(test_dataset, batch_size: int) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def run_single_experiment(config: dict[str, Any]) -> dict[str, Any]:
    exp = config["experiment"] if "experiment" in config else config
    seed = int(exp.get("seed", 1))
    seed_everything(seed)
    device = torch.device(exp.get("device", "cpu"))
    dataset_name = exp.get("dataset", "cifar10")
    output_root = Path(exp.get("output_root", "descargas"))
    scenario = exp.get("scenario", "default_scenario")
    method = exp.get("method", "fedavg")
    run_name = f"{method}_seed_{seed}"
    run_dir = ensure_dir(output_root / dataset_name / scenario / run_name)
    logger = setup_logging(run_dir / "run.log")
    save_run_metadata(run_dir, config)

    bundle = load_dataset(dataset_name, exp.get("data_root", "data"), download=bool(exp.get("download", True)))
    test_dataset = select_subset(bundle.test_dataset, exp.get("test_subset_size"), seed=seed + 999)

    base_train_indices = list(range(len(bundle.train_dataset)))
    train_subset_size = exp.get("train_subset_size")
    if train_subset_size is not None:
        subset_rng = np.random.default_rng(seed + 12345)
        base_train_indices = subset_rng.permutation(base_train_indices).tolist()[: int(train_subset_size)]
    subset_labels = [int(bundle.train_dataset[i][1]) for i in base_train_indices]
    partitions = build_client_partitions(
        labels=subset_labels,
        num_clients=int(exp.get("num_clients", 20)),
        seed=seed,
        partition=str(exp.get("partition", "dirichlet")),
        alpha=float(exp.get("alpha", 0.3)),
        rare_client_fraction=float(exp.get("rare_client_fraction", 0.2)),
        rare_client_size=int(exp.get("rare_client_size", 120)),
        noisy_client_fraction=float(exp.get("noisy_client_fraction", 0.2)),
        noise_rate=float(exp.get("noise_rate", 0.4)),
        num_classes=bundle.num_classes,
    )

    clients = []
    for spec in partitions:
        mapped_indices = [base_train_indices[i] for i in spec.train_indices]
        mapped_label_map = {base_train_indices[k]: v for k, v in spec.label_map.items()}
        client = Client(
            client_id=spec.client_id,
            train_dataset=bundle.train_dataset,
            probe_dataset=bundle.train_dataset,
            train_indices=mapped_indices,
            label_map=mapped_label_map,
            probe_seed=seed,
            is_rare=spec.is_rare,
            is_noisy=spec.is_noisy,
        )
        clients.append(client)

    model = _build_model(str(exp.get("model", "small_cnn")), bundle.num_classes, bundle.input_channels, bundle.image_size).to(device)
    server = FederatedServer(model=model, method=str(method), config=exp)
    test_loader = _make_test_loader(test_dataset, batch_size=int(exp.get("test_batch_size", 256)))

    participants_per_round = int(exp.get("participants_per_round", 5))
    rounds = int(exp.get("rounds", 100))
    local_epochs = int(exp.get("local_epochs", 1))
    batch_size = int(exp.get("batch_size", 32))
    probe_batch_size = int(exp.get("probe_batch_size", 32))
    lr = float(exp.get("lr", 0.01))
    momentum = float(exp.get("momentum", 0.9))
    weight_decay = float(exp.get("weight_decay", 0.0))
    mu = float(exp.get("mu", 0.01))

    rng = np.random.default_rng(seed)
    round_history: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    best_accuracy = 0.0
    best_round = -1

    for round_idx in tqdm(range(rounds), desc=f"{method}:{scenario}:seed{seed}"):
        participant_ids = rng.choice(len(clients), size=min(participants_per_round, len(clients)), replace=False).tolist()
        client_results = []
        for cid in participant_ids:
            client = clients[cid]
            result = client.train_round(
                global_model=copy.deepcopy(server.model).to(device),
                device=device,
                batch_size=batch_size,
                probe_batch_size=probe_batch_size,
                local_epochs=local_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                mu=mu,
                method=str(method).lower(),
            )
            client_results.append(result)

        stats = server.apply_round(client_results)
        test_metrics = evaluate(server.model.to(device), test_loader, device=device)
        accuracy = float(test_metrics["accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_round = round_idx

        round_payload = {
            "round": round_idx,
            "participants": participant_ids,
            "test_accuracy": accuracy,
            "test_loss": float(test_metrics["loss"]),
            "best_accuracy_so_far": best_accuracy,
            "best_round_so_far": best_round,
            "beta_alpha": stats.get("beta_alpha"),
            "beta_beta": stats.get("beta_beta"),
            "oui_mean": stats.get("oui_mean"),
            "oui_var": stats.get("oui_var"),
            "score_mode": stats.get("score_mode"),
            "fit_method": stats.get("fit_method"),
        }
        _save_round_row(run_dir, round_payload)
        round_rows.append(round_payload)
        round_history.append(stats)

        for res, w, s, a in zip(
            client_results,
            stats.get("weights", []),
            stats.get("scores", []),
            stats.get("alignments", []),
        ):
            participant_rows.append(
                {
                    "round": round_idx,
                    "client_id": res.client_id,
                    "n_k": res.n_k,
                    "oui": res.oui,
                    "score_s": s,
                    "alignment_a": a,
                    "weight": w,
                    "update_norm": res.update_norm,
                    "train_loss": res.train_loss,
                }
            )

        logger.info(
            "round=%s acc=%.4f best=%.4f oui_mean=%.4f",
            round_idx,
            accuracy,
            best_accuracy,
            float(stats.get("oui_mean", 0.0)),
        )

    summary = {
        "dataset": dataset_name,
        "model": str(exp.get("model", "small_cnn")),
        "method": str(method),
        "scenario": scenario,
        "seed": seed,
        "num_clients": int(exp.get("num_clients", 20)),
        "participants_per_round": participants_per_round,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "lr": lr,
        "momentum": momentum,
        "mu": mu,
        "gamma": float(exp.get("gamma", 1.0)),
        "eta": float(exp.get("eta", 1.0)),
        "rho": float(exp.get("rho", 1.0)),
        "epsilon": float(exp.get("epsilon", 1e-3)),
        "beta_mode": str(exp.get("beta_mode", "beta")),
        "probe_batch_size": probe_batch_size,
        "oui_layer": str(exp.get("oui_layer", "penultimate_preact")),
        "partition": str(exp.get("partition", "dirichlet")),
        "alpha": exp.get("alpha"),
        "rare_client_fraction": float(exp.get("rare_client_fraction", 0.2)),
        "rare_client_size": int(exp.get("rare_client_size", 120)),
        "noisy_client_fraction": float(exp.get("noisy_client_fraction", 0.2)),
        "noise_rate": float(exp.get("noise_rate", 0.4)),
        "final_accuracy": float(round_rows[-1]["test_accuracy"]) if round_rows else 0.0,
        "best_accuracy": float(best_accuracy),
        "best_round": int(best_round),
        "run_dir": str(run_dir),
        "round_history_path": str(run_dir / "rounds.jsonl"),
        "participant_rows": len(participant_rows),
    }

    write_json(run_dir / "summary.json", summary)
    write_yaml(run_dir / "summary.yaml", summary)
    write_json(run_dir / "participant_metrics.json", participant_rows)
    write_json(run_dir / "round_metrics.json", round_rows)
    logger.info("finished run %s", summary)
    return summary


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out
