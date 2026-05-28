from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.package_results import package_runs, package_scenario_summaries  # noqa: E402
from analysis.summarize_runs import write_outputs  # noqa: E402
from run_experiment import load_config, run_single_experiment  # noqa: E402


METHODS = ["fedavg", "fedprox", "fedalign", "fedoui"]


@dataclass(frozen=True)
class SmokeSpec:
    name: str
    dataset: str
    model: str
    scenario: str
    alpha: float
    num_clients: int
    participants_per_round: int
    train_subset_size: int
    test_subset_size: int
    rounds: int
    local_epochs: int = 1
    batch_size: int = 32
    test_batch_size: int = 256
    probe_batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    mu: float = 0.01
    gamma: float = 1.0
    eta: float = 1.0
    rho: float = 1.0
    epsilon: float = 1e-3
    beta_mode: str = "beta"
    beta_window: str = "current"
    beta_window_size: int = 5
    oui_layer: str = "penultimate_preact"
    partition: str = "dirichlet"
    rare_client_fraction: float = 0.0
    rare_client_size: int = 120
    noisy_client_fraction: float = 0.0
    noise_rate: float = 0.0
    device: str = "cpu"
    data_root: str = "data"


SMOKE_SPECS = [
    SmokeSpec(
        name="mnist_mlp",
        dataset="mnist",
        model="mlp",
        scenario="smoke_dirichlet_0.5",
        alpha=0.5,
        num_clients=20,
        participants_per_round=5,
        train_subset_size=1000,
        test_subset_size=200,
        rounds=1,
    ),
    SmokeSpec(
        name="fashion_mnist_cnn",
        dataset="fashion_mnist",
        model="tiny_cnn",
        scenario="smoke_dirichlet_0.5",
        alpha=0.5,
        num_clients=20,
        participants_per_round=5,
        train_subset_size=1000,
        test_subset_size=200,
        rounds=1,
    ),
    SmokeSpec(
        name="femnist_proxy_mlp",
        dataset="femnist",
        model="mlp",
        scenario="smoke_dirichlet_0.5",
        alpha=0.5,
        num_clients=20,
        participants_per_round=5,
        train_subset_size=1500,
        test_subset_size=300,
        rounds=1,
    ),
    SmokeSpec(
        name="food101_tinycnn",
        dataset="food101",
        model="tiny_cnn",
        scenario="smoke_dirichlet_0.8",
        alpha=0.8,
        num_clients=10,
        participants_per_round=5,
        train_subset_size=1500,
        test_subset_size=300,
        rounds=1,
        batch_size=24,
        probe_batch_size=24,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FedOUI smoke suite on 4 datasets")
    parser.add_argument("--seed", type=int, default=1, help="Seed used for every smoke run")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds for every smoke run")
    parser.add_argument("--train-subset-size", type=int, default=None, help="Override train subset size")
    parser.add_argument("--test-subset-size", type=int, default=None, help="Override test subset size")
    parser.add_argument("--output-root", type=str, default="descargas_smoke", help="Root directory for run outputs")
    parser.add_argument("--display-root", type=str, default="display/results_smoke", help="Root directory for tables and plots")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without executing them")
    parser.add_argument("--download", dest="download", action="store_true", help="Allow dataset download if missing")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Disable dataset download")
    parser.set_defaults(download=True)
    return parser.parse_args()


def build_experiment(base_config: dict, spec: SmokeSpec, method: str, args: argparse.Namespace) -> dict:
    exp = copy.deepcopy(base_config.get("experiment", base_config))
    exp.update(
        {
            "dataset": spec.dataset,
            "model": spec.model,
            "method": method,
            "scenario": spec.scenario,
            "seed": args.seed,
            "num_clients": spec.num_clients,
            "participants_per_round": spec.participants_per_round,
            "rounds": args.rounds if args.rounds is not None else spec.rounds,
            "local_epochs": spec.local_epochs,
            "batch_size": spec.batch_size,
            "test_batch_size": spec.test_batch_size,
            "probe_batch_size": spec.probe_batch_size,
            "lr": spec.lr,
            "momentum": spec.momentum,
            "weight_decay": spec.weight_decay,
            "mu": spec.mu,
            "gamma": spec.gamma,
            "eta": spec.eta,
            "rho": spec.rho,
            "epsilon": spec.epsilon,
            "beta_mode": spec.beta_mode,
            "beta_window": spec.beta_window,
            "beta_window_size": spec.beta_window_size,
            "oui_layer": spec.oui_layer,
            "partition": spec.partition,
            "alpha": spec.alpha,
            "rare_client_fraction": spec.rare_client_fraction,
            "rare_client_size": spec.rare_client_size,
            "noisy_client_fraction": spec.noisy_client_fraction,
            "noise_rate": spec.noise_rate,
            "train_subset_size": args.train_subset_size if args.train_subset_size is not None else spec.train_subset_size,
            "test_subset_size": args.test_subset_size if args.test_subset_size is not None else spec.test_subset_size,
            "output_root": args.output_root,
            "data_root": args.data_root,
            "device": args.device,
            "download": args.download,
        }
    )
    return {"experiment": exp}


def main() -> None:
    args = parse_args()
    base_config = load_config(ROOT / "config" / "base.yaml")
    runs: list[tuple[str, str]] = []
    for spec in SMOKE_SPECS:
        for method in METHODS:
            runs.append((spec.name, method))

    print("Planned smoke runs:")
    for spec in SMOKE_SPECS:
        rounds = args.rounds if args.rounds is not None else spec.rounds
        train_subset = args.train_subset_size if args.train_subset_size is not None else spec.train_subset_size
        test_subset = args.test_subset_size if args.test_subset_size is not None else spec.test_subset_size
        print(
            f"- {spec.name}: dataset={spec.dataset}, model={spec.model}, scenario={spec.scenario}, "
            f"rounds={rounds}, train_subset={train_subset}, test_subset={test_subset}, methods={', '.join(METHODS)}"
        )

    if args.dry_run:
        return

    completed: list[dict] = []
    for spec in SMOKE_SPECS:
        for method in METHODS:
            config = build_experiment(base_config, spec, method, args)
            summary = run_single_experiment(config)
            completed.append(summary)
            print(
                f"Finished {spec.name}/{method}: final={summary['final_accuracy']:.4f}, "
                f"best={summary['best_accuracy']:.4f}, run_dir={summary['run_dir']}"
            )

    write_outputs(args.output_root, args.display_root)
    package_runs(args.output_root)
    package_scenario_summaries(args.output_root)
    print(f"Wrote summary tables to {args.display_root}")
    print(f"Packaged run artifacts under {args.output_root}")
    print(f"Completed {len(completed)} smoke runs.")


if __name__ == "__main__":
    main()
