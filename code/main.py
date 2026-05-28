from __future__ import annotations

import argparse

from run_experiment import load_config, run_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated OUI experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--method", type=str, default=None, help="Override method")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds")
    parser.add_argument("--scenario", type=str, default=None, help="Override scenario")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset")
    parser.add_argument("--model", type=str, default=None, help="Override model")
    parser.add_argument("--partition", type=str, default=None, help="Override data partition")
    parser.add_argument("--alpha", type=float, default=None, help="Override Dirichlet alpha")
    parser.add_argument("--num_clients", type=int, default=None, help="Override number of clients")
    parser.add_argument("--participants_per_round", type=int, default=None, help="Override participants per round")
    parser.add_argument("--local_epochs", type=int, default=None, help="Override local epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override local batch size")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Override test batch size")
    parser.add_argument("--probe_batch_size", type=int, default=None, help="Override probe batch size")
    parser.add_argument("--train_subset_size", type=int, default=None, help="Override train subset size")
    parser.add_argument("--test_subset_size", type=int, default=None, help="Override test subset size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--momentum", type=float, default=None, help="Override SGD momentum")
    parser.add_argument("--mu", type=float, default=None, help="Override FedProx mu")
    parser.add_argument("--gamma", type=float, default=None, help="Override FedOUI gamma")
    parser.add_argument("--eta", type=float, default=None, help="Override alignment eta")
    parser.add_argument("--rho", type=float, default=None, help="Override size exponent rho")
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon")
    parser.add_argument("--beta_mode", type=str, default=None, help="Override OUI score mode")
    parser.add_argument("--beta_window", type=str, default=None, help="Override OUI window mode")
    parser.add_argument("--beta_window_size", type=int, default=None, help="Override OUI window size")
    parser.add_argument("--oui_layer", type=str, default=None, help="Override OUI layer")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--download", type=str, default=None, help="Override dataset download flag")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    exp = config.get("experiment", config)
    if args.method is not None:
        exp["method"] = args.method
    if args.seed is not None:
        exp["seed"] = args.seed
    if args.rounds is not None:
        exp["rounds"] = args.rounds
    if args.scenario is not None:
        exp["scenario"] = args.scenario
    if args.output_root is not None:
        exp["output_root"] = args.output_root
    for key in [
        "dataset",
        "model",
        "partition",
        "alpha",
        "num_clients",
        "participants_per_round",
        "local_epochs",
        "batch_size",
        "test_batch_size",
        "probe_batch_size",
        "train_subset_size",
        "test_subset_size",
        "lr",
        "momentum",
        "mu",
        "gamma",
        "eta",
        "rho",
        "epsilon",
        "beta_mode",
        "beta_window",
        "beta_window_size",
        "oui_layer",
        "device",
    ]:
        value = getattr(args, key)
        if value is not None:
            exp[key] = value
    if args.download is not None:
        exp["download"] = parse_bool(args.download)
    config["experiment"] = exp
    method = str(exp.get("method", "fedavg")).lower()
    if method == "all":
        methods = ["fedavg", "fedprox", "fedoui", "fedalign", "fedoui_align"]
        for item in methods:
            run_cfg = load_config(args.config)
            run_exp = run_cfg.get("experiment", run_cfg)
            run_exp.update(exp)
            run_exp["method"] = item
            run_cfg["experiment"] = run_exp
            run_single_experiment(run_cfg)
    else:
        run_single_experiment(config)


if __name__ == "__main__":
    main()
