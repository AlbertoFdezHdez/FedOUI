from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import ensure_dir, write_csv


def iter_run_dirs(descargas_root: str | Path):
    descargas_root = Path(descargas_root)
    if not descargas_root.exists():
        return
    for path in descargas_root.rglob("summary.json"):
        yield path.parent


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_run_rows(descargas_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    round_rows = []
    participant_rows = []
    for run_dir in iter_run_dirs(descargas_root):
        summary_path = run_dir / "summary.json"
        round_metrics_path = run_dir / "round_metrics.json"
        participant_metrics_path = run_dir / "participant_metrics.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        summary["run_dir"] = str(run_dir)
        round_metrics = load_json(round_metrics_path) if round_metrics_path.exists() else []
        participant_metrics = load_json(participant_metrics_path) if participant_metrics_path.exists() else []
        if round_metrics:
            accs = np.asarray([row.get("test_accuracy", 0.0) for row in round_metrics], dtype=float)
            summary["auc_accuracy"] = float(np.trapz(accs, dx=1.0) / max(len(accs) - 1, 1))
            summary["mean_round_accuracy"] = float(accs.mean())
        else:
            summary["auc_accuracy"] = None
            summary["mean_round_accuracy"] = None
        summary_rows.append(summary)
        for row in round_metrics:
            row = dict(row)
            row["dataset"] = summary["dataset"]
            row["scenario"] = summary["scenario"]
            row["method"] = summary["method"]
            row["seed"] = summary["seed"]
            row["run_dir"] = str(run_dir)
            round_rows.append(row)
        for row in participant_metrics:
            row = dict(row)
            row["dataset"] = summary["dataset"]
            row["scenario"] = summary["scenario"]
            row["method"] = summary["method"]
            row["seed"] = summary["seed"]
            row["run_dir"] = str(run_dir)
            participant_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(round_rows), pd.DataFrame(participant_rows)


def aggregate_tables(summary_df: pd.DataFrame, round_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    group_cols = ["dataset", "scenario", "method"]
    agg_rows = (
        summary_df.groupby(group_cols, dropna=False)
        .agg(
            final_accuracy_mean=("final_accuracy", "mean"),
            final_accuracy_std=("final_accuracy", "std"),
            best_accuracy_mean=("best_accuracy", "mean"),
            best_accuracy_std=("best_accuracy", "std"),
            auc_accuracy_mean=("auc_accuracy", "mean"),
            auc_accuracy_std=("auc_accuracy", "std"),
            seeds=("seed", "count"),
        )
        .reset_index()
    )
    ablation_rows = (
        summary_df.groupby(group_cols + ["gamma", "eta"], dropna=False)
        .agg(
            final_accuracy_mean=("final_accuracy", "mean"),
            final_accuracy_std=("final_accuracy", "std"),
            best_accuracy_mean=("best_accuracy", "mean"),
            best_accuracy_std=("best_accuracy", "std"),
            seeds=("seed", "count"),
        )
        .reset_index()
    )
    if round_df.empty:
        oui_rows = pd.DataFrame()
    else:
        oui_rows = (
            round_df.groupby(["dataset", "scenario", "method", "round"], dropna=False)
            .agg(
                mean_oui=("oui_mean", "mean"),
                var_oui=("oui_var", "mean"),
                beta_alpha=("beta_alpha", "first"),
                beta_beta=("beta_beta", "first"),
                score_mode=("score_mode", "first"),
            )
            .reset_index()
        )
    return agg_rows, ablation_rows, oui_rows


def write_outputs(descargas_root: str | Path, display_root: str | Path) -> dict[str, Path]:
    summary_df, round_df, participant_df = read_run_rows(descargas_root)
    main_df, ablation_df, oui_df = aggregate_tables(summary_df, round_df)
    display_root = Path(display_root)
    tables_dir = ensure_dir(display_root / "tables")
    ensure_dir(display_root / "plots")
    summary_df.to_csv(tables_dir / "summary_runs.csv", index=False)
    round_df.to_csv(tables_dir / "round_metrics.csv", index=False)
    participant_df.to_csv(tables_dir / "participant_metrics.csv", index=False)
    main_df.to_csv(tables_dir / "main_results.csv", index=False)
    ablation_df.to_csv(tables_dir / "ablations.csv", index=False)
    oui_df.to_csv(tables_dir / "oui_stats.csv", index=False)
    return {
        "summary_runs": tables_dir / "summary_runs.csv",
        "round_metrics": tables_dir / "round_metrics.csv",
        "participant_metrics": tables_dir / "participant_metrics.csv",
        "main_results": tables_dir / "main_results.csv",
        "ablations": tables_dir / "ablations.csv",
        "oui_stats": tables_dir / "oui_stats.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FL runs")
    parser.add_argument("--descargas-root", type=str, default="descargas")
    parser.add_argument("--display-root", type=str, default="display/results")
    args = parser.parse_args()
    write_outputs(args.descargas_root, args.display_root)


if __name__ == "__main__":
    main()
