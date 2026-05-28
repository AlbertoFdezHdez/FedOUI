from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis.summarize_runs import write_outputs  # noqa: E402


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_accuracy_vs_round(round_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if round_df.empty:
        ax.text(0.5, 0.5, "No round data available", ha="center", va="center")
        _save(fig, out_path)
        return
    group_cols = ["scenario", "method", "round"]
    agg = (
        round_df.groupby(group_cols, dropna=False)
        .agg(acc_mean=("test_accuracy", "mean"), acc_std=("test_accuracy", "std"))
        .reset_index()
    )
    for (scenario, method), subset in agg.groupby(["scenario", "method"], dropna=False):
        subset = subset.sort_values("round")
        ax.plot(subset["round"], subset["acc_mean"], label=f"{scenario} / {method}")
        if subset["acc_std"].notna().any():
            ax.fill_between(
                subset["round"],
                subset["acc_mean"] - subset["acc_std"].fillna(0.0),
                subset["acc_mean"] + subset["acc_std"].fillna(0.0),
                alpha=0.15,
            )
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs round")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_oui_distribution(round_df: pd.DataFrame, participant_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if participant_df.empty:
        ax.text(0.5, 0.5, "No participant data available", ha="center", va="center")
        _save(fig, out_path)
        return
    if not round_df.empty and {"beta_alpha", "beta_beta"}.issubset(round_df.columns):
        fitted = round_df.dropna(subset=["beta_alpha", "beta_beta"]).sort_values(["scenario", "method", "round"])
        if not fitted.empty:
            row = fitted.iloc[-1]
        else:
            row = participant_df.dropna(subset=["oui"]).sort_values(["scenario", "method", "round"]).iloc[-1]
    else:
        row = participant_df.dropna(subset=["oui"]).sort_values(["scenario", "method", "round"]).iloc[-1]
    subset = participant_df[
        (participant_df["scenario"] == row["scenario"])
        & (participant_df["method"] == row["method"])
        & (participant_df["round"] == row["round"])
        & (participant_df["seed"] == row["seed"])
    ]
    values = subset["oui"].to_numpy(dtype=float)
    ax.hist(values, bins=min(10, max(3, len(values))), density=True, alpha=0.6, label="OUI")
    round_row = round_df[
        (round_df["scenario"] == row["scenario"])
        & (round_df["method"] == row["method"])
        & (round_df["round"] == row["round"])
        & (round_df["seed"] == row["seed"])
    ]
    beta_alpha = round_row["beta_alpha"].dropna().iloc[0] if not round_row.empty and round_row["beta_alpha"].notna().any() else None
    beta_beta = round_row["beta_beta"].dropna().iloc[0] if not round_row.empty and round_row["beta_beta"].notna().any() else None
    if pd.notna(beta_alpha) and pd.notna(beta_beta):
        xs = np.linspace(0.001, 0.999, 200)
        ax2 = ax.twinx()
        ax2.plot(xs, stats.beta.pdf(xs, float(beta_alpha), float(beta_beta)), color="tab:red", label="Beta fit")
        ax2.set_ylabel("Density")
    ax.set_xlabel("OUI")
    ax.set_ylabel("Density")
    ax.set_title(f"OUI distribution, scenario={row['scenario']}, round={int(row['round'])}")
    _save(fig, out_path)


def plot_oui_vs_alignment(participant_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if participant_df.empty or "alignment_a" not in participant_df.columns:
        ax.text(0.5, 0.5, "No alignment data available", ha="center", va="center")
        _save(fig, out_path)
        return
    df = participant_df.dropna(subset=["oui", "alignment_a"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        _save(fig, out_path)
        return
    ax.scatter(df["oui"], df["alignment_a"], alpha=0.5, s=18)
    ax.set_xlabel("OUI")
    ax.set_ylabel("Alignment")
    ax.set_title("OUI vs alignment")
    _save(fig, out_path)


def plot_weight_vs_oui(participant_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    df = participant_df.dropna(subset=["oui", "weight"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No weight data available", ha="center", va="center")
        _save(fig, out_path)
        return
    ax.scatter(df["oui"], df["weight"], alpha=0.5, s=18)
    ax.set_xlabel("OUI")
    ax.set_ylabel("Weight")
    ax.set_title("Weight vs OUI")
    _save(fig, out_path)


def plot_beta_over_time(round_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    df = round_df.dropna(subset=["beta_alpha", "beta_beta"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No Beta fit data available", ha="center", va="center")
        _save(fig, out_path)
        return
    for (scenario, method), subset in df.groupby(["scenario", "method"], dropna=False):
        subset = subset.sort_values("round")
        ax.plot(subset["round"], subset["beta_alpha"], label=f"{scenario}/{method} alpha")
        ax.plot(subset["round"], subset["beta_beta"], linestyle="--", label=f"{scenario}/{method} beta")
    ax.set_xlabel("Round")
    ax.set_ylabel("Beta parameter")
    ax.set_title("Beta parameters over time")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make paper plots")
    parser.add_argument("--descargas-root", type=str, default=str(ROOT / "descargas"))
    parser.add_argument("--display-root", type=str, default=str(ROOT / "display" / "results"))
    args = parser.parse_args()
    paths = write_outputs(args.descargas_root, args.display_root)
    summary_df = pd.read_csv(paths["summary_runs"]) if Path(paths["summary_runs"]).exists() else pd.DataFrame()
    round_df = pd.read_csv(paths["round_metrics"]) if Path(paths["round_metrics"]).exists() else pd.DataFrame()
    participant_df = pd.read_csv(paths["participant_metrics"]) if Path(paths["participant_metrics"]).exists() else pd.DataFrame()
    plots_dir = Path(args.display_root) / "plots"
    plot_accuracy_vs_round(round_df, plots_dir / "accuracy_vs_round.png")
    plot_oui_distribution(round_df, participant_df, plots_dir / "oui_distribution_round_X.png")
    plot_oui_vs_alignment(participant_df, plots_dir / "oui_vs_alignment.png")
    plot_weight_vs_oui(participant_df, plots_dir / "client_weight_vs_oui.png")
    plot_beta_over_time(round_df, plots_dir / "beta_fit_over_time.png")


if __name__ == "__main__":
    main()
