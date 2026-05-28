from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.io import ensure_dir  # noqa: E402


SCENARIOS = ["cifar10_dirichlet_0.1_extended", "cifar10_noisy_clients_extended"]
METHODS = ["fedavg", "fedprox", "fedalign", "fedoui"]
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedalign": "FedAlign",
    "fedoui": "FedOUI",
}
SCENARIO_LABELS = {
    "cifar10_dirichlet_0.1_extended": r"Strong non-IID, $\alpha=0.1$",
    "cifar10_noisy_clients_extended": r"Noisy clients",
}
METHOD_COLORS = {
    "fedavg": "#4C72B0",
    "fedprox": "#55A868",
    "fedalign": "#8172B3",
    "fedoui": "#C44E52",
}
BLUE = "#4C72B0"
RED = "#C44E52"
OUI_XMAX = 0.5


def paper_rcparams(use_tex: bool) -> dict[str, object]:
    return {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,
        "pdf.compression": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    ensure_dir(output_base.parent)
    try:
        fig.savefig(output_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".png"), dpi=340, bbox_inches="tight")
    except PermissionError:
        unlocked_base = output_base.with_name(f"{output_base.name}_vector")
        fig.savefig(unlocked_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(unlocked_base.with_suffix(".png"), dpi=340, bbox_inches="tight")


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, which="major", linestyle="-", color="0.88")
        ax.grid(True, which="minor", linestyle=":", color="0.93")
    else:
        ax.grid(False, which="both")
    ax.minorticks_on()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)


def add_legend(ax: plt.Axes, **kwargs) -> None:
    defaults = {
        "frameon": True,
        "fancybox": True,
        "framealpha": 0.78,
        "facecolor": "white",
        "edgecolor": "0.75",
        "borderpad": 0.7,
    }
    defaults.update(kwargs)
    ax.legend(**defaults)


def load_tables(tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(tables_dir / "summary_runs.csv")
    round_df = pd.read_csv(tables_dir / "round_metrics.csv")
    participant_df = pd.read_csv(tables_dir / "participant_metrics.csv")
    return summary_df, round_df, participant_df


def selected_fedoui_window(
    summary_df: pd.DataFrame,
    participant_df: pd.DataFrame,
    window_size: int = 30,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    block = summary_df[
        (summary_df["scenario"] == SCENARIOS[0])
        & (summary_df["method"] == "fedoui")
        & (summary_df["seed"].isin([1, 2, 3]))
    ].copy()
    if block.empty:
        raise RuntimeError("No FedOUI run found for the strong non-IID scenario.")
    run = block.sort_values(["best_accuracy", "final_accuracy"], ascending=False).iloc[0]
    selected_round = int(run["best_round"])
    lower = max(0, selected_round - window_size + 1)
    subset = participant_df[
        (participant_df["scenario"] == SCENARIOS[0])
        & (participant_df["method"] == "fedoui")
        & (participant_df["seed"] == int(run["seed"]))
        & (participant_df["round"].between(lower, selected_round))
    ].copy()
    return subset, run, lower, selected_round


def draw_histogram(values: np.ndarray, output_base: Path) -> None:
    values = np.asarray(values, dtype=float)

    def _draw(use_tex: bool) -> None:
        with plt.rc_context(paper_rcparams(use_tex)):
            fig, ax = plt.subplots(figsize=(4.6, 4.25))
            bins = min(24, max(10, int(np.ceil(np.sqrt(values.size) * 2.0))))
            ax.hist(
                values,
                bins=bins,
                density=True,
                color=BLUE,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.8,
                label=rf"OUI values ($n={values.size}$)",
            )
            beta_a = beta_b = None
            if values.size >= 3 and np.ptp(values) > 1e-9:
                try:
                    beta_a, beta_b, _, _ = stats.beta.fit(values, floc=0, fscale=1)
                except Exception:
                    beta_a = beta_b = None
            if beta_a is not None and beta_b is not None:
                xs = np.linspace(0.001, OUI_XMAX, 500)
                ax.plot(
                    xs,
                    stats.beta.pdf(xs, beta_a, beta_b),
                    color=RED,
                    lw=2.1,
                    label=rf"Beta fit ($\alpha={beta_a:.2f}$, $\beta={beta_b:.2f}$)",
                )
            mean_val = float(np.mean(values))
            ax.axvline(mean_val, color="black", linestyle="--", lw=1.2, label=rf"Mean $={mean_val:.3f}$")
            ax.set_xlabel(r"$\mathrm{OUI}$")
            ax.set_ylabel(r"Density")
            ax.set_xlim(0.0, OUI_XMAX)
            style_axes(ax)
            add_legend(ax, loc="upper right")
            fig.tight_layout(pad=0.25)
            save_figure(fig, output_base)
            plt.close(fig)

    try:
        _draw(shutil.which("latex") is not None)
    except Exception:
        _draw(False)


def draw_weight_vs_oui(participant_df: pd.DataFrame, output_base: Path) -> None:
    # Match the original client_weight_vs_oui figure: every available client-round observation.
    df = participant_df.dropna(subset=["oui", "weight"]).copy()

    def _draw(use_tex: bool) -> None:
        with plt.rc_context(paper_rcparams(use_tex)):
            fig, ax = plt.subplots(figsize=(4.6, 4.25))
            ax.scatter(
                df["oui"],
                df["weight"],
                s=3.8,
                color=BLUE,
                alpha=0.52,
                edgecolor="none",
            )
            ax.set_xlabel(r"$\mathrm{OUI}$")
            ax.set_ylabel(r"Aggregation weight $w_k^t$")
            ax.set_xlim(0.0, OUI_XMAX)
            ax.set_ylim(-0.02, 1.05)
            style_axes(ax, grid=False)
            fig.tight_layout(pad=0.25)
            save_figure(fig, output_base)
            plt.close(fig)

    try:
        _draw(shutil.which("latex") is not None)
    except Exception:
        _draw(False)


def aggregate_curves(round_df: pd.DataFrame, scenario: str, method: str, metric: str) -> pd.DataFrame:
    block = round_df[
        (round_df["scenario"] == scenario)
        & (round_df["method"] == method)
        & (round_df["seed"].isin([1, 2, 3]))
    ].copy()
    grouped = (
        block.groupby("round", as_index=False)[metric]
        .agg(median="median", lo="min", hi="max")
        .sort_values("round")
    )
    return grouped


def robust_loss_ylim(round_df: pd.DataFrame) -> tuple[float, float] | None:
    median_values: list[np.ndarray] = []
    for scenario in SCENARIOS:
        for method in METHODS:
            curve = aggregate_curves(round_df, scenario, method, "test_loss")
            if not curve.empty:
                median_values.append(curve["median"].to_numpy(dtype=float))
    if not median_values:
        return None
    values = np.concatenate(median_values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = max(0.05, 0.08 * (hi - lo))
    return max(0.0, lo - pad), hi + pad


def draw_training_grid(round_df: pd.DataFrame, output_base: Path) -> None:
    def _draw(use_tex: bool) -> None:
        with plt.rc_context(paper_rcparams(use_tex)):
            fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex=True)
            metrics = [("test_accuracy", r"Test accuracy"), ("test_loss", r"Test loss")]
            loss_ylim = robust_loss_ylim(round_df)
            for row_idx, scenario in enumerate(SCENARIOS):
                for col_idx, (metric, ylabel) in enumerate(metrics):
                    ax = axes[row_idx, col_idx]
                    for method in METHODS:
                        curve = aggregate_curves(round_df, scenario, method, metric)
                        if curve.empty:
                            continue
                        x = curve["round"].to_numpy(dtype=float)
                        y = curve["median"].to_numpy(dtype=float)
                        lo = curve["lo"].to_numpy(dtype=float)
                        hi = curve["hi"].to_numpy(dtype=float)
                        color = METHOD_COLORS[method]
                        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
                        ax.plot(x, y, color=color, lw=1.8, label=METHOD_LABELS[method])
                    if row_idx == 0:
                        ax.set_title(ylabel)
                    ax.text(
                        0.035,
                        0.94,
                        SCENARIO_LABELS[scenario],
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=9,
                        bbox={
                            "boxstyle": "round,pad=0.28",
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.78,
                        },
                    )
                    if row_idx == 1:
                        ax.set_xlabel(r"Communication round")
                    if col_idx == 0:
                        ax.set_ylabel(r"Accuracy")
                    else:
                        ax.set_ylabel(r"Loss")
                        if loss_ylim is not None:
                            ax.set_ylim(*loss_ylim)
                    style_axes(ax)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=4,
                frameon=True,
                fancybox=True,
                framealpha=0.78,
                facecolor="white",
                edgecolor="0.75",
                bbox_to_anchor=(0.54, 1.02),
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=1.0, w_pad=1.0)
            save_figure(fig, output_base)
            plt.close(fig)

    try:
        _draw(shutil.which("latex") is not None)
    except Exception:
        _draw(False)


def main() -> None:
    release_tables_dir = ROOT / "results" / "tables"
    dev_tables_dir = ROOT / "display" / "results" / "tables"
    tables_dir = release_tables_dir if release_tables_dir.exists() else dev_tables_dir
    plots_dir = ROOT / "results" / "figures"
    summary_df, round_df, participant_df = load_tables(tables_dir)

    window_df, _, _, _ = selected_fedoui_window(summary_df, participant_df)
    draw_histogram(window_df["oui"].to_numpy(dtype=float), plots_dir / "paper_oui_histogram_square")
    draw_weight_vs_oui(participant_df, plots_dir / "paper_client_weight_vs_oui")
    draw_training_grid(round_df, plots_dir / "paper_training_dynamics_grid")


if __name__ == "__main__":
    main()
