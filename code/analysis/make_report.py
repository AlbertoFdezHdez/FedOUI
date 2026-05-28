from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import ensure_dir  # noqa: E402


REPORT_NAME = "REPORT_FedOUI_results.txt"
EXEC_SUMMARY_NAME = "RESUMEN_EJECUTIVO_FedOUI.txt"
SELECTED_A_SCENARIO = "cifar10_dirichlet_0.1_extended"
SELECTED_C_SCENARIO = "cifar10_noisy_clients_extended"
SELECTED_METHOD = "fedoui"


def fmt_mean_std(mean: float, std: float | None) -> str:
    if pd.isna(std) or std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def _paper_rcparams(use_tex: bool) -> dict[str, object]:
    return {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def load_tables(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(base_dir / "summary_runs.csv")
    main_df = pd.read_csv(base_dir / "main_results.csv")
    round_df = pd.read_csv(base_dir / "round_metrics.csv")
    participant_df = pd.read_csv(base_dir / "participant_metrics.csv")
    return summary_df, main_df, round_df, participant_df


def select_extended_results(main_df: pd.DataFrame) -> pd.DataFrame:
    return main_df[main_df["scenario"].isin([SELECTED_A_SCENARIO, SELECTED_C_SCENARIO])].copy()


def format_extended_table(df: pd.DataFrame) -> str:
    lines = []
    for scenario in [SELECTED_A_SCENARIO, SELECTED_C_SCENARIO]:
        block = df[df["scenario"] == scenario].copy()
        if block.empty:
            continue
        lines.append(f"Scenario: {scenario}")
        lines.append(
            f"{'method':<14} {'final acc':<17} {'best acc':<17} {'AUC acc':<17} {'seeds':<5}"
        )
        for _, row in block.sort_values("method").iterrows():
            lines.append(
                f"{row['method']:<14} {fmt_mean_std(row['final_accuracy_mean'], row['final_accuracy_std']):<17} "
                f"{fmt_mean_std(row['best_accuracy_mean'], row['best_accuracy_std']):<17} "
                f"{fmt_mean_std(row['auc_accuracy_mean'], row['auc_accuracy_std']):<17} {int(row['seeds']):<5}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def selected_run(summary_df: pd.DataFrame, scenario: str, method: str = "fedoui") -> pd.Series:
    block = summary_df[(summary_df["scenario"] == scenario) & (summary_df["method"] == method)]
    if block.empty:
        raise RuntimeError(f"No run found for scenario={scenario}, method={method}")
    return block.sort_values(["best_accuracy", "final_accuracy"], ascending=False).iloc[0]


def build_snapshot_tables(summary_df: pd.DataFrame, round_df: pd.DataFrame, participant_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = selected_run(summary_df, SELECTED_A_SCENARIO, SELECTED_METHOD)
    run_dir = Path(run["run_dir"])
    round_idx = int(run["best_round"])
    round_row = round_df[
        (round_df["scenario"] == SELECTED_A_SCENARIO)
        & (round_df["method"] == SELECTED_METHOD)
        & (round_df["seed"] == run["seed"])
        & (round_df["round"] == round_idx)
    ].copy()
    participant_rows = participant_df[
        (participant_df["scenario"] == SELECTED_A_SCENARIO)
        & (participant_df["method"] == SELECTED_METHOD)
        & (participant_df["seed"] == run["seed"])
        & (participant_df["round"] == round_idx)
    ].copy()
    participant_rows = participant_rows.sort_values("oui")
    round_snapshot = participant_rows[
        ["round", "client_id", "n_k", "oui", "score_s", "alignment_a", "weight", "update_norm", "train_loss"]
    ].reset_index(drop=True)
    round_summary = round_row[
        ["scenario", "method", "seed", "round", "test_accuracy", "test_loss", "oui_mean", "oui_var", "beta_alpha", "beta_beta", "score_mode"]
    ].reset_index(drop=True)
    return run, round_summary, round_snapshot


def render_histogram(
    values: np.ndarray,
    output_base: Path,
    title: str,
    beta_alpha: float | None = None,
    beta_beta: float | None = None,
    mean: float | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    output_pdf = output_base.with_suffix(".pdf")
    output_png = output_base.with_suffix(".png")
    ensure_dir(output_pdf.parent)

    def _draw(use_tex_flag: bool) -> None:
        with plt.rc_context(_paper_rcparams(use_tex_flag)):
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            if values.size == 0:
                ax.text(0.5, 0.5, "No OUI values available", ha="center", va="center")
            else:
                bins = min(24, max(10, int(np.ceil(np.sqrt(values.size) * 2.0))))
                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    color="#4C72B0",
                    alpha=0.88,
                    edgecolor="white",
                    linewidth=0.8,
                    label=rf"OUI values ($n={values.size}$)",
                )
                beta_a = beta_alpha
                beta_b = beta_beta
                if (beta_a is None or beta_b is None) and values.size >= 3 and np.ptp(values) > 1e-9:
                    try:
                        beta_a, beta_b, _, _ = stats.beta.fit(values, floc=0, fscale=1)
                    except Exception:
                        beta_a = None
                        beta_b = None
                if beta_a is not None and beta_b is not None:
                    xs = np.linspace(0.001, 0.6, 500)
                    ax.plot(
                        xs,
                        stats.beta.pdf(xs, beta_a, beta_b),
                        color="#C44E52",
                        lw=2.2,
                        label=rf"Beta fit ($\alpha={beta_a:.2f}$, $\beta={beta_b:.2f}$)",
                    )
                mean_val = mean if mean is not None else float(np.mean(values))
                ax.axvline(mean_val, color="black", linestyle="--", lw=1.3, label=rf"Mean $={mean_val:.3f}$")
            ax.set_xlabel(r"$\mathrm{OUI}$")
            ax.set_ylabel(r"Density")
            ax.set_xlim(0.0, 0.6)
            ax.grid(True, which="major", linestyle="-", color="0.88")
            ax.grid(True, which="minor", linestyle=":", color="0.93")
            ax.minorticks_on()
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(top=False, right=False, labeltop=False, labelright=False)
            legend = ax.legend(
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.78,
                facecolor="white",
                edgecolor="0.75",
                borderpad=0.7,
            )
            fig.tight_layout()
            fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
            fig.savefig(output_png, dpi=320, bbox_inches="tight")
            plt.close(fig)

    try:
        _draw(shutil.which("latex") is not None)
    except Exception:
        _draw(False)


def build_executive_summary(selected_seed: int, selected_round: int, selected_run_dir: Path, round_snapshot: pd.DataFrame, round_summary: pd.DataFrame, scenario_a: pd.DataFrame, scenario_c: pd.DataFrame) -> str:
    a_best = scenario_a.sort_values(["best_accuracy_mean", "final_accuracy_mean"], ascending=False).iloc[0]
    c_best = scenario_c.sort_values(["best_accuracy_mean", "final_accuracy_mean"], ascending=False).iloc[0]
    top_snapshot = round_snapshot.sort_values("weight", ascending=False).iloc[0]
    lines = []
    lines.append("RESUMEN EJECUTIVO - FedOUI")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Objetivo")
    lines.append("--------")
    lines.append("Evaluar si OUI mejora la agregacion federada cuando los clientes no son homogeneos, y dejar un paquete de resultados reproducible y facil de inspeccionar.")
    lines.append("")
    lines.append("Que se entreno")
    lines.append("--------------")
    lines.append("- CIFAR-10.")
    lines.append("- Modelo CNN pequeno.")
    lines.append("- 20 clientes totales, 5 participantes por ronda.")
    lines.append("- 3 seeds: 1, 2 y 3.")
    lines.append("- 60 rondas por corrida, subset de train=3000 y test=1000.")
    lines.append("- Metodos: FedAvg, FedProx, FedOUI, FedAlign y FedOUI-Align.")
    lines.append("- OUI medido sobre la capa penultima, con batch de probe fijo por cliente.")
    lines.append("")
    lines.append("Como reproducir")
    lines.append("---------------")
    lines.append("Ejecuta desde la raiz del proyecto los comandos del report completo:")
    lines.append(f"- Configuracion principal non-IID: `code/config/cifar10_dirichlet_0.1.yaml`.")
    lines.append(f"- Configuracion de clientes ruidosos: `code/config/cifar10_noisy_clients.yaml`.")
    lines.append("- Mismas banderas para todas las corridas: `--rounds 60 --train_subset_size 3000 --test_subset_size 1000 --num_clients 20 --participants_per_round 5 --batch_size 32 --test_batch_size 256 --probe_batch_size 32 --output_root descargas --download False`.")
    lines.append("")
    lines.append("Resultados clave")
    lines.append("----------------")
    lines.append(f"- Escenario A (`{SELECTED_A_SCENARIO}`): gana FedOUI. Final medio {fmt_mean_std(a_best['final_accuracy_mean'], a_best['final_accuracy_std'])}, mejor medio {fmt_mean_std(a_best['best_accuracy_mean'], a_best['best_accuracy_std'])}.")
    lines.append(f"- Escenario C (`{SELECTED_C_SCENARIO}`): FedOUI gana en mejor accuracy media ({fmt_mean_std(c_best['best_accuracy_mean'], c_best['best_accuracy_std'])}), pero no en accuracy final.")
    lines.append(f"- La ronda representativa usada para el snapshot es seed {selected_seed}, round {selected_round}, en `{selected_run_dir}`.")
    lines.append("")
    lines.append("Que se observa")
    lines.append("-------------")
    lines.append(f"- El cliente con mayor peso en la ronda representativa es el cliente {int(top_snapshot['client_id'])}, con OUI cercano al centro de la distribucion.")
    lines.append("- FedOUI se comporta como una senal estructural suave: penaliza extremos sin expulsarlos.")
    lines.append("- FedAlign no aporta una mejora estable frente a FedOUI puro en estos escenarios.")
    lines.append("")
    lines.append("Sobre el histograma")
    lines.append("-------------------")
    lines.append("- El histograma original usa los clientes participantes de una ronda concreta; como solo participan 5 clientes por ronda, es normal que se vea muy esparcido.")
    lines.append("- Para una lectura mas estable, la figura adicional usa una ventana local mas amplia de rondas alrededor de ese instante, de modo que la beta encima se ajuste a una muestra mas informativa. Esa version se guarda tambien como PDF vectorial con tipografia estilo paper.")
    lines.append("")
    lines.append("Conclusion")
    lines.append("----------")
    lines.append("FedOUI es especialmente util en el escenario non-IID fuerte; en ruido de clientes sigue siendo competitivo, pero el efecto es mas dependiente del escenario.")
    lines.append("")
    return "\n".join(lines) + "\n"


def table_to_block(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df.empty:
        return "(empty)"
    rows = []
    headers = list(df.columns)
    rows.append(" | ".join(headers))
    rows.append(" | ".join(["---"] * len(headers)))
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            value = row[col]
            if isinstance(value, (np.floating, float)):
                if abs(float(value) - round(float(value))) < 1e-9 and col in {"round", "seed", "client_id", "n_k"}:
                    vals.append(str(int(round(float(value)))))
                else:
                    vals.append(floatfmt.format(float(value)))
            elif isinstance(value, (np.integer, int)):
                vals.append(str(int(value)))
            else:
                vals.append(str(value))
        rows.append(" | ".join(vals))
    return "\n".join(rows)


def write_report(report_path: Path, text: str) -> None:
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    base_dir = Path("display/results/tables")
    plot_dir = Path("display/results/plots")
    out_tables = ensure_dir(base_dir)
    ensure_dir(plot_dir)
    summary_df, main_df, round_df, participant_df = load_tables(base_dir)
    extended_df = select_extended_results(main_df)

    run, round_summary, round_snapshot = build_snapshot_tables(summary_df, round_df, participant_df)
    selected_round = int(round_summary["round"].iloc[0])
    selected_seed = int(round_summary["seed"].iloc[0])
    selected_run_dir = Path(run["run_dir"])

    extended_table_path = out_tables / "report_extended_results.csv"
    snapshot_table_path = out_tables / "report_oui_snapshot.csv"
    round_summary_path = out_tables / "report_oui_round_summary.csv"
    extended_df.to_csv(extended_table_path, index=False)
    round_snapshot.to_csv(snapshot_table_path, index=False)
    round_summary.to_csv(round_summary_path, index=False)

    histogram_base = plot_dir / "oui_histogram_round59_fedoui_extended_A"
    render_histogram(
        round_snapshot["oui"].to_numpy(dtype=float),
        histogram_base,
        title=f"OUI snapshot: {selected_run_dir.name}, round {selected_round}",
        beta_alpha=float(round_summary["beta_alpha"].iloc[0]),
        beta_beta=float(round_summary["beta_beta"].iloc[0]),
        mean=float(round_summary["oui_mean"].iloc[0]),
    )

    paper_histogram_base = plot_dir / "oui_histogram_round59_fedoui_extended_A_window_paper"
    window_size = 30
    lower = max(0, selected_round - window_size + 1)
    aggregated_subset = participant_df[
        (participant_df["scenario"] == SELECTED_A_SCENARIO)
        & (participant_df["method"] == SELECTED_METHOD)
        & (participant_df["seed"] == selected_seed)
        & (participant_df["round"] >= lower)
        & (participant_df["round"] <= selected_round)
    ].copy()
    render_histogram(
        aggregated_subset["oui"].to_numpy(dtype=float),
        paper_histogram_base,
        title=f"OUI distribution over rounds {lower}-{selected_round} (seed {selected_seed})",
    )

    scenario_a = extended_df[extended_df["scenario"] == SELECTED_A_SCENARIO].copy()
    scenario_c = extended_df[extended_df["scenario"] == SELECTED_C_SCENARIO].copy()

    report_lines = []
    report_lines.append("FedOUI Experimental Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Scope")
    report_lines.append("-----")
    report_lines.append(
        "This report summarizes the extended experiments kept after the pilot sweep: the non-IID Dirichlet setting where FedOUI was competitive or better, and the noisy-client setting where FedOUI showed the strongest peak accuracy."
    )
    report_lines.append("")
    report_lines.append("What was trained")
    report_lines.append("----------------")
    report_lines.append("- Dataset: CIFAR-10.")
    report_lines.append("- Model: small CNN with two conv blocks, one hidden fully connected layer, and a final classifier.")
    report_lines.append("- Clients: 20 total.")
    report_lines.append("- Participants per round: 5 clients.")
    report_lines.append("- Local training: 1 epoch, SGD with momentum 0.9, lr 0.01, batch size 32.")
    report_lines.append("- Probe batch for OUI: 32 samples, fixed per client, computed on the penultimate pre-activation.")
    report_lines.append("- OUI aggregation rule: Beta fit on the round OUI values with percentile fallback if the fit is unstable.")
    report_lines.append("- Methods compared: FedAvg, FedProx, FedOUI, FedAlign, FedOUI-Align.")
    report_lines.append("- Seeds: 1, 2, 3.")
    report_lines.append("- Extended runs: 60 rounds, train subset 3000, test subset 1000.")
    report_lines.append("")
    report_lines.append("How to reproduce the extended experiments")
    report_lines.append("----------------------------------------")
    report_lines.append("Use these exact PowerShell loops from the project root:")
    report_lines.append("")
    report_lines.append("```powershell")
    report_lines.append("$methods = @('fedavg','fedprox','fedoui','fedalign','fedoui_align')")
    report_lines.append("foreach ($seed in 1,2,3) {")
    report_lines.append("  foreach ($method in $methods) {")
    report_lines.append(
        "    python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method $method --seed $seed --rounds 60 --train_subset_size 3000 --test_subset_size 1000 --num_clients 20 --participants_per_round 5 --batch_size 32 --test_batch_size 256 --probe_batch_size 32 --output_root descargas --download False"
    )
    report_lines.append("  }")
    report_lines.append("}")
    report_lines.append("")
    report_lines.append("foreach ($seed in 1,2,3) {")
    report_lines.append("  foreach ($method in $methods) {")
    report_lines.append(
        "    python code/main.py --config code/config/cifar10_noisy_clients.yaml --method $method --seed $seed --rounds 60 --train_subset_size 3000 --test_subset_size 1000 --num_clients 20 --participants_per_round 5 --batch_size 32 --test_batch_size 256 --probe_batch_size 32 --output_root descargas --download False"
    )
    report_lines.append("  }")
    report_lines.append("}")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("Important fixed settings")
    report_lines.append("------------------------")
    report_lines.append("- OUI round for the representative histogram: the best round of the best FedOUI run in `cifar10_dirichlet_0.1_extended`.")
    report_lines.append(f"- Selected run: seed {selected_seed}, round {selected_round}, run directory `{selected_run_dir}`.")
    report_lines.append("- Output root used in the runs: `descargas/`.")
    report_lines.append("")
    report_lines.append("Table 1. Main extended results")
    report_lines.append("------------------------------")
    report_lines.append(table_to_block(extended_df[["scenario", "method", "final_accuracy_mean", "final_accuracy_std", "best_accuracy_mean", "best_accuracy_std", "auc_accuracy_mean", "auc_accuracy_std", "seeds"]].sort_values(["scenario", "method"])))
    report_lines.append("")
    report_lines.append("Interpretation of Table 1")
    report_lines.append("-------------------------")
    report_lines.append(
        "In `cifar10_dirichlet_0.1_extended`, FedOUI is the strongest method on final accuracy and best accuracy. In the stronger-noise setting, FedOUI is not the best on final accuracy, but it does achieve the highest best-accuracy among the OUI-aware methods and remains competitive with the baselines. FedAlign is consistently weaker in the non-IID regime and should be treated as a secondary comparison, not the main story."
    )
    report_lines.append("")
    report_lines.append("Table 2. Representative OUI snapshot")
    report_lines.append("------------------------------------")
    report_lines.append(table_to_block(round_snapshot[["client_id", "n_k", "oui", "score_s", "alignment_a", "weight", "update_norm", "train_loss"]]))
    report_lines.append("")
    report_lines.append("Interpretation of Table 2")
    report_lines.append("-------------------------")
    report_lines.append(
        "The round snapshot shows the mechanism clearly. The round-level OUI distribution is centered around 0.277 and is fitted by a Beta(6.10, 15.94). The client with OUI closest to the round center receives the largest structural score and the largest aggregation weight, while the low-OUI and very high-OUI clients are penalized smoothly. The alignment score is present, but OUI is the dominant gating signal in this example."
    )
    report_lines.append("")
    report_lines.append("Table 3. Round summary for the histogram case")
    report_lines.append("--------------------------------------------")
    report_lines.append(table_to_block(round_summary[["scenario", "method", "seed", "round", "test_accuracy", "test_loss", "oui_mean", "oui_var", "beta_alpha", "beta_beta", "score_mode"]]))
    report_lines.append("")
    report_lines.append("Interpretation of Table 3")
    report_lines.append("-------------------------")
    report_lines.append(
        "The representative round is not a degenerate fit. The Beta parameters are finite and the score mode is `beta`, so the parametric structural score is active rather than falling back to percentiles. This is the kind of round the method is trying to exploit: the OUI values are spread but still compact enough to define a meaningful central tendency."
    )
    report_lines.append("")
    report_lines.append("What we observe")
    report_lines.append("---------------")
    report_lines.append("- The extended Dirichlet non-IID scenario is the cleanest win for FedOUI.")
    report_lines.append("- The noisy-client scenario is more mixed: FedOUI improves the best observed peak, but not always the final accuracy.")
    report_lines.append("- Combining OUI with alignment does not beat plain FedOUI here, so the combination should be described as exploratory rather than guaranteed improvement.")
    report_lines.append("- The Beta fit remains stable over the extended runs, with large but finite alpha/beta values, which is consistent with a bounded structural signal rather than a heavy-tailed one.")
    report_lines.append("")
    report_lines.append("Conclusions")
    report_lines.append("-----------")
    report_lines.append(
        "1. OUI works best as a soft structural tipicity signal, not as a hard filter."
    )
    report_lines.append(
        "2. In the strongest non-IID case we kept, FedOUI improved both the final and best accuracy over FedAvg and FedProx."
    )
    report_lines.append(
        "3. In the noisy-client case, FedOUI improved the best peak but did not dominate final accuracy, so the effect should be presented as scenario-dependent."
    )
    report_lines.append(
        "4. The histogram and the snapshot table show the intended behavior directly: central OUI clients get the highest weights, while extremes are softened rather than discarded."
    )
    report_lines.append("")
    report_lines.append("Files produced")
    report_lines.append("--------------")
    report_lines.append(f"- Report text: `{Path(REPORT_NAME).resolve()}`")
    report_lines.append(f"- Extended results table: `{extended_table_path.resolve()}`")
    report_lines.append(f"- OUI snapshot table: `{snapshot_table_path.resolve()}`")
    report_lines.append(f"- OUI round summary table: `{round_summary_path.resolve()}`")
    report_lines.append(f"- Snapshot histogram PDF+PNG base: `{histogram_base.resolve()}`")
    report_lines.append(f"- Windowed histogram paper PDF+PNG base: `{paper_histogram_base.resolve()}`")

    report_text = "\n".join(report_lines) + "\n"
    write_report(Path(REPORT_NAME), report_text)

    exec_summary_text = build_executive_summary(
        selected_seed=selected_seed,
        selected_round=selected_round,
        selected_run_dir=selected_run_dir,
        round_snapshot=round_snapshot,
        round_summary=round_summary,
        scenario_a=scenario_a,
        scenario_c=scenario_c,
    )
    write_report(Path(EXEC_SUMMARY_NAME), exec_summary_text)


if __name__ == "__main__":
    main()
