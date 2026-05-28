from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis.summarize_runs import write_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Make paper tables")
    parser.add_argument("--descargas-root", type=str, default=str(ROOT / "descargas"))
    parser.add_argument("--display-root", type=str, default=str(ROOT / "display" / "results"))
    args = parser.parse_args()
    paths = write_outputs(args.descargas_root, args.display_root)
    tables_dir = Path(args.display_root) / "tables"
    main_df = pd.read_csv(paths["main_results"]) if Path(paths["main_results"]).exists() else pd.DataFrame()
    ablation_df = pd.read_csv(paths["ablations"]) if Path(paths["ablations"]).exists() else pd.DataFrame()
    oui_df = pd.read_csv(paths["oui_stats"]) if Path(paths["oui_stats"]).exists() else pd.DataFrame()
    if not main_df.empty:
        main_df.to_csv(tables_dir / "main_results.csv", index=False)
        try:
            main_df.to_latex(tables_dir / "main_results.tex", index=False, float_format="%.4f")
        except Exception:
            pass
    if not ablation_df.empty:
        ablation_df.to_csv(tables_dir / "ablations.csv", index=False)
    if not oui_df.empty:
        oui_df.to_csv(tables_dir / "oui_stats.csv", index=False)


if __name__ == "__main__":
    main()

