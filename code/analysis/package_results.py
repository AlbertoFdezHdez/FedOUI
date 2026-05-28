from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import ensure_dir


def zip_dir(source_dir: str | Path, zip_path: str | Path) -> Path:
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))
    return zip_path


def package_runs(descargas_root: str | Path) -> list[Path]:
    descargas_root = Path(descargas_root)
    packaged = []
    for summary_path in descargas_root.rglob("summary.json"):
        run_dir = summary_path.parent
        zip_path = run_dir.parent / f"{run_dir.name}.zip"
        packaged.append(zip_dir(run_dir, zip_path))
    return packaged


def package_scenario_summaries(descargas_root: str | Path) -> list[Path]:
    descargas_root = Path(descargas_root)
    packed = []
    for dataset_dir in [p for p in descargas_root.iterdir() if p.is_dir()]:
        for scenario_dir in [p for p in dataset_dir.iterdir() if p.is_dir()]:
            zip_path = scenario_dir / "aggregated_summary.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for summary_file in scenario_dir.rglob("summary.json"):
                    zf.write(summary_file, summary_file.relative_to(scenario_dir))
                for file_name in ["summary.yaml", "metadata.json", "config.yaml", "rounds.jsonl", "participant_metrics.json", "round_metrics.json"]:
                    for file_path in scenario_dir.rglob(file_name):
                        zf.write(file_path, file_path.relative_to(scenario_dir))
            packed.append(zip_path)
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description="Package FL results")
    parser.add_argument("--descargas-root", type=str, default="descargas")
    args = parser.parse_args()
    package_runs(args.descargas_root)
    package_scenario_summaries(args.descargas_root)


if __name__ == "__main__":
    main()
