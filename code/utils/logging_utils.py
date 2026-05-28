from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from utils.io import ensure_dir, write_json, write_yaml


def setup_logging(log_path: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("fedoui")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def save_run_metadata(run_dir: str | Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    run_dir = ensure_dir(run_dir)
    write_yaml(run_dir / "config.yaml", config)
    meta = {"git_commit": git_commit_hash()}
    if extra:
        meta.update(extra)
    write_json(run_dir / "metadata.json", meta)

