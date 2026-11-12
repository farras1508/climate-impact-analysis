"""Experiment tracking: MLflow setup and reproducibility tags.

Every run is tagged with what is needed to reproduce it: the code version
(git commit), the data version (manifest hash) and the analysis settings
(config hashes).
"""

from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
from pathlib import Path

import mlflow

from climate_impact.config import CONFIG_DIR, PROJECT_ROOT, get_path, load_config

logger = logging.getLogger(__name__)


def tracking_uri() -> str:
    """Return the absolute SQLite URI of the project's MLflow store."""
    db: Path = PROJECT_ROOT / str(load_config()["tracking"]["mlflow_db"])
    return f"sqlite:///{db.as_posix()}"


def configure_mlflow(uri: str | None = None) -> str:
    """Point MLflow at the project store (or ``uri``) and return the URI used."""
    uri = uri or tracking_uri()
    mlflow.set_tracking_uri(uri)
    return uri


def get_or_create_experiment(name: str, artifact_root: Path | None = None) -> str:
    """Return the ID of experiment ``name``, creating it if it does not exist."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    root = artifact_root or PROJECT_ROOT / str(load_config()["tracking"]["artifact_dir"])
    return str(mlflow.create_experiment(name, artifact_location=(root / name).resolve().as_uri()))


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _sha256_or_missing(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def run_context_tags() -> dict[str, str]:
    """Build the reproducibility tags attached to every MLflow run."""
    status = _git("status", "--porcelain")
    dirty = status not in ("", "unknown")
    if dirty:
        logger.warning("Uncommitted changes: this run cannot be reproduced from a commit.")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(dirty).lower(),
        "config_sha256": _sha256_or_missing(CONFIG_DIR / "config.yaml"),
        "datasets_sha256": _sha256_or_missing(CONFIG_DIR / "datasets.yaml"),
        "manifest_sha256": _sha256_or_missing(get_path("manifest")),
        "random_seed": str(load_config()["project"]["random_seed"]),
        "python_version": platform.python_version(),
    }
