"""Load and validate project configuration.

All paths, years, thresholds and seeds come from config/*.yaml via this module.
Nothing else in the codebase may hard-code them.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/climate_impact/config.py -> repo root
CONFIG_DIR = PROJECT_ROOT / "config"


def _read_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if the configuration is internally inconsistent.

    Uses explicit exceptions, not ``assert`` (asserts are stripped under ``python -O``).
    """
    a, s, st = cfg["analysis"], cfg["splits"], cfg["statistics"]
    b0, b1 = a["baseline_period"]
    start, end = a["study_start_year"], a["study_end_year"]

    if b1 - b0 + 1 != 30:
        raise ValueError(f"Baseline {b0}-{b1} is not a 30-year climate normal.")
    if not (start <= b0 and b1 <= end):
        raise ValueError(f"Baseline {b0}-{b1} must lie inside study window {start}-{end}.")
    if not (start < s["train_end_year"] < s["validation_end_year"] < end):
        raise ValueError("Splits must satisfy start < train_end < validation_end < study_end.")
    if not (0 < st["alpha"] < 1 and 0 < st["ci_level"] < 1):
        raise ValueError("alpha and ci_level must be in (0, 1).")
    if abs((1 - st["alpha"]) - st["ci_level"]) > 1e-9:
        raise ValueError("ci_level must equal 1 - alpha so tests and CIs agree.")
    if not 1 <= a["min_valid_months"] <= 12:
        raise ValueError("min_valid_months must be between 1 and 12.")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the validated project configuration (cached)."""
    cfg = _read_yaml("config.yaml")
    validate_config(cfg)
    return cfg


@lru_cache(maxsize=1)
def load_datasets() -> dict[str, Any]:
    """Return the dataset registry from config/datasets.yaml (cached)."""
    return _read_yaml("datasets.yaml")["datasets"]


def get_path(layer: str) -> Path:
    """Absolute path for a configured location, e.g. get_path('raw')."""
    return PROJECT_ROOT / load_config()["paths"][layer]


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Seeded NumPy Generator. Pass it explicitly to every stochastic function."""
    return np.random.default_rng(load_config()["project"]["random_seed"] if seed is None else seed)


def set_global_seeds(seed: int | None = None) -> int:
    """Seed legacy global RNGs (for libraries that ignore explicit generators)."""
    seed = load_config()["project"]["random_seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy API, intentionally seeded
    return seed