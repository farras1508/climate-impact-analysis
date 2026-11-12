"""Load and validate project configuration.

All paths, years, thresholds and seeds come from ``config/*.yaml`` through this
module. Nothing else in the codebase may hard-code them.
"""

from __future__ import annotations

import copy
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _read_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):  # an empty file parses to None
        msg = f"{name} must contain a top-level mapping, got {type(data).__name__}."
        raise ValueError(msg)
    return data


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that the configuration is internally consistent.

    Parameters
    ----------
    cfg : dict
        Parsed contents of ``config/config.yaml``.

    Raises
    ------
    ValueError
        If any rule is violated. Explicit exceptions are used instead of
        ``assert``, which Python strips under ``-O``.
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
def _config_cached() -> dict[str, Any]:
    cfg = _read_yaml("config.yaml")
    validate_config(cfg)
    return cfg


@lru_cache(maxsize=1)
def _datasets_cached() -> dict[str, Any]:
    datasets = _read_yaml("datasets.yaml").get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("datasets.yaml must contain a top-level 'datasets' mapping.")
    return datasets


def load_config() -> dict[str, Any]:
    """Return the validated project configuration.

    Returns
    -------
    dict
        A deep copy, so no caller can mutate the shared configuration.
    """
    return copy.deepcopy(_config_cached())


def load_datasets() -> dict[str, Any]:
    """Return the dataset registry from ``config/datasets.yaml``.

    Returns
    -------
    dict
        A deep copy, keyed by dataset identifier.
    """
    return copy.deepcopy(_datasets_cached())


def get_path(layer: str) -> Path:
    """Return the absolute path of a configured location.

    Parameters
    ----------
    layer : str
        Key under ``paths`` in ``config.yaml``, e.g. ``"raw"`` or ``"processed"``.

    Returns
    -------
    pathlib.Path
        Absolute path inside the project root.

    Raises
    ------
    KeyError
        If ``layer`` is not configured.
    """
    relative = load_config()["paths"][layer]
    if not isinstance(relative, str):
        raise TypeError(f"paths.{layer} must be a string, got {type(relative).__name__}.")
    return PROJECT_ROOT / relative


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a seeded NumPy ``Generator``.

    Pass the generator explicitly to every stochastic function rather than
    relying on global state.

    Parameters
    ----------
    seed : int, optional
        Overrides ``project.random_seed`` (used for multi-seed robustness runs).

    Returns
    -------
    numpy.random.Generator
    """
    if seed is None:
        seed = int(load_config()["project"]["random_seed"])
    return np.random.default_rng(seed)


def set_global_seeds(seed: int | None = None) -> int:
    """Seed the legacy global RNGs for libraries that ignore explicit generators.

    Parameters
    ----------
    seed : int, optional
        Overrides ``project.random_seed``.

    Returns
    -------
    int
        The seed that was applied (log it with every experiment).
    """
    if seed is None:
        seed = int(load_config()["project"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy API, intentionally seeded
    return seed
