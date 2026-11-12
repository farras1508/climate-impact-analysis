"""Configuration is a contract: test that it is complete and consistent."""

import copy

import pytest

from climate_impact.config import load_config, load_datasets, validate_config

REQUIRED_DATASET_KEYS = {"provider", "access", "license", "priority", "questions"}


def test_config_loads_and_validates():
    assert load_config()["analysis"]["baseline_period"] == [1991, 2020]


@pytest.mark.parametrize(
    ("section", "key", "bad_value", "expected_error"),
    [
        ("splits", "train_end_year", 2020, "Splits must satisfy"),
        ("analysis", "baseline_period", [1981, 2020], "30-year climate normal"),
        ("statistics", "ci_level", 0.90, "ci_level must equal"),
        ("analysis", "min_valid_months", 13, "min_valid_months must be between"),
    ],
)
def test_validator_rejects_bad_config(section, key, bad_value, expected_error):
    cfg = copy.deepcopy(load_config())
    cfg[section][key] = bad_value
    with pytest.raises(ValueError, match=expected_error):
        validate_config(cfg)


def test_callers_cannot_mutate_shared_config():
    cfg = load_config()
    cfg["analysis"]["study_end_year"] = 1900
    assert load_config()["analysis"]["study_end_year"] == 2025


def test_every_dataset_has_provenance_fields():
    for name, meta in load_datasets().items():
        missing = REQUIRED_DATASET_KEYS - meta.keys()
        assert not missing, f"{name} missing {missing}"


def test_core_datasets_cover_every_question():
    """Each analytical question Q1-Q5 must be served by at least one core dataset."""
    covered = {
        q
        for meta in load_datasets().values()
        if meta["priority"] == "core"
        for q in meta["questions"]
    }
    assert covered >= {"Q1", "Q2", "Q3", "Q4", "Q5"}


def test_triangulation_needs_three_temperature_datasets():
    """Success criterion: global trend estimated from >= 3 independent datasets."""
    names = load_config()["analysis"]["temperature_datasets"]
    datasets = load_datasets()
    assert len(names) >= 3
    assert all(n in datasets for n in names)
