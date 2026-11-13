"""ABT schemas are pre-registered data contracts: test that they are well formed."""

import pytest

from climate_impact.config import load_abt_schemas, load_config

ALLOWED_DTYPES = {"int", "float", "str", "bool", "date"}
SCHEMAS = load_abt_schemas()


@pytest.mark.parametrize("table", sorted(SCHEMAS))
def test_primary_key_columns_exist_and_are_required(table):
    spec = SCHEMAS[table]
    columns = {c["name"]: c for c in spec["columns"]}
    for key in spec["primary_key"]:
        assert key in columns, f"{table}: key '{key}' not declared as a column"
        assert columns[key]["nullable"] is False, f"{table}: key '{key}' must be non-nullable"


@pytest.mark.parametrize("table", sorted(SCHEMAS))
def test_every_column_has_dtype_unit_and_valid_range(table):
    names = [c["name"] for c in SCHEMAS[table]["columns"]]
    assert len(names) == len(set(names)), f"{table}: duplicate column names"
    for col in SCHEMAS[table]["columns"]:
        assert col["dtype"] in ALLOWED_DTYPES, f"{table}.{col['name']}: bad dtype"
        assert col.get("unit"), f"{table}.{col['name']}: missing unit"
        if "range" in col:
            low, high = col["range"]
            assert low < high, f"{table}.{col['name']}: empty range"


def test_foreign_keys_reference_declared_columns():
    for table, spec in SCHEMAS.items():
        own = {c["name"] for c in spec["columns"]}
        for fk in spec.get("foreign_keys", []):
            target = {c["name"] for c in SCHEMAS[fk["table"]]["columns"]}
            assert set(fk["columns"]) <= own, f"{table}: FK columns not in table"
            assert set(fk["columns"]) <= target, f"{table}: FK target columns missing"


def test_global_row_counts_match_coverage():
    start, end = SCHEMAS["abt_global_annual"]["coverage"]
    years = end - start + 1
    assert SCHEMAS["abt_global_annual"]["expected_rows"] == years
    assert SCHEMAS["abt_global_monthly"]["expected_rows"] == 12 * years


def test_coverage_agrees_with_study_window_in_config():
    analysis = load_config()["analysis"]
    window = [analysis["study_start_year"], analysis["study_end_year"]]
    assert SCHEMAS["abt_country_year"]["coverage"] == window
    assert SCHEMAS["abt_crop_panel"]["coverage"] == window
    assert SCHEMAS["abt_global_annual"]["coverage"][1] == analysis["study_end_year"]
