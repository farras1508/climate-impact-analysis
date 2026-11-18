"""Profiler tests (offline, synthetic data)."""

import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_impact.ingest.readers import RawTable
from climate_impact.provenance import ManifestEntry
from climate_impact.quality import profile

GISTEMP = (
    "Land-Ocean: Global Means\n"
    "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON\n"
    "1880,-.19,-.25,-.10,-.17,-.10,-.21,-.19,-.11,-.15,-.23,-.22,-.18,-.18,***,***,-.12,-.17,-.20\n"
    "1881,-.20,-.14,.03,.05,.06,-.19,.00,-.04,-.15,-.22,***,***,***,***,-.09,.05,-.08,-.17\n"
)


def entry(dataset, file):
    return ManifestEntry(dataset, file, "0" * 64, 0, "", "", "2026-11-18", "", "http")


def write(root, rel, text):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return rel


def test_numeric_column_profile():
    p = profile.profile_column(pd.Series([1.0, -2.0, 0.0, np.nan]))
    assert p["kind"] == "numeric"
    assert (p["n_missing"], p["n_negative"], p["n_zero"]) == (1, 1, 1)
    assert p["min"] == -2.0
    assert p["max"] == 1.0
    assert p["pct_missing"] == 25.0


def test_text_values_listed_only_for_low_cardinality():
    few = profile.profile_column(pd.Series(["a", "b", "a"]))
    assert few["top_values"] == {"a": 2, "b": 1}
    many = profile.profile_column(pd.Series([f"event {i}" for i in range(60)]))
    assert "top_values" not in many  # licence safeguard: no record-level text in reports
    assert many["n_unique"] == 60


def test_all_missing_column_does_not_fail():
    p = profile.profile_column(pd.Series([np.nan, np.nan], dtype="float64"))
    assert p["min"] is None
    assert p["n_missing"] == 2


def test_table_periods_distinguish_rows_from_values():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-11-01", "2025-12-01", "2026-01-01"]),
            "iso3": ["KEN", "KEN", None],
            "v": [1.0, 2.0, np.nan],
        }
    )
    table = RawTable("d", "t", frame, ["f"], "date", "iso3", "M", ["v"], ["iso3", "date"])
    p = profile.profile_table(table)
    assert p["period_all_rows"] == {"first": "2025-11", "last": "2026-01", "n_periods": 3}
    assert p["period_with_values"]["last"] == "2025-12"
    assert (p["n_entities"], p["rows_without_entity"]) == (1, 1)
    assert p["missing_cells"] == 2


def era5_member(path, var, units, hour, values, expver):
    times = pd.to_datetime(["2020-01-01", "2020-02-01"]) + pd.Timedelta(hours=hour)
    ds = xr.Dataset(
        {var: (("valid_time", "latitude", "longitude"), values, {"units": units})},
        coords={
            "valid_time": times,
            "latitude": [90.0, 0.0, -90.0],
            "longitude": [0.0, 90.0, 180.0, 270.0],
            "expver": ("valid_time", expver),
        },
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def era5_zip(tmp_path):
    t2m = np.full((2, 3, 4), 288.0)
    t2m[:, 1, :] = 300.0  # equator warmer: weighted mean must exceed the plain mean
    tp = np.full((2, 3, 4), 0.003)
    tp[1, 0, 0] = -1e-9  # tiny negative accumulation: must be counted, not hidden
    members = [
        era5_member(tmp_path / "a.nc", "t2m", "K", 0, t2m, ["0001", "0001"]),
        era5_member(tmp_path / "b.nc", "tp", "m", 6, tp, ["0001", "0005"]),
    ]
    rel = "era5/era5_monthly_2020_2020.zip"
    (tmp_path / "era5").mkdir()
    with zipfile.ZipFile(tmp_path / rel, "w") as archive:
        for m in members:
            archive.write(m, m.name)
    return tmp_path, rel


def test_inspect_era5(era5_zip):
    root, rel = era5_zip
    result = profile.inspect_era5([entry("era5_monthly_single_levels", rel)], root)
    (chunk,) = result["chunks"]
    t2m_member, tp_member = chunk["members"]
    t2m = t2m_member["variables"]["t2m"]
    assert t2m["units"] == "K"
    assert t2m["area_weighted_mean"] == pytest.approx(300.0)  # cos(+-90 deg) = 0 weight
    assert (t2m["min"], t2m["max"]) == (288.0, 300.0)
    assert tp_member["time_of_day"] == ["06:00"]
    assert tp_member["expver_counts"] == {"0001": 1, "0005": 1}
    assert tp_member["variables"]["tp"]["n_negative"] == 1
    assert t2m_member["latitude"] == {"first": 90.0, "last": -90.0, "n": 3}
    assert result["coverage"]["tp"] == {
        "first": "2020-01",
        "last": "2020-02",
        "n_months": 2,
        "n_duplicated": 0,
        "missing_months": [],
    }
    text = "\n".join(profile.render_era5(result))
    assert "| `tp` |" in text


def test_profile_dataset_and_reports(tmp_path):
    rel = write(tmp_path, "gistemp/GLB.csv", GISTEMP)
    manifest = [entry("gistemp_v4_global_monthly", rel)]
    catalogue = {"gistemp_v4_global_monthly": {"native_baseline": [1951, 1980], "url": "x"}}
    result = profile.profile_dataset("gistemp_v4_global_monthly", manifest, catalogue, tmp_path)
    assert result["declared"] == {"native_baseline": [1951, 1980]}  # only declared fields
    (table,) = result["tables"]
    assert table["period_with_values"] == {"first": "1880-01", "last": "1881-10", "n_periods": 22}
    clean = profile._jsonable(result)
    page = profile.render_dataset("gistemp_v4_global_monthly", clean)
    assert "`value: ***` x2" in page
    inventory = profile.render_inventory({"gistemp_v4_global_monthly": clean})
    assert "1880-01 .. 1881-10 (22)" in inventory
