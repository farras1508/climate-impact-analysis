"""ERA5 acquisition tests: a fake CDS client writes small NetCDF files (offline)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_impact.config import load_config, load_datasets
from climate_impact.ingest.era5 import DATASET, download_year, validate_era5, year_request
from climate_impact.ingest.http import DownloadError
from climate_impact.provenance import load_manifest

META = load_datasets()[DATASET]


def write_fake_era5(path, months=12, drop=None, tp_units="m"):
    shape = (months, 3, 4)
    rng = np.random.default_rng(0)
    dims = ("valid_time", "latitude", "longitude")
    ds = xr.Dataset(
        {
            "t2m": (dims, rng.normal(288.0, 5.0, shape), {"units": "K"}),
            "tp": (dims, rng.random(shape) * 0.01, {"units": tp_units}),
        },
        coords={
            "valid_time": pd.date_range("2000-01-01", periods=months, freq="MS"),
            "latitude": [10.0, 0.0, -10.0],
            "longitude": [0.0, 90.0, 180.0, 270.0],
        },
    )
    if drop:
        ds = ds.drop_vars(drop)
    ds.to_netcdf(path, engine="netcdf4")


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def retrieve(self, name, request, target):
        self.calls.append((name, request))
        write_fake_era5(target, **self.kwargs)


@pytest.fixture
def layout(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    return tmp_path, raw, raw / "manifest.json"


def run_year(layout, client, year=2000):
    root, raw, manifest = layout
    return download_year(
        year, META, client=client, raw_dir=raw, manifest=manifest, root=root, min_bytes=0
    )


def test_request_is_built_per_year_without_mutating_catalogue():
    request = year_request(META, 1961)
    assert request["year"] == ["1961"]
    assert request["variable"] == ["2m_temperature", "total_precipitation"]
    assert "year" not in META["request"]


def test_catalogue_years_include_the_season_start_year():
    analysis = load_config()["analysis"]
    assert META["years"] == [analysis["study_start_year"] - 1, analysis["study_end_year"]]


def test_valid_year_is_stored_and_recorded(layout):
    _, raw, manifest = layout
    result = run_year(layout, FakeClient())
    assert result.action == "downloaded"
    assert (raw / "era5" / "era5_monthly_2000.nc").exists()
    [entry] = load_manifest(manifest)
    assert entry.retrieval == "cdsapi"
    assert entry.request["year"] == ["2000"]


def test_rerun_skips_without_calling_cds(layout):
    run_year(layout, FakeClient())
    client = FakeClient()
    assert run_year(layout, client).action == "skipped"
    assert client.calls == []


def test_missing_variable_is_rejected_and_nothing_stored(layout):
    _, raw, manifest = layout
    with pytest.raises(DownloadError, match="missing variable tp"):
        run_year(layout, FakeClient(drop="tp"))
    assert not any(raw.rglob("*.nc*"))
    assert load_manifest(manifest) == []


def test_incomplete_year_is_rejected(layout):
    with pytest.raises(DownloadError, match="expected 12 months"):
        run_year(layout, FakeClient(months=11))


def test_unexpected_units_are_rejected(layout):
    with pytest.raises(DownloadError, match="units"):
        run_year(layout, FakeClient(tp_units="mm"))


def test_non_netcdf_file_is_rejected(tmp_path):
    page = tmp_path / "bad.nc"
    page.write_bytes(b"<html>queue error</html>")
    with pytest.raises(DownloadError, match="not a NetCDF"):
        validate_era5(page, min_bytes=0)
