"""ERA5 acquisition tests: a fake CDS API serves small ZIP-of-NetCDF files (offline)."""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_impact.config import load_config, load_datasets
from climate_impact.ingest.era5 import (
    DATASET,
    Chunk,
    chunk_request,
    make_chunks,
    run_once,
    validate_era5_zip,
)
from climate_impact.ingest.http import DownloadError
from climate_impact.provenance import load_manifest

META = load_datasets()[DATASET]


def write_member(path, var, units, years, hour=0, short=0):
    times = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")
    times = times[: len(times) - short] + pd.Timedelta(hours=hour)
    dims = ("valid_time", "latitude", "longitude")
    data = np.ones((len(times), 2, 3), dtype="float32")
    ds = xr.Dataset(
        {var: (dims, data, {"units": units})},
        coords={"valid_time": times, "latitude": [10.0, -10.0], "longitude": [0.0, 90.0, 180.0]},
    )
    ds.to_netcdf(path, engine="netcdf4")


def make_zip(path, years, drop=None, t2m_units="K", short=0):
    """Build a ZIP shaped like the real CDS delivery (tp stamped at 06:00)."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path, "w") as archive:
        for var, units, step, hour in (("t2m", t2m_units, "avgua", 0), ("tp", "m", "avgad", 6)):
            if var == drop:
                continue
            member = Path(tmp) / f"data_stream-moda_stepType-{step}.nc"
            write_member(member, var, units, years, hour=hour, short=short)
            archive.write(member, arcname=member.name)


class FakeApi:
    def __init__(self, **zip_kwargs):
        self.zip_kwargs = zip_kwargs
        self.statuses = {}
        self.years = {}
        self.submitted = []

    def submit(self, dataset, request):
        job_id = f"job-{request['year'][0]}"
        self.submitted.append(job_id)
        self.statuses[job_id] = "accepted"
        self.years[job_id] = [int(y) for y in request["year"]]
        return job_id

    def status(self, job_id):
        return self.statuses[job_id]

    def download(self, job_id, target):
        make_zip(target, self.years[job_id], **self.zip_kwargs)


@pytest.fixture
def layout(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    return tmp_path, raw, raw / "manifest.json"


CHUNKS = [Chunk(2000, 2000), Chunk(2001, 2001), Chunk(2002, 2002)]


def run(layout, api, state, max_active=2):
    root, raw, manifest = layout
    return run_once(
        META,
        CHUNKS,
        api=api,
        state=state,
        raw_dir=raw,
        manifest=manifest,
        root=root,
        max_active=max_active,
        grid=None,
    )


def test_chunks_cover_every_year_exactly_once():
    chunks = make_chunks(1960, 2025, 6)
    assert [y for c in chunks for y in c.years] == list(range(1960, 2026))
    assert len(chunks) == 11
    assert chunks[-1] == Chunk(2020, 2025)


def test_catalogue_years_include_the_season_start_year():
    analysis = load_config()["analysis"]
    assert META["years"] == [analysis["study_start_year"] - 1, analysis["study_end_year"]]


def test_request_lists_all_chunk_years_without_mutating_catalogue():
    request = chunk_request(META, Chunk(1960, 1962))
    assert request["year"] == ["1960", "1961", "1962"]
    assert "year" not in META["request"]


def test_validator_accepts_the_real_layout_with_06h_precipitation_stamps(tmp_path):
    path = tmp_path / "ok.zip"
    make_zip(path, [2000, 2001])
    validate_era5_zip(path, [2000, 2001], grid=None)


def test_validator_rejects_missing_variable(tmp_path):
    path = tmp_path / "bad.zip"
    make_zip(path, [2000], drop="tp")
    with pytest.raises(DownloadError, match="missing variable tp"):
        validate_era5_zip(path, [2000], grid=None)


def test_validator_rejects_missing_month(tmp_path):
    path = tmp_path / "bad.zip"
    make_zip(path, [2000], short=1)
    with pytest.raises(DownloadError, match="covers 11 of 12 months"):
        validate_era5_zip(path, [2000], grid=None)


def test_validator_rejects_wrong_units(tmp_path):
    path = tmp_path / "bad.zip"
    make_zip(path, [2000], t2m_units="degC")
    with pytest.raises(DownloadError, match="units"):
        validate_era5_zip(path, [2000], grid=None)


def test_validator_rejects_non_zip(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"<html>queue error</html>")
    with pytest.raises(DownloadError, match="not a ZIP"):
        validate_era5_zip(path, [2000], grid=None)


def test_first_pass_submits_up_to_max_active_and_remembers_ids(layout):
    api, state = FakeApi(), {}
    counts = run(layout, api, state)
    assert counts["submitted"] == 2
    assert state == {"2000_2000": "job-2000", "2001_2001": "job-2001"}


def test_second_pass_reattaches_without_resubmitting(layout):
    api, state = FakeApi(), {}
    run(layout, api, state)
    counts = run(layout, api, state)
    assert counts["waiting"] == 2
    assert api.submitted == ["job-2000", "job-2001"]


def test_successful_job_is_validated_stored_and_registered(layout):
    _, raw, manifest = layout
    api, state = FakeApi(), {}
    run(layout, api, state)
    api.statuses["job-2000"] = "successful"
    counts = run(layout, api, state)
    assert counts["stored"] == 1
    assert (raw / "era5" / "era5_monthly_2000_2000.zip").exists()
    [entry] = load_manifest(manifest)
    assert entry.retrieval == "cdsapi"
    assert entry.request["job_id"] == "job-2000"
    assert "2000_2000" not in state


def test_invalid_result_is_not_stored_and_job_id_is_kept(layout):
    _, raw, manifest = layout
    api, state = FakeApi(drop="tp"), {}
    run(layout, api, state)
    api.statuses["job-2000"] = "successful"
    counts = run(layout, api, state)
    assert counts["failed"] == 1
    assert not any((raw / "era5").glob("*.zip*"))
    assert load_manifest(manifest) == []
    assert state["2000_2000"] == "job-2000"


def test_failed_cds_job_is_forgotten_for_resubmission(layout):
    api, state = FakeApi(), {}
    run(layout, api, state)
    api.statuses["job-2000"] = "failed"
    counts = run(layout, api, state)
    assert counts["failed"] == 1
    assert "2000_2000" not in state
    assert counts["submitted"] == 0


def test_stored_chunk_is_skipped_on_rerun(layout):
    api, state = FakeApi(), {}
    run(layout, api, state)
    api.statuses["job-2000"] = "successful"
    run(layout, api, state)
    counts = run(layout, api, state)
    assert counts["done"] == 1
    assert api.submitted.count("job-2000") == 1
