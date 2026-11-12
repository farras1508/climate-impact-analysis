"""Smoke tests: the scientific stack imports and its compiled I/O layers work."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PACKAGES = [
    "pyarrow", "yaml", "netCDF4", "dask", "geopandas", "shapely", "regionmask",
    "scipy", "statsmodels", "linearmodels", "pyfixest", "pymannkendall",
    "ruptures", "piecewise_regression", "pyextremes", "sklearn", "lightgbm",
    "optuna", "shap", "mapie", "mlflow", "pandera", "matplotlib", "seaborn",
    "plotly", "streamlit", "requests", "cdsapi", "wbgapi", "country_converter",
    "missingno", "climate_impact",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_import(name: str) -> None:
    importlib.import_module(name)


def test_netcdf_roundtrip(tmp_path) -> None:
    """HDF5/netCDF C libraries: write and read back a tiny ERA5-like cube."""
    rng = np.random.default_rng(42)
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), rng.normal(288.0, 5.0, (2, 3, 4)))},
        coords={"time": [0, 1], "latitude": [-10.0, 0.0, 10.0],
                "longitude": [0.0, 90.0, 180.0, 270.0]},
    )
    ds["t2m"].attrs["units"] = "K"
    path = tmp_path / "cube.nc"
    ds.to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4") as back:
        xr.testing.assert_allclose(ds, back)
        assert back["t2m"].attrs["units"] == "K"  # metadata must survive


def test_parquet_roundtrip(tmp_path) -> None:
    """Arrow layer: types must survive the interim/processed storage format."""
    df = pd.DataFrame({"iso3": ["IDN", "BRA"], "year": [2020, 2020],
                       "yield_kg_ha": [5200.5, 5600.0]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    pd.testing.assert_frame_equal(df, pd.read_parquet(path))


def test_projection_stack() -> None:
    """PROJ/GEOS: reproject a point (Jakarta) to Web Mercator."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(geometry=[Point(106.8, -6.2)], crs="EPSG:4326")
    assert gdf.to_crs("EPSG:3857").crs.to_epsg() == 3857

def test_running_in_project_python() -> None:
    """Guard: tests must run in the project's Python 3.12 environment."""
    import sys

    assert sys.version_info[:2] == (3, 12), (
        f"Running {sys.version.split()[0]} at {sys.executable}; "
        "activate the env: conda activate climate-impact"
    )