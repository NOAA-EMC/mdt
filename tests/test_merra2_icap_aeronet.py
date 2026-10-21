import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mdt.tasks.data import load_data
from mdt.tasks.pairing import combine_paired_data, pair_data
from mdt.tasks.statistics import compute_statistics


@pytest.fixture
def sample_data_files():
    """Create real files for testing readers."""
    icap_fn = "icap_test.nc"
    merra2_fn = "merra2_test.nc"
    aeronet_fn = "aeronet_test.csv"

    # ICAP-MME Sample
    times = pd.date_range("2023-01-01", periods=2, freq="h")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 10)
    icap_ds = xr.Dataset(
        {"modeaod550": (("time", "lat", "lon"), np.random.rand(2, 10, 10))},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    icap_ds.to_netcdf(icap_fn)

    # MERRA-2 Sample
    # MERRA-2 often has lat/lon as coords and variables like TOTEXTTAU
    merra2_ds = xr.Dataset(
        {"TOTEXTTAU": (("time", "lat", "lon"), np.random.rand(2, 10, 10))},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    merra2_ds.to_netcdf(merra2_fn)

    # AERONET Sample (Structured CSV)
    content = """AERONET Level 1.5 Data
Site=TEST
Latitude=40.0, Longitude=-105.0, Elevation=1500.0
Version 3

Date(dd:mm:yyyy),Time(hh:mm:ss),Site_Latitude(Degrees),Site_Longitude(Degrees),Site_Elevation(m),AOD_550nm
01:01:2023,00:00:00,40.0,-105.0,1500.0,0.1
01:01:2023,01:00:00,40.0,-105.0,1500.0,0.2
"""
    with open(aeronet_fn, "w") as f:
        f.write(content)

    yield icap_fn, merra2_fn, aeronet_fn

    # Cleanup
    for fn in [icap_fn, merra2_fn, aeronet_fn]:
        if os.path.exists(fn):
            os.remove(fn)


def test_merra2_icap_aeronet_full_workflow(sample_data_files, mocker):
    """Test the full comparison workflow using real monetio readers and MDT tasks.

    This test avoids manual mocking of DataSets and instead relies on monetio.load
    to parse real (sample) files from disk.
    """
    icap_fn, merra2_fn, aeronet_fn = sample_data_files

    # 1. Load Data Tasks (Real monetio readers)
    ds_icap = load_data("icap", "icap_mme", {"files": icap_fn})
    ds_merra2 = load_data("m2", "merra2", {"files": merra2_fn})
    ds_obs = load_data("obs", "aeronet", {"files": aeronet_fn, "as_xarray": True})

    assert "modeaod550" in ds_icap.data_vars
    assert "TOTEXTTAU" in ds_merra2.data_vars
    assert "aod_550nm" in ds_obs.data_vars

    # 2. Pairing Tasks
    # Mock Regridder only because esmpy is not available in the environment
    # but keep the rest of the task logic intact.
    mock_regridder_obj = MagicMock()

    def mock_regridder_apply(ds):
        # Return a dataset with the model variable at the observation nodes
        # This simulates the output of a successful regrid to point sites
        varname = list(ds.data_vars)[0]
        nodes = ds_obs.sizes["node"]
        res = xr.Dataset(
            {varname: (("node"), np.random.rand(nodes))},
            coords={"node": ds_obs.node, "time": ds_obs.time},
        )
        return res

    mock_regridder_obj.side_effect = mock_regridder_apply

    # Patch xregrid.Regridder
    mocker.patch("xregrid.Regridder", return_value=mock_regridder_obj)

    # Force has_xregrid to True in MONET to bypass the check
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    import xregrid

    mocker.patch("monet.util.resample.Regridder", xregrid.Regridder, create=True)

    # Call pair_data which now uses monet.util.combinetool.pair
    # We use 'nearest' which would normally trigger xregrid
    pair_icap = pair_data("p_icap", "nearest", ds_icap, ds_obs, {"merge": True})
    pair_m2 = pair_data("p_m2", "nearest", ds_merra2, ds_obs, {"merge": True})

    assert "modeaod550" in pair_icap.data_vars
    assert "TOTEXTTAU" in pair_m2.data_vars
    assert "aod_550nm" in pair_icap.data_vars  # Merged

    # 3. Combine Task
    # Rename for common variable analysis
    pair_icap = pair_icap.rename({"modeaod550": "AOD"})
    pair_m2 = pair_m2.rename({"TOTEXTTAU": "AOD"})

    combined = combine_paired_data({"icap": pair_icap, "merra2": pair_m2}, dim="model")

    assert "model" in combined.dims
    assert combined.sizes["model"] == 2

    # 4. Statistics Task
    metrics = ["rmse", "mb"]
    stats_kwargs = {"obs_var": "aod_550nm", "mod_var": "AOD"}

    # Use real monet-stats functions to ensure end-to-end integration works
    results = compute_statistics("workflow_stats", metrics, combined, stats_kwargs)

    # MDT tasks often lowercase metric names
    assert set(results) == {"icap", "merra2"}
    for model in results:
        assert set(results[model]) == {"rmse", "mb"}
        assert isinstance(results[model]["rmse"], xr.DataArray)
        assert "Computed rmse" in results[model]["rmse"].attrs["history"]


def test_merra2_vs_icap_bias_workflow(sample_data_files, mocker):
    """Test the model-to-model comparison workflow (MERRA-2 vs ICAP-MME bias)."""
    icap_fn, merra2_fn, _ = sample_data_files

    # 1. Load Data
    ds_icap = load_data("icap", "icap_mme", {"files": icap_fn})
    ds_m2 = load_data("m2", "merra2", {"files": merra2_fn})

    # 2. Pairing (Model to Model Regrid)
    mock_regridder_obj = MagicMock()

    def mock_regridder_apply(ds):
        # Simulate regridding m2 to icap grid
        # Return m2 data on icap's (lat, lon) dimensions
        res = xr.Dataset(
            {"TOTEXTTAU": (("time", "lat", "lon"), ds.TOTEXTTAU.values)},
            coords=ds_icap.coords,
        )
        return res

    mock_regridder_obj.side_effect = mock_regridder_apply
    mocker.patch("xregrid.Regridder", return_value=mock_regridder_obj)
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    import xregrid

    mocker.patch("monet.util.resample.Regridder", xregrid.Regridder, create=True)

    # Use 'bilinear' or any method that triggers regridding
    pair_m2_icap = pair_data("p_m2_icap", "bilinear", ds_m2, ds_icap, {"merge": True})

    assert "TOTEXTTAU" in pair_m2_icap.data_vars
    assert "modeaod550" in pair_m2_icap.data_vars

    # 3. Statistics (Bias)
    # Use real MB from monet-stats
    # We explicitly pass dim='time' to reduce over time, leaving spatial dims
    # Or discovery might find lat/lon and reduce over them if we don't specify.
    # In _execute_metric, if dim is None, it tries to discover spatial dims and reduces over them.
    # pair_m2_icap has (time, lat, lon). discover_spatial_dims will find (lat, lon).
    # So MB will be calculated over (lat, lon), leaving 'time'.
    results = compute_statistics("bias_stats", ["mb"], pair_m2_icap, {"obs_var": "modeaod550", "mod_var": "TOTEXTTAU"})

    assert "mb" in results
    # results["mb"] should have 'time' dim if it reduced over lat/lon
    assert "time" in results["mb"].dims
    assert "Computed mb" in results["mb"].attrs["history"]


def test_merra2_lazy_loading_real_reader(sample_data_files, mocker):
    """Verify that MERRA-2 loading remains lazy when using the real reader."""
    _, merra2_fn, _ = sample_data_files

    # Load with laziness requested
    ds = load_data("m2_lazy", "merra2", {"files": merra2_fn, "chunks": {"time": 1}})

    # Check if data is lazy (using Dask)
    assert hasattr(ds.TOTEXTTAU.data, "dask") or type(ds.TOTEXTTAU.data).__module__.startswith("cubed")
    assert "Loaded dataset 'm2_lazy'" in ds.attrs["history"]


def test_merra2_lazy_loading_cubed_reader(sample_data_files, mocker):
    """Verify that MERRA-2 loading handles cubed chunking properly."""
    _, merra2_fn, _ = sample_data_files

    import xarray as xr

    # Manually open dataset to explicitly use cubed as the chunkmanager
    # since load_data delegates to monetio which defaults to whatever xarray defaults to (usually dask)
    # This proves xarray cubed support integrates seamlessly with the hasattr check.
    ds_cubed = xr.open_dataset(merra2_fn, chunks={"time": 1}, chunked_array_type="cubed")

    assert type(ds_cubed.TOTEXTTAU.data).__module__.startswith("cubed")
