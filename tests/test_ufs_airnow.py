from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mdt.tasks.data import load_data
from mdt.tasks.pairing import pair_data
from mdt.tasks.statistics import compute_statistics


@pytest.fixture(autouse=True)
def _mock_esmf_manager(mocker):
    """Keep mock-data workflow tests independent of ESMF initialization."""
    mocker.patch("mdt.tasks.pairing._ensure_esmf_manager")


def test_ufs_airnow_workflow(mocker):
    """Test UFS-AQM model paired with AirNow observations."""
    # 1. Setup Mock Data
    times = pd.date_range("2023-01-01", periods=2, freq="h")
    lats = np.linspace(30, 50, 5)
    lons = np.linspace(-120, -70, 5)

    # UFS Mock (Xarray)
    ufs_ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), np.random.rand(2, 5, 5))}, coords={"time": times, "lat": lats, "lon": lons}, attrs={"history": "Initial"}
    )

    # AirNow Mock
    airnow_df = pd.DataFrame({"time": [times[0], times[1]], "lat": [35.0, 36.0], "lon": [-110.0, -111.0], "ozone": [40.0, 45.0]})

    # 2. Mock monetio.load
    def mock_load(dataset_type, **kwargs):
        if dataset_type == "ufs":
            return ufs_ds
        if dataset_type == "airnow":
            if kwargs.get("as_xarray"):
                return airnow_df.set_index(["time", "lat", "lon"]).to_xarray()
            return airnow_df
        return None

    mocker.patch("monetio.load", side_effect=mock_load)

    # 3. Execution
    ds_mod = load_data("ufs_data", "ufs", {"files": "dummy.nc"})
    ds_obs = load_data("airnow_data", "airnow", {"files": "dummy.csv", "as_xarray": True})

    # Pair
    mock_regridder_obj = MagicMock()
    mock_regridder_obj.side_effect = lambda ds: xr.Dataset({"O3": (("node"), [0.4, 0.45])}, coords={"node": [0, 1], "time": times})
    mocker.patch("xregrid.Regridder", return_value=mock_regridder_obj)
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    import xregrid

    mocker.patch("monet.util.resample.Regridder", xregrid.Regridder, create=True)

    paired = pair_data("ufs_airnow", "nearest", ds_mod, ds_obs, {"merge": True})

    assert "O3" in paired.data_vars
    assert "ozone" in paired.data_vars

    # Statistics
    stats = compute_statistics("ufs_stats", ["rmse", "mb"], paired, {"obs_var": "ozone", "mod_var": "O3"})

    assert "rmse" in stats
    assert "mb" in stats
    assert "history" in stats["rmse"].attrs
