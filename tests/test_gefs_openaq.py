from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr

from mdt.tasks.data import load_data
from mdt.tasks.pairing import pair_data
from mdt.tasks.statistics import compute_statistics


def test_gefs_openaq_workflow(mocker, tmp_path):
    """Test GEFS-Aerosol model paired with OpenAQ observations."""
    mocker.patch("mdt.tasks.pairing._ensure_esmf_manager")
    # 1. Setup Mock Data
    times = pd.date_range("2023-01-01", periods=2, freq="h")
    lats = np.linspace(30, 50, 5)
    lons = np.linspace(-120, -70, 5)

    # GEFS Mock (Xarray)
    gefs_ds = xr.Dataset(
        {"dust": (("time", "lat", "lon"), np.random.rand(2, 5, 5))}, coords={"time": times, "lat": lats, "lon": lons}, attrs={"history": "Initial"}
    )

    # OpenAQ mock source data is tabular, but its MonetIO reader contract returns xarray.
    openaq_df = pd.DataFrame({"time": [times[0], times[1]], "lat": [40.0, 41.0], "lon": [-100.0, -101.0], "pm25": [10.5, 12.0]})

    # 2. Mock monetio.load
    def mock_load(dataset_type, **kwargs):
        if dataset_type == "gefs":
            return gefs_ds
        if dataset_type == "openaq":
            if kwargs.get("as_xarray"):
                return openaq_df.set_index(["time", "lat", "lon"]).to_xarray()
            return openaq_df
        return None

    mocker.patch("monetio.load", side_effect=mock_load)

    # 3. Execution
    grib_file = tmp_path / "sample.grib2"
    grib_file.touch()
    ds_mod = load_data("gefs_data", "gefs", {"files": str(grib_file)})
    ds_obs = load_data("openaq_data", "openaq", {"files": "dummy.json", "as_xarray": True})

    # Pair
    # Mock Regridder for point pairing
    mock_regridder_obj = MagicMock()
    mock_regridder_obj.side_effect = lambda ds: xr.Dataset({"dust": (("node"), [0.5, 0.6])}, coords={"node": [0, 1], "time": times})
    mocker.patch("xregrid.Regridder", return_value=mock_regridder_obj)
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    import xregrid

    mocker.patch("monet.util.resample.Regridder", xregrid.Regridder, create=True)

    paired = pair_data("gefs_openaq", "nearest", ds_mod, ds_obs, {"merge": True})

    assert "dust" in paired.data_vars
    assert "pm25" in paired.data_vars

    # Statistics
    stats = compute_statistics("gefs_stats", ["rmse", "mb"], paired, {"obs_var": "pm25", "mod_var": "dust"})

    assert "rmse" in stats
    assert "mb" in stats
    assert "history" in stats["rmse"].attrs
