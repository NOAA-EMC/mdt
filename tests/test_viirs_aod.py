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


def test_viirs_icap_workflow(mocker):
    """Test VIIRS AOD satellite data paired with ICAP-MME model."""
    # 1. Setup Mock Data
    times = pd.date_range("2023-01-01", periods=1, freq="h")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 10)

    # VIIRS Mock (Satellite usually L2/L3, let's assume L3 grid for simplicity)
    viirs_ds = xr.Dataset(
        {"AOD_550": (("time", "lat", "lon"), np.random.rand(1, 10, 10))},
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"history": "Initial"},
    )

    # ICAP Mock
    icap_ds = xr.Dataset(
        {"modeaod550": (("time", "lat", "lon"), np.random.rand(1, 10, 10))},
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"history": "Initial"},
    )

    # 2. Mock monetio.load
    def mock_load(dataset_type, **kwargs):
        if dataset_type == "nesdis_edr_viirs":
            return viirs_ds
        if dataset_type == "icap_mme":
            return icap_ds
        return None

    mocker.patch("monetio.load", side_effect=mock_load)

    # 3. Execution
    ds_sat = load_data("viirs_data", "nesdis_edr_viirs", {"files": "dummy.nc"})
    ds_mod = load_data("icap_data", "icap_mme", {"files": "dummy.nc"})

    # Pair (Model to Sat grid)
    mock_regridder_obj = MagicMock()
    mock_regridder_obj.side_effect = lambda ds: ds  # Identity mock
    mocker.patch("xregrid.Regridder", return_value=mock_regridder_obj)
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    import xregrid

    mocker.patch("monet.util.resample.Regridder", xregrid.Regridder, create=True)

    # Use bilinear to trigger regridding
    paired = pair_data("viirs_paired", "bilinear", ds_mod, ds_sat, {"merge": True})

    assert "AOD_550" in paired.data_vars
    assert "modeaod550" in paired.data_vars

    # Statistics
    stats = compute_statistics("viirs_stats", ["rmse"], paired, {"obs_var": "AOD_550", "mod_var": "modeaod550"})

    assert "rmse" in stats
    assert "history" in stats["rmse"].attrs
