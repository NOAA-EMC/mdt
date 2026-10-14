"""DataArray output persistence tests."""

import numpy as np
import xarray as xr

from mdt.tasks.data import save_data


def test_save_dataarray_to_zarr(tmp_path):
    """Persist and reload DataArray task output without converting it to pandas."""
    array = xr.DataArray(np.arange(4.0), dims="sample", name="value")
    url = tmp_path / "dataarray.zarr"

    save_data("dataarray", array, backend="zarr", url=str(url))

    xr.testing.assert_equal(array, xr.open_dataarray(url))
