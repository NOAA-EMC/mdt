import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from mdt.tasks.data import load_data
from mdt.tasks.pairing import pair_data
from mdt.tasks.statistics import compute_statistics


def test_load_data_provenance(mocker):
    """Test load_data and ensure it adds provenance metadata."""
    # Create dummy dataset
    ds = xr.Dataset({"val": (("x"), [1, 2, 3])})

    # Mock monetio.load
    mocker.patch("monetio.load", return_value=ds)

    # Load data
    res = load_data("test_data", "cmaq", {"files": "dummy.nc"})

    # Check history
    assert "Loaded dataset 'test_data'" in res.attrs["history"]


def test_load_data_icap_date_alias_to_dates(mocker):
    """Ensure MDT maps legacy `date` to reader-expected `dates` for icap_mme."""
    ds = xr.Dataset({"val": (("x"), [1, 2, 3])})
    mock_load = mocker.patch("monetio.load", return_value=ds)

    load_data(
        "icap_mme_model",
        "icap_mme",
        {
            "date": "2023-01-04",
            "product": "MMC",
            "data_var": "modeaod550",
        },
    )

    call_kwargs = mock_load.call_args.kwargs
    assert "dates" in call_kwargs
    assert call_kwargs["dates"] == "2023-01-04"
    assert "date" not in call_kwargs


def test_pair_data_double_check(mocker):
    """Test pair_data with both Eager (NumPy) and Lazy (Dask) data (Aero Protocol)."""
    # Setup mock data (NumPy)
    data = np.random.rand(10, 10)
    lat = np.linspace(0, 10, 10)
    lon = np.linspace(0, 10, 10)
    ds_eager = xr.Dataset({"val": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon})

    # Setup mock data (Dask)
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 5})

    # Force has_xregrid to True and mock monet.pair to avoid esmpy dependency
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    mocker.patch("monet.util.combinetool.pair", side_effect=lambda src, tgt, **kwargs: src)

    # Eager run
    res_eager = pair_data("test_eager", "interpolate", ds_eager, ds_eager, {})

    # Lazy run
    res_lazy = pair_data("test_lazy", "interpolate", ds_lazy, ds_lazy, {})

    # Check that the lazy run is still lazy
    import dask.array as darray

    assert isinstance(res_lazy.val.data, darray.Array) or type(res_lazy.val.data).__module__.startswith("cubed")

    # Double-Check: Results must be identical after compute
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Check history update
    assert "Paired using method 'interpolate'" in res_eager.attrs["history"]


def test_compute_statistics_double_check(mocker):
    """Test compute_statistics with both Eager (NumPy) and Lazy (Dask) data (Aero Protocol)."""
    # Metrics and kwargs
    metrics = ["RMSE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod"}

    # Mock monet_stats.RMSE
    def mock_rmse(obs, mod, **kwargs):
        if hasattr(obs, "data") and (hasattr(obs.data, "dask") or type(obs.data).__module__.startswith("cubed")):
            # Return a Lazy DataArray
            res = xr.DataArray(da.from_array(0.5, chunks=()), attrs={"history": "initial"})
            return res
        elif hasattr(obs, "attrs"):
            # Return a NumPy DataArray
            return xr.DataArray(0.5, attrs={"history": "initial"})
        return 0.5

    import monet_stats

    # We need to patch BOTH RMSE and rmse if the code searches for them
    mocker.patch.object(monet_stats, "RMSE", side_effect=mock_rmse, create=True)
    mocker.patch.object(monet_stats, "rmse", side_effect=mock_rmse, create=True)

    # 1. Eager Run (Pandas)
    df = pd.DataFrame({"obs": np.random.rand(10), "mod": np.random.rand(10)})
    results_df = compute_statistics("test_stats", metrics, df, kwargs)
    assert results_df["RMSE"] == 0.5

    # 2. Eager Run (Xarray NumPy)
    ds_eager = xr.Dataset({"obs": (("x"), np.random.rand(10)), "mod": (("x"), np.random.rand(10))})
    results_eager = compute_statistics("test_eager_stats", metrics, ds_eager, kwargs)
    assert float(results_eager["RMSE"]) == 0.5
    assert "Computed RMSE" in results_eager["RMSE"].attrs["history"]

    # 3. Lazy Run (Xarray Dask)
    ds_lazy = xr.Dataset(
        {
            "obs": (("x"), da.from_array(np.random.rand(10), chunks=5)),
            "mod": (("x"), da.from_array(np.random.rand(10), chunks=5)),
        }
    )
    results_lazy = compute_statistics("test_lazy_stats", metrics, ds_lazy, kwargs)

    # Ensure result is still lazy if input was lazy and result is an xarray object
    if hasattr(results_lazy["RMSE"], "data"):
        import dask.array as darray

        assert isinstance(results_lazy["RMSE"].data, darray.Array) or type(results_lazy["RMSE"].data).__module__.startswith("cubed")

    # Double-Check: Results must be identical
    assert float(results_lazy["RMSE"].compute()) == float(results_eager["RMSE"])
    assert "Computed RMSE" in results_lazy["RMSE"].attrs["history"]


def test_load_data_wind_speed_calculation(mocker):
    """Test that load_data dynamically calculates WSPD_10maboveground from wind components."""
    # 1. Test with UGRD and VGRD
    ds1 = xr.Dataset({"UGRD": (("x"), [3.0]), "VGRD": (("x"), [4.0])})
    mocker.patch("monetio.load", return_value=ds1)
    res1 = load_data("test_wind_ugrd", "gfs", {})
    assert "WSPD_10maboveground" in res1.variables
    np.testing.assert_allclose(res1["WSPD_10maboveground"].values, [5.0])
    assert res1["WSPD_10maboveground"].attrs["units"] == "m s-1"

    # 2. Test with u_wind and v_wind
    ds2 = xr.Dataset({"u_wind": (("x"), [3.0]), "v_wind": (("x"), [4.0])})
    mocker.patch("monetio.load", return_value=ds2)
    res2 = load_data("test_wind_u_wind", "gfs", {})
    assert "WSPD_10maboveground" in res2.variables
    np.testing.assert_allclose(res2["WSPD_10maboveground"].values, [5.0])
    assert res2["WSPD_10maboveground"].attrs["units"] == "m s-1"


def test_pair_data_time_sorting(mocker):
    """Test that pair_data correctly sorts the paired dataset by the time dimension."""
    # Create an unsorted paired dataset
    times = pd.to_datetime(["2023-08-01 05:00", "2023-08-01 02:00", "2023-08-01 08:00"])
    paired_unsorted = xr.Dataset({"val": (("time"), [1, 2, 3])}, coords={"time": times})

    # Setup dummy source and target datasets
    src = xr.Dataset({"UGRD": (("time"), [1, 2, 3])}, coords={"time": times})
    tgt = xr.Dataset({"ws": (("time"), [1, 2, 3])}, coords={"time": times})

    # Force has_xregrid to True and mock monet.pair to avoid esmpy dependency
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    mocker.patch("monet.util.combinetool.pair", return_value=paired_unsorted)

    # Call pair_data
    res = pair_data("test_sorting", "nearest", src, tgt, {})

    # Verify monotonic time index
    assert res.indexes["time"].is_monotonic_increasing
    np.testing.assert_allclose(res["val"].values, [2, 1, 3])  # sorted as 02:00, 05:00, 08:00


def test_pair_data_preserves_point_observation_times(mocker):
    """Test that pair_data does NOT drop duplicate time entries for point observation datasets."""
    # Create point observation dataset with duplicate times (different stations)
    times = pd.to_datetime(["2023-08-01 00:00", "2023-08-01 00:00", "2023-08-01 00:00"])
    # Point datasets do not have 'x' or 'y' dimensions
    point_ds = xr.Dataset({"val": (("time"), [1, 2, 3])}, coords={"time": times})

    # Setup dummy source dataset (gridded)
    src = xr.Dataset({"UGRD": (("y", "x"), [[1]])}, coords={"y": [0], "x": [0]})

    # Force has_xregrid to True and mock monet.pair to avoid esmpy dependency
    mocker.patch("monet.accessors.base.has_xregrid", True)
    mocker.patch("monet.util.resample.has_xregrid", True)
    mocker.patch("monet.util.combinetool.pair", return_value=point_ds)

    # Call pair_data
    res = pair_data("test_preservation", "nearest", src, point_ds, {})

    # Verify that duplicate times are preserved (length remains 3, not collapsed to 1)
    assert len(res["time"]) == 3


