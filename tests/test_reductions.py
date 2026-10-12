import numpy as np
import pytest
import xarray as xr

from mdt.tasks.reductions import calculate_reduction, spatial_mean


@pytest.mark.parametrize(
    ("method", "dim", "message"),
    [
        ("median", "lat", "Unsupported reduction method"),
        ("mean", "missing", "Unknown reduction dimension"),
        ("mean", [], "non-empty list"),
    ],
)
def test_reduction_rejects_invalid_method_or_dimensions(method, dim, message):
    """Reject invalid reduction arguments before invoking xarray operations."""
    data = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))
    with pytest.raises(ValueError, match=message):
        calculate_reduction(data, method=method, dim=dim)


def test_spatial_mean_double_check():
    """
    Aero Protocol: Double-Check Test.

    Verifies that spatial_mean yields identical results for Eager (NumPy)
    and Lazy (Dask) backends and preserves laziness.
    """
    # 1. Setup Eager Data (NumPy)
    # A simple 10x10 grid with latitude from -90 to 90
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    data = np.random.rand(10, 10)

    da_eager = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="test_data", attrs={"history": "Initial data"})

    # 2. Setup Lazy Data (Dask)
    da_lazy = da_eager.chunk({"lat": 5, "lon": 5})

    # 3. Execute Eager (NumPy)
    res_eager = spatial_mean(da_eager)

    # 4. Execute Lazy (Dask)
    res_lazy = spatial_mean(da_lazy)

    # 5. Assertions
    # Verify laziness: The underlying data should be a Dask array
    assert hasattr(res_lazy.data, "dask") or type(res_lazy.data).__module__.startswith("cubed"), "Result should be lazy-backed for lazy input"

    # Verify identical results: NumPy-computed vs Dask-computed
    # We use allclose for floating point comparison
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values)

    # Verify provenance
    assert "via monet-stats" in res_eager.attrs["history"]
    assert "via monet-stats" in res_lazy.attrs["history"]

    print("\n✅ Aero Protocol Double-Check Passed: Eager == Lazy (Dask) for spatial_mean")


def test_spatial_mean_dataset():
    """Verify spatial_mean works on xr.Dataset objects."""
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    ds = xr.Dataset(
        {
            "var1": (("lat", "lon"), np.random.rand(5, 5)),
            "var2": (("lat", "lon"), np.random.rand(5, 5)),
        },
        coords={"lat": lat, "lon": lon},
    )

    res = spatial_mean(ds)
    assert "var1" in res.data_vars
    assert "var2" in res.data_vars
    assert res.var1.dims == ()
