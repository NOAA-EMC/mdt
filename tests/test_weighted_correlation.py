import numpy as np
import pytest
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_explicit_unweighted_correlation_fallback_aero_protocol(mocker):
    """
    Aero Protocol Double-Check: Explicit unweighted Pearson fallback.

    Ensures Eager (NumPy) and Lazy (Dask) backends yield identical results.
    """
    # 1. Setup Synthetic Grid Data
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    # Generate reproducible random data
    rng = np.random.default_rng(42)
    obs_data = rng.standard_normal((len(lats), len(lons)))
    mod_data = obs_data * 0.8 + rng.standard_normal((len(lats), len(lons))) * 0.2

    # Eager Dataset (NumPy)
    ds_eager = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_data),
            "mod": (("lat", "lon"), mod_data),
        },
        coords={"lat": lats, "lon": lons},
    )

    # Add weights (cos lat)
    weights = np.cos(np.deg2rad(ds_eager.lat))
    # Broaden weights to match data shape for explicit passing or use coordinate
    ds_eager.coords["w"] = weights

    # 2. Lazy Dataset (Dask)
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 10})

    # 3. Execution
    metrics = ["pearsonr"]
    # We pass 'w' as the weight variable
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "allow_unweighted_fallback": True}

    # Note: We don't mock monet_stats here because we want to test the REAL integration
    # unless it's missing in the environment. Since I installed it, I'll use it.

    res_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)["pearsonr"]
    res_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)["pearsonr"]

    # 4. Assertions
    # Verify laziness (Aero Protocol Rule 1.2)
    assert hasattr(res_lazy.data, "dask") or type(res_lazy.data).__module__.startswith("cubed"), "Result should be lazy-backed for lazy input"

    # Verify identical results (Double-Check Rule)
    # Using small tolerance for float64
    xr.testing.assert_allclose(res_eager, res_lazy.compute(), atol=1e-15)

    # Verify provenance
    assert "Computed pearsonr" in res_eager.attrs["history"]
    assert "computed unweighted" in res_eager.attrs["history"]

    print("\n✅ Weighted Correlation Double-Check Passed: Eager == Lazy (Dask)")


if __name__ == "__main__":
    pytest.main([__file__])
