import numpy as np
import xarray as xr

from mdt.tasks.statistics import _find_metric, compute_statistics


def test_find_metric_robustness(mocker):
    """Test _find_metric with various naming conventions and submodules."""

    class MockStats:
        def my_metric(self):
            pass

    class MockModule:
        def __init__(self):
            self.stats = MockStats()

        def top_metric(self):
            pass

        def lower_metric(self):
            pass

    mock_mod = MockModule()

    # Top level case insensitive
    assert _find_metric(mock_mod, "top_metric") == mock_mod.top_metric
    assert _find_metric(mock_mod, "LOWER_METRIC") == mock_mod.lower_metric

    # Submodule discovery
    assert _find_metric(mock_mod, "my_metric") == mock_mod.stats.my_metric

    # Not found
    assert _find_metric(mock_mod, "nonexistent") is None


def test_execute_metric_no_manual_fallback_rmse(mocker):
    """Verify that manual fallback is NOT used anymore for RMSE."""
    # 1. Setup Eager Data
    lat = np.array([10, 20])
    lon = np.array([100, 110])
    ds = xr.Dataset(
        {
            "obs": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]]),
            "mod": (("lat", "lon"), [[1.1, 2.1], [3.1, 4.1]]),
            "w": (("lat", "lon"), [[1.0, 1.0], [0.5, 0.5]]),
        },
        coords={"lat": lat, "lon": lon},
    )

    # 2. Mock a metric function that DOES NOT have 'weights' in signature
    def mock_rmse_no_weights(obs, mod, axis=None):
        # Should return unweighted result
        return xr.DataArray(((mod - obs) ** 2).mean())

    mock_rmse_no_weights.__name__ = "RMSE"

    # We need to mock monet_stats.weighted_spatial_mean to ensure it's NOT called

    mock_wsm = mocker.patch("monet_stats.weighted_spatial_mean")

    # Mock discovery to return our no-weights function
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=mock_rmse_no_weights)

    # 3. Execute
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "dim": ["lat", "lon"], "allow_unweighted_fallback": True}
    results = compute_statistics("test_fallback", ["RMSE"], ds, kwargs)

    # 4. Verify result (unweighted calculation)
    # diff = [0.1, 0.1, 0.1, 0.1], diff^2 = [0.01, 0.01, 0.01, 0.01]
    # mean = 0.01
    assert np.isclose(results["RMSE"].values, 0.01)
    assert not mock_wsm.called
    assert "Computed RMSE" in results["RMSE"].attrs["history"]
