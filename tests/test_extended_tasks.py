from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics


def test_compute_statistics_no_manual_fallbacks(mocker):
    """Test that manual weighted fallbacks are NOT used anymore."""
    import monet_stats

    # Mock monet_stats.weighted_spatial_mean to ensure it's NOT called by the orchestrator fallback
    mock_wsm = mocker.patch("monet_stats.weighted_spatial_mean")

    # Mock MB to NOT have 'weights' in its signature
    def mock_mb_no_weights(obs, mod, axis=None):
        return xr.DataArray(0.5)

    mocker.patch.object(monet_stats, "MB", mock_mb_no_weights, create=True)
    mock_mb_no_weights.__name__ = "MB"

    ds = xr.Dataset(
        {
            "obs": (("lat", "lon"), np.random.rand(10, 10)),
            "mod": (("lat", "lon"), np.random.rand(10, 10)),
            "w": (("lat", "lon"), np.random.rand(10, 10)),
        },
        coords={"lat": np.arange(10), "lon": np.arange(10)},
    )

    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w", "allow_unweighted_fallback": True}

    # Test MB - should call the mock_mb_no_weights directly (unweighted) and NOT the fallback wsm
    res = compute_statistics("test_mb", ["MB"], ds, kwargs)
    assert float(res["MB"]) == 0.5
    assert not mock_wsm.called


def test_compute_statistics_rejects_unknown_metrics_before_running_any(mocker):
    """Reject invalid metrics before invoking valid calculators."""
    import monet_stats

    known = mocker.Mock(return_value=xr.DataArray(1.0))
    mocker.patch.object(monet_stats, "MB", known, create=True)
    ds = xr.Dataset({"obs": ("x", [1.0]), "mod": ("x", [2.0])})
    with pytest.raises(ValueError, match="not_a_metric"):
        compute_statistics("invalid", ["MB", "not_a_metric"], ds, {"obs_var": "obs", "mod_var": "mod"})
    known.assert_not_called()


def test_generate_plot_dynamic_discovery(mocker):
    """Test generate_plot for dynamic discovery of monet-plots classes."""
    import monet_plots

    # Mock a new plot class that doesn't have a priority mapping
    mock_custom_plot = mocker.Mock()
    mocker.patch.object(monet_plots, "CustomPlot", mock_custom_plot, create=True)

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    # Test discovery of 'Custom' (should find CustomPlot)
    generate_plot("test_custom", "Custom", df, {"savename": "custom.png"}, track="A")
    monet_plots.CustomPlot.assert_called_once()
    assert isinstance(monet_plots.CustomPlot.call_args.args[0], xr.Dataset)


def test_plot_title_computes_requested_metric_lazily(mocker):
    """Compute requested xarray metrics as lazy scalar reductions."""
    import dask.array as da

    from mdt.tasks.plotting import _build_plot_title

    data = xr.Dataset(
        {
            "obs": ("x", da.from_array(np.array([1.0, 2.0]), chunks=1)),
            "mod": ("x", da.from_array(np.array([2.0, 4.0]), chunks=1)),
        }
    )
    title = _build_plot_title("comparison", "scatter", data, {"col1": "obs", "col2": "mod"}, None, "rmse")
    assert "RMSE=1.581" in title


def test_generic_plot_title_does_not_compute_a_default_stat(mocker):
    """Avoid calculating a metric solely to label a generic plot."""
    from mdt.tasks import plotting

    mocker.patch.object(plotting, "_compute_global_stat", side_effect=AssertionError("unexpected metric computation"))
    title = plotting._build_plot_title(
        "comparison",
        "scatter",
        pd.DataFrame({"obs": [1], "mod": [2]}),
        {"col1": "obs", "col2": "mod"},
        None,
        None,
    )
    assert title == "mod vs obs"


def test_save_statistics_reduces_lazy_xarray_results_to_scalar(tmp_path):
    """Write one scalar summary without materializing an array in the caller."""
    import dask.array as da

    from mdt.tasks.statistics import _save_statistics

    result = xr.DataArray(da.from_array(np.arange(100.0), chunks=10), dims="x")
    path = tmp_path / "summary.json"
    _save_statistics("summary", {"value": result}, str(path))
    assert "49.5" in path.read_text()


def test_spatial_bias_scatter_cbar_label_sent_to_constructor(mocker):
    """Ensure cbar_label is not forwarded to plot() where matplotlib scatter rejects it."""

    class DummySpatialBiasScatter:
        instances: ClassVar[list] = []

        def __init__(self, data, **kwargs):
            self.data = data
            self.constructor_kwargs = kwargs
            self.plot_kwargs = None
            DummySpatialBiasScatter.instances.append(self)

        def plot(self, **kwargs):
            self.plot_kwargs = kwargs

        def save(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    mocker.patch("mdt.tasks.plotting._find_plot_class", return_value=DummySpatialBiasScatter)

    df = pd.DataFrame(
        {
            "obs": [0.1, 0.2],
            "mod": [0.3, 0.1],
            "lat": [35.0, 36.0],
            "lon": [-97.0, -98.0],
        }
    )

    generate_plot(
        "bias_plot",
        "spatial_bias_scatter",
        df,
        {
            "col1": "obs",
            "col2": "mod",
            "cbar_label": "AOD Bias (Model - Obs)",
            "savename": "bias_plot.png",
        },
        track="A",
    )

    assert len(DummySpatialBiasScatter.instances) == 1
    plot_obj = DummySpatialBiasScatter.instances[0]
    assert plot_obj.constructor_kwargs.get("cbar_label") == "AOD Bias (Model - Obs)"
    assert "cbar_label" not in (plot_obj.plot_kwargs or {})
