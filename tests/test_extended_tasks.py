import numpy as np
import pandas as pd
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

    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # Test MB - should call the mock_mb_no_weights directly (unweighted) and NOT the fallback wsm
    res = compute_statistics("test_mb", ["MB"], ds, kwargs)
    assert float(res["MB"]) == 0.5
    assert not mock_wsm.called


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


def test_spatial_bias_scatter_cbar_label_sent_to_constructor(mocker):
    """Ensure cbar_label is not forwarded to plot() where matplotlib scatter rejects it."""

    class DummySpatialBiasScatter:
        instances = []

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
