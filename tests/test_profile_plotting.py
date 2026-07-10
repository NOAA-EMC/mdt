import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch

from mdt.tasks.plotting import _generate_static_plot


class DummyProfilePlot:
    instances = []

    def __init__(self, data, var1, var2, label, **kwargs):
        self.data = data
        self.var1 = var1
        self.var2 = var2
        self.label = label
        
        class DummyAx:
            def __init__(self):
                self.yaxis = self
            def plot(self, x, y, **kwargs):
                x_arr = np.asanyarray(x)
                y_arr = np.asanyarray(y)
                if x_arr.shape[0] != y_arr.shape[0]:
                    raise ValueError(f"x and y must have same first dimension, but have shapes {x_arr.shape} and {y_arr.shape}")
            def set_ylabel(self, val):
                pass
            def set_yscale(self, val):
                pass
            def invert_yaxis(self):
                pass
            def legend(self):
                pass
            def set_major_formatter(self, formatter):
                pass
            def set_minor_formatter(self, formatter):
                pass
                
            def set_title(self, title):
                pass

        self.ax = DummyAx()
        self.fig = self
        DummyProfilePlot.instances.append(self)

    def plot(self, **kwargs):
        pass

    def save(self, *_args, **_kwargs):
        return None

    def close(self):
        return None

    def tight_layout(self):
        pass


@patch("mdt.tasks.plotting._find_plot_class")
def test_profile_plotting_interpolation_success(mock_find_plot_class):
    """Verifies that GFS profiles are successfully interpolated and plotted."""
    mock_find_plot_class.return_value = DummyProfilePlot
    DummyProfilePlot.instances.clear()

    # Create mock dataset
    # GFS has 41 isobaric levels, IGRA has 2 levels, valid_time has size 2
    valid_times = [np.datetime64("2023-08-01T00:00:00"), np.datetime64("2023-08-01T12:00:00")]
    isobaric_surfaces = np.linspace(10, 1000, 41)
    levels = np.arange(2)

    # Mock variables
    temperature_vals = np.random.rand(2, 41)  # GFS (valid_time, isobaric_surface)
    temp_vals = np.random.rand(2, 2)          # IGRA (valid_time, level)
    press_vals = np.array([[1000, 500], [950, 450]])  # IGRA press (valid_time, level)

    ds = xr.Dataset(
        data_vars={
            "temperature": (("valid_time", "isobaric_surface"), temperature_vals),
            "temp": (("valid_time", "level"), temp_vals),
            "press": (("valid_time", "level"), press_vals),
        },
        coords={
            "valid_time": valid_times,
            "isobaric_surface": isobaric_surfaces,
            "level": levels,
        }
    )

    kwargs = {
        "savename": "test_profile_comparison.png",
        "var2": "press",
        "columns": {
            "temp": {"label": "IGRA", "color": "black"},
            "temperature": {"label": "GFS", "color": "red"}
        }
    }

    # Run _generate_static_plot.
    # With our interpolation fix, GFS temperature should be interpolated to 2 levels to match press, and succeed!
    res = _generate_static_plot("test_profile", "profile", ds, kwargs)
    assert res is not None
    assert len(DummyProfilePlot.instances) == 1


@patch("mdt.tasks.plotting._find_plot_class")
def test_profile_plotting_defense_in_depth(mock_find_plot_class, caplog):
    """Verifies that non-interpolatable columns of mismatched shape are safely skipped."""
    mock_find_plot_class.return_value = DummyProfilePlot
    DummyProfilePlot.instances.clear()

    # Create mock dataset
    valid_times = [np.datetime64("2023-08-01T00:00:00"), np.datetime64("2023-08-01T12:00:00")]
    levels = np.arange(2)

    # Mock variables
    temp_vals = np.random.rand(2, 2)          # IGRA (valid_time, level)
    press_vals = np.array([[1000, 500], [950, 450]])  # IGRA press (valid_time, level)
    unmatched_vals = np.random.rand(41)       # Mismatching shape (41,) and not named temperature/TMP

    ds = xr.Dataset(
        data_vars={
            "temp": (("valid_time", "level"), temp_vals),
            "press": (("valid_time", "level"), press_vals),
            "unmatched_var": unmatched_vals,  # 1D variable on an untracked dimension
        },
        coords={
            "valid_time": valid_times,
            "level": levels,
        }
    )

    kwargs = {
        "savename": "test_profile_comparison.png",
        "var2": "press",
        "columns": {
            "temp": {"label": "IGRA", "color": "black"},
            "unmatched_var": {"label": "Unmatched", "color": "blue"}
        }
    }

    # Run _generate_static_plot.
    # It should succeed without throwing an exception, skipping unmatched_var gracefully
    res = _generate_static_plot("test_profile", "profile", ds, kwargs)
    assert res is not None
    assert len(DummyProfilePlot.instances) == 1
    
    # Assert warning was logged
    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("Skipping overlay plot for column 'unmatched_var'" in w for w in warnings)
