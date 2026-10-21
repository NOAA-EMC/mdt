"""Behavior checks for the typed xarray task boundaries."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mdt.tasks.data import _promote_time_coordinate, _time_role
from mdt.tasks.pairing import combine_paired_data
from mdt.tasks.plotting import _compute_xarray_global_stat, _format_time_range, generate_plot
from mdt.tasks.statistics import compute_statistics


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("metric", ["rmse", "mb", "mae", "corr"])
def test_global_statistics_preserve_xarray_reductions(lazy, metric):
    """Native xarray arithmetic retains finite-pair filtering for every metric."""
    data = xr.Dataset({"obs": ("x", [0.0, 2.0, np.nan, 4.0]), "mod": ("x", [1.0, 0.0, 99.0, 7.0])})
    if lazy:
        data = data.chunk({"x": 2})
    expected = {"rmse": np.sqrt(14.0 / 3), "mb": 2.0 / 3, "mae": 2.0, "corr": np.corrcoef([0, 2, 4], [1, 0, 7])[0, 1]}
    assert _compute_xarray_global_stat(data, "obs", "mod", metric) == pytest.approx(expected[metric])
    assert (data.mod.chunks is not None) == lazy


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (["NaT", "NaT"], None),
        (["NaT", "2023-08-01"], "2023-08-01"),
        (["2023-08-02", "NaT", "2023-08-01"], "2023-08-01 to 2023-08-02"),
    ],
)
def test_time_range_ignores_missing_values(dates, expected):
    """Filtering NaT before min/max preserves empty, single-date, and range output."""
    data = xr.Dataset(coords={"time": pd.to_datetime(dates, format="mixed")})
    assert _format_time_range(data) == expected


@pytest.mark.parametrize("kind", ["string", "datetime"])
def test_scalar_forecast_time_normalization(kind):
    """Scalar strings and datetime coordinates both become a one-record time axis."""
    value = "2023-08-01T06:00:00" if kind == "string" else np.datetime64("2023-08-01T06:00:00")
    data = xr.Dataset({"value": 1.0}, coords={"when": value})
    normalized = _promote_time_coordinate(data, "scalar", {"valid_time": "when"})
    assert normalized.sizes["time"] == 1
    assert pd.Timestamp(normalized.time.values[0]) == pd.Timestamp("2023-08-01T06:00:00")


def test_cf_coordinate_names_can_be_hashable():
    """CF discovery retains non-string xarray variable names and exclusions."""
    data = xr.Dataset({7: xr.DataArray(np.datetime64("2023-08-01"), attrs={"standard_name": "time"})})
    assert _time_role(data, None, "time") == 7
    assert _time_role(data, None, "time", exclude={7}) is None


@pytest.mark.parametrize("as_dataset", [False, True])
def test_combine_preserves_homogeneous_xarray_types(as_dataset):
    """Both supported concat overloads preserve values and model labels."""
    first = xr.DataArray([1.0, 2.0], dims="x", name="value")
    second = xr.DataArray([3.0, 4.0], dims="x", name="value")
    if as_dataset:
        first, second = first.to_dataset(), second.to_dataset()
    result = combine_paired_data({"first": first, "second": second})
    assert isinstance(result, xr.Dataset if as_dataset else xr.DataArray)
    assert result.model.values.tolist() == ["first", "second"]
    xr.testing.assert_equal(result.sel(model="first", drop=True), first)
    xr.testing.assert_equal(result.sel(model="second", drop=True), second)


def test_combine_rejects_mixed_xarray_types():
    """Mixed inputs raise a descriptive error rather than dropping either input."""
    array = xr.DataArray([1.0], dims="x", name="value")
    with pytest.raises(TypeError, match="same xarray type"):
        combine_paired_data({"array": array, "dataset": array.to_dataset()})


def test_combine_logs_and_reraises_xarray_concat_error(mocker, caplog):
    """Preserve backend concat failures and record a useful error message."""
    import logging

    mocker.patch("mdt.tasks.pairing.xr.concat", side_effect=RuntimeError("coordinate conflict"))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="coordinate conflict"):
        combine_paired_data({"first": xr.Dataset({"value": ("x", [1])})})
    assert "Failed to combine xarray objects" in caplog.text


@pytest.mark.parametrize("task", ["statistics", "plot"])
def test_regional_tasks_require_a_dataset(task, tmp_path):
    """Region labels must come from a Dataset before filtering can begin."""
    data = xr.DataArray([1.0, 2.0], dims="x")
    kwargs = {"regions": ["North"], "savename": str(tmp_path / "result_{region}.csv")}
    function, argument = (compute_statistics, ["mb"]) if task == "statistics" else (generate_plot, "timeseries")
    with pytest.raises(ValueError, match="Region filtering requires an xarray Dataset"):
        function("regional", argument, data, kwargs)


@pytest.mark.parametrize("color", [None, "blue"])
def test_timeseries_overlay_uses_xarray_line_plot(mocker, tmp_path, color):
    """Typed line plotting preserves overlay labels, values, and optional colors."""
    import matplotlib.pyplot as plt

    from mdt.tasks.plotting import _generate_static_plot

    class FirstSeries:
        def __init__(self, data, y, label, **kwargs):
            self.fig, self.ax = plt.subplots()
            self.ax.plot(data.time.values, data[y].values, label=label)

        def save(self, path):
            self.fig.savefig(path)

        def close(self):
            plt.close(self.fig)

    mocker.patch("mdt.tasks.plotting._find_plot_class", return_value=FirstSeries)
    data = xr.Dataset({"first": ("time", [1.0, 2.0]), "second": ("time", [3.0, 4.0])}, coords={"time": pd.date_range("2023-08-01", periods=2)})
    path = tmp_path / "series.png"
    result = _generate_static_plot(
        "series",
        "timeseries",
        data,
        {"columns": {"first": {"label": "First"}, "second": {"label": "Second", "color": color}}, "savename": str(path)},
    )
    assert len(result.ax.lines) == 2
    np.testing.assert_array_equal(result.ax.lines[1].get_ydata(), [3.0, 4.0])
    assert result.ax.lines[1].get_label() == "Second"
    if color:
        assert result.ax.lines[1].get_color() == color
    assert path.is_file()
