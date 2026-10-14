"""Tests for durable ecFlow task artifact handoff."""

import numpy as np
import xarray as xr

from mdt.ecflow_io import load_artifact, save_artifact


def test_xarray_artifact_round_trip(tmp_path):
    """Write and reload a lazy Zarr backed Dataset."""
    dataset = xr.Dataset({"value": (("x",), np.arange(4.0))}, coords={"x": np.arange(4)})
    save_artifact(tmp_path, "load_model", dataset)
    loaded = load_artifact(tmp_path, "load_model")
    xr.testing.assert_equal(loaded.compute(), dataset)


def test_nested_statistics_artifact_round_trip(tmp_path):
    """Persist nested xarray metric mappings between tasks."""
    result = {
        "rmse": xr.DataArray(2.5),
        "summary": xr.Dataset({"value": ("x", [1, 2])}),
    }
    save_artifact(tmp_path, "stats_task", result)
    loaded = load_artifact(tmp_path, "stats_task")
    assert float(loaded["rmse"]) == 2.5
    xr.testing.assert_equal(loaded["summary"].compute(), result["summary"])


def test_none_artifact_round_trip(tmp_path):
    """Side-effect-only tasks can complete without storing non-data results."""
    save_artifact(tmp_path, "plot_task", None)
    assert load_artifact(tmp_path, "plot_task") is None


def test_artifact_rejects_pandas_values(tmp_path):
    """Pandas objects cannot cross the ecFlow handoff boundary."""
    import pandas as pd
    import pytest

    with pytest.raises(TypeError, match="Unsupported ecFlow task result type"):
        save_artifact(tmp_path, "invalid", pd.DataFrame({"value": [1]}))
