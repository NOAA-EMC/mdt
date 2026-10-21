"""Regression tests for explicit statistical weighting policy."""

import numpy as np
import pytest
import xarray as xr

from mdt.tasks.statistics import compute_statistics


@pytest.fixture
def weighted_data():
    """Unequal errors and weights make accidental unweighted results observable."""
    return xr.Dataset({"obs": ("x", [0.0, 0.0]), "mod": ("x", [1.0, 3.0]), "w": ("x", [9.0, 1.0])})


def test_native_weighted_metric_preserves_requested_weights(weighted_data, mocker):
    """The backend receives weights and dimension without MDT control options."""

    def weighted_rmse(obs, mod, weights, dim):
        return np.sqrt(((mod - obs) ** 2).weighted(weights).mean(dim))

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=weighted_rmse)
    result = compute_statistics("weighted", ["rmse"], weighted_data, {"weights": "w", "dim": "x", "allow_unweighted_fallback": True})
    assert float(result["rmse"]) == pytest.approx(np.sqrt(1.8))
    assert "computed unweighted" not in result["rmse"].attrs["history"]


@pytest.mark.parametrize("allow_fallback", [False, True])
@pytest.mark.parametrize("array_input", [False, True])
def test_missing_named_weight_always_raises(weighted_data, mocker, allow_fallback, array_input):
    """Opting into unsupported-metric fallback never permits a missing variable."""
    metric = mocker.Mock()
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=metric)
    data = weighted_data.mod if array_input else weighted_data
    with pytest.raises(ValueError, match="Requested weight variable 'missing'"):
        compute_statistics("invalid", ["rmse"], data, {"weights": "missing", "allow_unweighted_fallback": allow_fallback})
    metric.assert_not_called()


def test_unsupported_weighted_metric_requires_explicit_fallback(weighted_data, mocker):
    """Unsupported weighting cannot quietly alter the scientific calculation."""

    def rmse(obs, mod):
        pytest.fail("Unsupported weighted metric must not execute")

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=rmse)
    with pytest.raises(ValueError, match="does not support requested weights"):
        compute_statistics("strict", ["rmse"], weighted_data, {"weights": "w"})


def test_explicit_fallback_records_unweighted_calculation(weighted_data, mocker, caplog):
    """Warnings and returned provenance reveal the explicit unweighted fallback."""

    def rmse(obs, mod):
        return np.sqrt(((mod - obs) ** 2).mean())

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=rmse)
    kwargs = {"weights": "w", "allow_unweighted_fallback": True}
    result = compute_statistics("fallback", ["rmse"], weighted_data, kwargs)["rmse"]
    assert float(result) == pytest.approx(np.sqrt(5.0))
    assert "computed unweighted" in result.attrs["history"]
    assert "allow_unweighted_fallback=True" in result.attrs["history"]
    assert "computed unweighted" in caplog.text
    assert kwargs == {"weights": "w", "allow_unweighted_fallback": True}


def test_unweighted_metric_is_unchanged(weighted_data, mocker):
    """No requested weights leaves the ordinary backend call unchanged."""

    def rmse(obs, mod):
        return np.sqrt(((mod - obs) ** 2).mean())

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=rmse)
    result = compute_statistics("unweighted", ["rmse"], weighted_data, {})["rmse"]
    assert float(result) == pytest.approx(np.sqrt(5.0))
    assert "computed unweighted" not in result.attrs["history"]


def test_dataarray_coordinate_weights(weighted_data, mocker):
    """Single-array metrics can use a named weight coordinate."""

    def weighted_mean(data, weights, dim):
        return data.weighted(weights).mean(dim)

    mocker.patch("mdt.tasks.statistics._find_metric", return_value=weighted_mean)
    data = weighted_data.mod.assign_coords(w=weighted_data.w)
    result = compute_statistics("array", ["mean"], data, {"weights": "w", "dim": "x"})
    assert float(result["mean"]) == pytest.approx(1.2)


def test_fallback_option_requires_boolean(weighted_data, mocker):
    """A quoted YAML boolean must not silently enable fallback."""
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=lambda obs, mod: mod.mean())
    with pytest.raises(ValueError, match="must be a boolean"):
        compute_statistics("invalid", ["mean"], weighted_data, {"weights": "w", "allow_unweighted_fallback": "false"})


def test_multimodel_statistics_returns_results_by_model(mocker):
    """Compute metrics independently for each model slice."""
    data = xr.Dataset(
        {"obs": (("model", "x"), [[0.0, 0.0], [0.0, 0.0]]), "mod": (("model", "x"), [[1.0, 3.0], [2.0, 4.0]])},
        coords={"model": ["a", "b"]},
    )
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=lambda obs, mod: (mod - obs).mean())
    result = compute_statistics("multi", ["mb"], data, {})
    assert float(result["a"]["mb"]) == pytest.approx(2.0)
    assert float(result["b"]["mb"]) == pytest.approx(3.0)


def test_multimodel_statistics_splits_savename(mocker, tmp_path):
    """Give each per-model statistics output its own file name."""
    data = xr.Dataset(
        {"obs": (("model", "x"), [[0.0], [0.0]]), "mod": (("model", "x"), [[1.0], [2.0]])},
        coords={"model": ["a", "b"]},
    )
    single = mocker.patch("mdt.tasks.statistics._compute_single", side_effect=[{"mb": 1}, {"mb": 2}])
    result = compute_statistics("multi", ["mb"], data, {"savename": str(tmp_path / "stats.csv")})
    assert result == {"a": {"mb": 1}, "b": {"mb": 2}}
    assert single.call_args_list[0].args[3]["savename"].endswith("stats_a.csv")
    assert single.call_args_list[1].args[3]["savename"].endswith("stats_b.csv")


def test_unpaired_dataarray_metric_receives_single_input(mocker):
    """Call single-array metrics without constructing a paired obs/model pair."""
    data = xr.DataArray([1.0, 3.0], dims="x", name="value")
    metric = mocker.Mock(side_effect=lambda values: values.mean())
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=metric)

    result = compute_statistics("array", ["mean"], data, {})

    assert float(result["mean"]) == pytest.approx(2.0)
    metric.assert_called_once()
