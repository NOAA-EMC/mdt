"""Exercise pairing inputs through both Prefect execution paths."""

import logging
import sys
import types
from concurrent.futures import Future
from types import SimpleNamespace

import networkx as nx
import pytest
import xarray as xr

from mdt.engine import PrefectEngine


@pytest.fixture
def prefect_runtime(monkeypatch):
    """Run task bodies locally while preserving Prefect's future input contract."""

    def resolve(value):
        if isinstance(value, Future):
            return value.result()
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        return value

    class Task:
        def __init__(self, function):
            self.function = function

        def with_options(self, **kwargs):
            return self

        def submit(self, **kwargs):
            future = Future()
            try:
                future.set_result(self.function(**resolve(kwargs)))
            except Exception as exc:
                future.set_exception(exc)
            return future

    class Flow:
        def __init__(self, function):
            self.function = function

        def with_options(self, **kwargs):
            return self

        def __call__(self):
            return self.function()

    prefect = types.ModuleType("prefect")
    prefect.task = lambda **options: Task
    prefect.flow = lambda **options: Flow
    prefect.get_run_logger = lambda: logging.getLogger("test.prefect")
    policies = types.ModuleType("prefect.cache_policies")
    policies.NONE = None
    runners = types.ModuleType("prefect.task_runners")
    runners.ConcurrentTaskRunner = object
    monkeypatch.setitem(sys.modules, "prefect", prefect)
    monkeypatch.setitem(sys.modules, "prefect.cache_policies", policies)
    monkeypatch.setitem(sys.modules, "prefect.task_runners", runners)


def pairing_dag(mask=None, source_count=1):
    """Create loader dependencies and an optionally masked pairing node."""
    dag = nx.DiGraph()
    sources = [f"model{i}" for i in range(source_count)]
    for name in [*sources, "obs"]:
        dag.add_node(f"load_{name}", task_type="load_data", name=name, dataset_type="example", kwargs={})
    dag.add_node("pair_eval", task_type="pair_data", name="eval", method="nearest", kwargs={}, source_names=sources, target_name="obs", mask=mask)
    dag.add_edges_from((f"load_{name}", "pair_eval") for name in [*sources, "obs"])
    return dag


@pytest.mark.parametrize("mode", ["serial", "concurrent"])
@pytest.mark.parametrize("mask", [None, "land"])
def test_prefect_forwards_mask_to_pairing(prefect_runtime, mocker, tmp_path, mode, mask):
    """The configured mask reaches MONET and its labels reach regional statistics."""
    dag = pairing_dag(mask)
    dataset = xr.Dataset({"obs": ("x", [1.0, 2.0]), "mod": ("x", [2.0, 5.0])})
    mocker.patch("mdt.engine.load_data", return_value=dataset)
    mocker.patch("mdt.tasks.pairing._ensure_esmf_manager")
    mocker.patch("monet.util.combinetool.pair", return_value=dataset)
    query_mask = mocker.patch("monet.util.mask.query_mask", side_effect=lambda data, name: data.assign(land=("x", ["North", "South"])))
    if mask:
        dag.add_node(
            "stats_eval",
            task_type="compute_statistics",
            name="stats",
            metrics=["mb"],
            kwargs={"regions": ["North"], "savename": str(tmp_path / "stats_{region}.csv")},
        )
        dag.add_edge("pair_eval", "stats_eval")
        mocker.patch("mdt.tasks.statistics._find_metric", return_value=lambda obs, mod: (mod - obs).mean())
    config = SimpleNamespace(execution={"pairing_mode": mode, "clusters": {"compute": {"mode": "local", "workers": 1}}})
    results = PrefectEngine(dag, config).execute()
    if mask:
        query_mask.assert_called_once()
        assert "land" in results["pair_eval"]
        assert float(results["stats_eval"]["North"]["mb"]) == 1.0
    else:
        query_mask.assert_not_called()
        assert "land" not in results["pair_eval"]


def test_region_filter_computes_only_label_indexer():
    """Reloaded lazy labels can select a region without materializing model data."""
    from mdt.tasks.plotting import _filter_by_region

    data = xr.Dataset({"mod": ("x", [1.0, 3.0]), "land": ("x", ["North", "South"])}).chunk({"x": 1})
    selected = _filter_by_region(data, "land", "North")
    assert selected.sizes["x"] == 1
    assert selected.mod.chunks is not None
    assert float(selected.mod.compute().item()) == 1.0


def test_serial_multimodel_pairing_receives_datasets(prefect_runtime, mocker):
    """Resolve each loader future in the source mapping before inline pairing."""
    dag = pairing_dag(source_count=2)
    dataset = xr.Dataset({"value": ("x", [1.0, 3.0])})
    mocker.patch("mdt.engine.load_data", return_value=dataset)

    def pair(name, method, source_data, target_data, kwargs, mask=None):
        assert list(source_data) == ["model0", "model1"]
        assert all(isinstance(value, xr.Dataset) for value in source_data.values())
        assert isinstance(target_data, xr.Dataset)
        return xr.concat(list(source_data.values()), dim=xr.IndexVariable("model", list(source_data)))

    mocker.patch("mdt.engine.pair_data", side_effect=pair)
    config = SimpleNamespace(execution={"clusters": {"compute": {"mode": "local", "workers": 1}}})
    result = PrefectEngine(dag, config).execute()["pair_eval"]
    assert result.model.values.tolist() == ["model0", "model1"]


def test_serial_multimodel_failure_prevents_pairing(prefect_runtime, mocker):
    """An upstream failure must propagate before inline pairing is called."""
    dag = pairing_dag(source_count=2)
    dataset = xr.Dataset({"value": ("x", [1.0])})

    def load(name, *args):
        if name == "model1":
            raise RuntimeError("model1 unavailable")
        return dataset

    mocker.patch("mdt.engine.load_data", side_effect=load)
    pair = mocker.patch("mdt.engine.pair_data", return_value=dataset)
    config = SimpleNamespace(execution={"clusters": {"compute": {"mode": "local", "workers": 1}}})
    with pytest.raises(RuntimeError, match="model1 unavailable"):
        PrefectEngine(dag, config).execute()
    pair.assert_not_called()


def test_nested_future_resolution_preserves_containers_and_xarray():
    """Recursive inputs retain names, list/tuple structure, and lazy xarray identity."""
    from mdt.engine import _resolve_task_output

    data = xr.Dataset({"mod": ("x", [1.0])}).chunk({"x": 1})
    future = Future()
    future.set_result(data)
    nested = {"model0": [future, (future,)]}
    result = _resolve_task_output(nested)
    assert isinstance(result["model0"], list)
    assert isinstance(result["model0"][1], tuple)
    assert result["model0"][0] is data
    assert result["model0"][1][0] is data
    assert nested["model0"][0] is future


def test_single_model_pairing_consumes_mdt_options(mocker):
    """MDT's model dimension option must not leak into the MONET backend call."""
    from mdt.tasks.pairing import pair_data

    data = xr.Dataset({"mod": ("x", [1.0])})
    mocker.patch("mdt.tasks.pairing._ensure_esmf_manager")
    backend = mocker.patch("monet.util.combinetool.pair", return_value=data)
    kwargs = {"model_dim": "experiment", "interp_time": True}
    pair_data("single", "nearest", data, data, kwargs)
    assert backend.call_args.kwargs == {"method": "nearest", "interp_time": True}
    assert kwargs == {"model_dim": "experiment", "interp_time": True}


def test_prefect_accepts_legacy_single_source_node_metadata(prefect_runtime, mocker):
    """Resolve the legacy source_name field when source_names is absent."""
    dag = pairing_dag()
    del dag.nodes["pair_eval"]["source_names"]
    dag.nodes["pair_eval"]["source_name"] = "model0"
    dataset = xr.Dataset({"value": ("x", [1.0])})
    mocker.patch("mdt.engine.load_data", return_value=dataset)
    pairing = mocker.patch("mdt.engine.prefect_pair_data", return_value=dataset)
    config = SimpleNamespace(execution={"pairing_mode": "serial", "clusters": {"compute": {"mode": "local"}}})

    PrefectEngine(dag, config).execute()

    assert isinstance(pairing.call_args.kwargs["source_data"], xr.Dataset)


def test_serial_pairing_waits_for_previous_pair_result(prefect_runtime, mocker):
    """Resolve an earlier local pairing result before starting the next pair."""
    dag = pairing_dag()
    dag.add_node("load_model1", task_type="load_data", name="model1", dataset_type="example", kwargs={})
    dag.add_node(
        "pair_second",
        task_type="pair_data",
        name="second",
        method="nearest",
        kwargs={},
        source_names=["model1"],
        target_name="obs",
        model_dim="model",
        mask=None,
    )
    dag.add_edge("load_model1", "pair_second")
    dag.add_edge("load_obs", "pair_second")
    dataset = xr.Dataset({"value": ("x", [1.0])})
    mocker.patch("mdt.engine.load_data", return_value=dataset)
    pairing = mocker.patch("mdt.engine.prefect_pair_data", return_value=dataset)
    config = SimpleNamespace(execution={"pairing_mode": "serial", "clusters": {"compute": {"mode": "local"}}})

    PrefectEngine(dag, config).execute()

    assert pairing.call_count == 2


def test_multimodel_pairing_rejects_empty_sources():
    """Reject an empty mapping before invoking the pairing backend."""
    from mdt.tasks.pairing import pair_data

    with pytest.raises(ValueError, match="at least one source"):
        pair_data("eval", "nearest", {}, xr.Dataset(), {})


def test_duplicate_gridded_time_is_dropped(caplog):
    """Drop repeated timestamps from ordinary gridded data."""
    import pandas as pd

    from mdt.tasks.pairing import _drop_duplicate_time_entries

    data = xr.Dataset({"value": (("time", "x"), [[1], [2]])}, coords={"time": pd.to_datetime(["2025-08-01", "2025-08-01"]), "x": [0]})
    result = _drop_duplicate_time_entries(data, "model", "source")
    assert result.sizes["time"] == 1
    assert "keeping first occurrence" in caplog.text


def test_spatial_coordinate_aliases_are_added_and_redundant_2d_aliases_removed():
    """Create 1D aliases and remove redundant 2D coordinate aliases."""
    import numpy as np

    from mdt.tasks.pairing import _harmonize_spatial_coordinates

    one_dimensional = xr.Dataset(coords={"latitude": ("y", [40.0]), "longitude": ("x", [-100.0])})
    aliased = _harmonize_spatial_coordinates(one_dimensional)
    np.testing.assert_array_equal(aliased.lat.values, aliased.latitude.values)
    np.testing.assert_array_equal(aliased.lon.values, aliased.longitude.values)

    two_dimensional = xr.Dataset(
        coords={
            "lat": ("y", [40.0]),
            "lon": ("x", [-100.0]),
            "latitude": (("y", "x"), [[40.0]]),
            "longitude": (("y", "x"), [[-100.0]]),
        }
    )
    canonical = _harmonize_spatial_coordinates(two_dimensional)
    assert "latitude" not in canonical.coords
    assert "longitude" not in canonical.coords


def test_duplicate_valid_times_from_forecast_cycles_are_preserved(caplog):
    """Retain repeated valid times when they belong to different cycles."""
    import pandas as pd

    from mdt.tasks.pairing import _drop_duplicate_time_entries

    times = pd.to_datetime(["2025-08-01", "2025-08-01"])
    data = xr.Dataset(
        {"value": (("time", "x"), [[1], [2]])},
        coords={
            "time": times,
            "x": [0],
            "forecast_reference_time": ("time", pd.to_datetime(["2025-08-01", "2025-07-31"])),
            "lead_time": ("time", [0, 24]),
        },
    )
    with caplog.at_level(logging.INFO):
        result = _drop_duplicate_time_entries(data, "model", "source")
    assert result.sizes["time"] == 2
    assert "preserving all forecast records" in caplog.text


def test_multimodel_pairing_combines_results_and_collapses_shared_fields(mocker):
    """Combine model fields while storing identical observations once."""
    from mdt.tasks.pairing import pair_data

    first = xr.Dataset({"obs": ("x", [1.0]), "mod": ("x", [2.0])})
    second = xr.Dataset({"obs": ("x", [1.0]), "mod": ("x", [3.0])})
    mocker.patch("mdt.tasks.pairing._ensure_esmf_manager")
    mocker.patch("monet.util.combinetool.pair", side_effect=[first, second])
    result = pair_data("eval", "nearest", {"a": first, "b": second}, first, {"model_dim": "experiment"})
    assert result.attrs["mdt_model_dim"] == "experiment"
    assert result.mod.dims == ("experiment", "x")
    assert result.obs.dims == ("x",)


def test_multimodel_pairing_skips_fields_without_model_dimension(mocker):
    """Leave already shared fields untouched while inspecting model variables."""
    from mdt.tasks.pairing import pair_data

    source = xr.Dataset({"value": ("x", [1.0])})
    combined = xr.Dataset(
        {"mod": (("experiment", "x"), [[1.0], [2.0]]), "obs": ("x", [3.0])},
        coords={"experiment": ["a", "b"]},
    )
    mocker.patch("mdt.tasks.pairing._ensure_esmf_manager")
    mocker.patch("monet.util.combinetool.pair", return_value=source)
    mocker.patch("mdt.tasks.pairing.combine_paired_data", return_value=combined)

    result = pair_data("eval", "nearest", {"a": source, "b": source}, source, {"model_dim": "experiment"})

    assert result.obs.dims == ("x",)
    assert result.mod.dims == ("experiment", "x")
