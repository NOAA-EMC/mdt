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
            "stats_eval", task_type="compute_statistics", name="stats", metrics=["mb"],
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
