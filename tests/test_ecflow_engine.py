"""Unit tests for EcFlowEngine initialisation and config defaults.

Validates: Requirements 3.6, 6.5, 8.1, 8.2, 8.3
"""

import os
import sys
import types
from unittest.mock import MagicMock

import networkx as nx
import pytest


@pytest.fixture
def _fake_ecflow(monkeypatch):
    """Inject a fake ``ecflow`` module so tests run without the real package."""
    fake = types.ModuleType("ecflow")

    # Each ecflow class returns a plain MagicMock (no spec restriction)
    # so that .add_family(), .add_task(), .add_variable(), etc. work.
    fake.Defs = lambda *a, **kw: MagicMock(name="Defs")
    fake.Suite = lambda *a, **kw: MagicMock(name="Suite")
    fake.Family = lambda *a, **kw: MagicMock(name="Family")
    fake.Task = lambda *a, **kw: MagicMock(name="Task")
    fake.Client = MagicMock
    monkeypatch.setitem(sys.modules, "ecflow", fake)
    return fake


@pytest.fixture
def simple_dag():
    """A minimal DAG with one load node."""
    g = nx.DiGraph()
    g.add_node("load_obs", task_type="load_data", name="obs", dataset_type="aeronet", kwargs={})
    return g


@pytest.fixture
def _make_config():
    """Factory that builds a mock config with a given execution dict."""

    def _factory(exec_dict=None):
        cfg = MagicMock()
        cfg.execution = exec_dict if exec_dict is not None else {}
        return cfg

    return _factory


class TestEcFlowEngineInit:
    """Tests for EcFlowEngine.__init__ configuration handling."""

    def test_defaults_when_no_ecflow_keys(self, _fake_ecflow, simple_dag, _make_config):
        """All ecFlow settings fall back to documented defaults."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        assert engine.host == "localhost"
        assert engine.port == 3141
        assert engine.suite_name == "mdt"
        assert engine.task_script_dir == "./ecflow_tasks/"
        assert engine.timeout == 86400

    def test_custom_config_values(self, _fake_ecflow, simple_dag, _make_config):
        """Explicit config values override the defaults."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config(
            {
                "ecflow_host": "ecflow.hpc.local",
                "ecflow_port": "4000",
                "suite_name": "my_suite",
                "task_script_dir": "/tmp/ecf_scripts/",
            }
        )
        engine = EcFlowEngine(simple_dag, cfg)

        assert engine.host == "ecflow.hpc.local"
        assert engine.port == 4000
        assert engine.suite_name == "my_suite"
        assert engine.task_script_dir == "/tmp/ecf_scripts/"

    def test_custom_wait_timeout(self, _fake_ecflow, simple_dag, _make_config):
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config({"ecflow_timeout": 42}))
        assert engine.timeout == 42

    def test_port_cast_to_int(self, _fake_ecflow, simple_dag, _make_config):
        """Port is always stored as int even when YAML provides a string."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"ecflow_port": "5555"})
        engine = EcFlowEngine(simple_dag, cfg)

        assert engine.port == 5555
        assert isinstance(engine.port, int)

    def test_dag_and_config_stored(self, _fake_ecflow, simple_dag, _make_config):
        """The DAG and config objects are stored on the instance."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config()
        engine = EcFlowEngine(simple_dag, cfg)

        assert engine.dag is simple_dag
        assert engine.config is cfg

    def test_ecflow_module_stored(self, _fake_ecflow, simple_dag, _make_config):
        """The lazily-imported ecflow module is stored for later use."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        assert engine.ecflow is _fake_ecflow


class TestEcFlowEngineLazyImport:
    """Tests for lazy-import behaviour of ecflow."""

    def test_importerror_when_ecflow_missing(self, monkeypatch, simple_dag, _make_config):
        """ImportError with install instructions when ecflow is not available."""
        # Ensure ecflow is NOT importable
        monkeypatch.delitem(sys.modules, "ecflow", raising=False)
        import builtins

        _real_import = builtins.__import__

        def _block_ecflow(name, *args, **kwargs):
            if name == "ecflow":
                raise ImportError("No module named 'ecflow'")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_ecflow)

        from mdt.ecflow_engine import EcFlowEngine

        with pytest.raises(ImportError, match="ecFlow Python bindings are not installed"):
            EcFlowEngine(simple_dag, _make_config())


class TestEcFlowEngineStubs:
    """Verify stub methods raise NotImplementedError."""

    def test_build_suite_returns_defs(self, _fake_ecflow, simple_dag, _make_config):
        """Verify build_suite returns an ecFlow Defs object."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        defs = engine.build_suite()
        # build_suite should return the Defs object (no longer a stub)
        assert defs is not None
        defs.add_suite.assert_called_once()

    def test_generate_task_wrappers_returns_paths(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Verify generate_task_wrappers creates files and returns their paths."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "ecf_scripts")})
        engine = EcFlowEngine(simple_dag, cfg)

        paths = engine.generate_task_wrappers()
        assert len(paths) == 1
        assert paths[0].endswith("load_obs.ecf")

    def test_load_and_start_calls_client_load_and_begin(self, _fake_ecflow, simple_dag, _make_config):
        """_load_and_start creates a client, loads defs, and begins the suite."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        # Replace Client with a proper mock so return_value works
        mock_client = MagicMock(name="Client-instance")
        engine.ecflow.Client = MagicMock(return_value=mock_client)

        fake_defs = MagicMock(name="Defs")
        engine._load_and_start(fake_defs)

        engine.ecflow.Client.assert_called_once_with("localhost", 3141)
        mock_client.load.assert_called_once_with(fake_defs)
        mock_client.begin_suite.assert_called_once_with("mdt")

    def test_execute_returns_completed_result(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """execute() waits for its configured suite waiter and returns completion."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "ecf_scripts")})
        engine = EcFlowEngine(simple_dag, cfg)

        # Replace Client with a proper mock so _load_and_start succeeds
        mock_client = MagicMock(name="Client-instance")
        engine.ecflow.Client = MagicMock(return_value=mock_client)
        engine._wait_for_suite = lambda: None

        result = engine.execute()

        assert result == {"suite": "mdt", "status": "complete"}


# ---------------------------------------------------------------------------
# Fixtures for build_suite() tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _tracking_ecflow(monkeypatch):
    """Fake ``ecflow`` module that records constructor arguments.

    Returns a dict with ``families``, ``tasks``, ``suites``, and ``defs``
    lists so tests can inspect what was created and how objects were wired.
    """
    fake = types.ModuleType("ecflow")
    created = {"defs": [], "suites": [], "families": {}, "tasks": {}}

    def _make_task(name):
        m = MagicMock(name=f"Task({name})")
        m._task_name = name
        m._variables = {}

        def _record_var(key, value):
            m._variables[key] = value

        m.add_variable = _record_var
        created["tasks"][name] = m
        return m

    def _make_family(name):
        m = MagicMock(name=f"Family({name})")
        m._family_name = name
        m.add_task = MagicMock(side_effect=_make_task)
        created["families"][name] = m
        return m

    def _make_suite(name):
        m = MagicMock(name=f"Suite({name})")
        m._suite_name = name
        m.add_family = MagicMock(side_effect=_make_family)
        created["suites"].append(m)
        return m

    def _make_defs(*a, **kw):
        m = MagicMock(name="Defs")
        m.add_suite = MagicMock(side_effect=_make_suite)
        created["defs"].append(m)
        return m

    fake.Defs = _make_defs
    fake.Suite = _make_suite
    fake.Family = _make_family
    fake.Task = _make_task
    fake.Client = MagicMock
    monkeypatch.setitem(sys.modules, "ecflow", fake)
    return created


@pytest.fixture
def multi_node_dag():
    """DAG with two load nodes feeding one pair node — tests triggers."""
    g = nx.DiGraph()
    g.add_node(
        "load_merra2",
        task_type="load_data",
        name="merra2",
        dataset_type="merra2",
        kwargs={},
        cluster="service",
    )
    g.add_node(
        "load_aeronet",
        task_type="load_data",
        name="aeronet",
        dataset_type="aeronet",
        kwargs={},
        cluster="service",
    )
    g.add_node(
        "pair_merra2_aeronet",
        task_type="pair_data",
        name="merra2_aeronet",
        source_name="merra2",
        target_name="aeronet",
        method="interpolate",
        kwargs={"radius": 50},
        cluster="compute",
    )
    g.add_edge("load_merra2", "pair_merra2_aeronet")
    g.add_edge("load_aeronet", "pair_merra2_aeronet")
    return g


@pytest.fixture
def full_pipeline_dag():
    """DAG covering all five task types for family-creation tests."""
    g = nx.DiGraph()
    g.add_node("load_obs", task_type="load_data", name="obs", dataset_type="aeronet", kwargs={})
    g.add_node("load_model", task_type="load_data", name="model", dataset_type="merra2", kwargs={})
    g.add_node("pair_obs_model", task_type="pair_data", name="obs_model", source_name="obs", target_name="model", kwargs={})
    g.add_node("combine_all", task_type="combine_paired_data", name="all", sources=["obs_model"], kwargs={})
    g.add_node("stats_bias", task_type="compute_statistics", name="bias", kwargs={}, metrics=["mb", "rmse"])
    g.add_node("plot_spatial", task_type="generate_plot", name="spatial", kwargs={}, plot_type="spatial")

    g.add_edge("load_obs", "pair_obs_model")
    g.add_edge("load_model", "pair_obs_model")
    g.add_edge("pair_obs_model", "combine_all")
    g.add_edge("combine_all", "stats_bias")
    g.add_edge("stats_bias", "plot_spatial")
    return g


# ---------------------------------------------------------------------------
# Tests for build_suite()
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
# ---------------------------------------------------------------------------


class TestBuildSuiteFamilyCreation:
    """Requirement 4.1 — one ecFlow family per task type."""

    def test_all_task_families_created(self, _tracking_ecflow, full_pipeline_dag, _make_config):
        """build_suite creates families for every supported DAG task type."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()

        expected = {"load", "pair", "combine", "statistics", "plot", "reduction", "save"}
        assert set(_tracking_ecflow["families"].keys()) == expected

    def test_families_added_to_suite(self, _tracking_ecflow, full_pipeline_dag, _make_config):
        """Each family is added to the suite via add_family."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()

        suite = _tracking_ecflow["suites"][0]
        assert suite.add_family.call_count == 7

    def test_families_created_even_without_matching_nodes(self, _tracking_ecflow, simple_dag, _make_config):
        """All supported task families are created even if the DAG only has load nodes."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())
        engine.build_suite()

        expected = {"load", "pair", "combine", "statistics", "plot", "reduction", "save"}
        assert set(_tracking_ecflow["families"].keys()) == expected


class TestBuildSuiteTaskCreation:
    """Requirement 4.2 — one ecFlow task per DAG node."""

    def test_task_per_dag_node(self, _tracking_ecflow, full_pipeline_dag, _make_config):
        """One ecFlow task is created for each DAG node."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()

        assert set(_tracking_ecflow["tasks"].keys()) == set(full_pipeline_dag.nodes)

    def test_tasks_placed_in_correct_families(self, _tracking_ecflow, full_pipeline_dag, _make_config):
        """Each task is added to the family matching its task_type."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()

        families = _tracking_ecflow["families"]
        # load family should have add_task called for load_obs and load_model
        load_calls = [c.args[0] for c in families["load"].add_task.call_args_list]
        assert sorted(load_calls) == ["load_model", "load_obs"]

        # pair family should have pair_obs_model
        pair_calls = [c.args[0] for c in families["pair"].add_task.call_args_list]
        assert pair_calls == ["pair_obs_model"]


class TestBuildSuiteTriggers:
    """Requirement 4.3 — trigger expressions from DAG edges."""

    def test_pair_node_trigger_from_two_loads(self, _tracking_ecflow, multi_node_dag, _make_config):
        """Pair node gets a trigger referencing both load predecessors."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(multi_node_dag, _make_config())
        engine.build_suite()

        pair_task = _tracking_ecflow["tasks"]["pair_merra2_aeronet"]
        pair_task.add_trigger.assert_called_once()
        trigger = pair_task.add_trigger.call_args.args[0]

        # Both predecessors should appear (sorted alphabetically)
        assert "/mdt/load/load_aeronet == complete" in trigger
        assert "/mdt/load/load_merra2 == complete" in trigger
        assert " and " in trigger

    def test_root_nodes_have_no_trigger(self, _tracking_ecflow, multi_node_dag, _make_config):
        """Load nodes (no predecessors) should not have add_trigger called."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(multi_node_dag, _make_config())
        engine.build_suite()

        load_m = _tracking_ecflow["tasks"]["load_merra2"]
        load_a = _tracking_ecflow["tasks"]["load_aeronet"]
        load_m.add_trigger.assert_not_called()
        load_a.add_trigger.assert_not_called()

    def test_chain_trigger_expression(self, _tracking_ecflow, full_pipeline_dag, _make_config):
        """In a linear chain, each node triggers on its single predecessor."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()

        stats_task = _tracking_ecflow["tasks"]["stats_bias"]
        stats_task.add_trigger.assert_called_once()
        trigger = stats_task.add_trigger.call_args.args[0]
        assert trigger == "/mdt/combine/combine_all == complete"

        plot_task = _tracking_ecflow["tasks"]["plot_spatial"]
        plot_task.add_trigger.assert_called_once()
        trigger = plot_task.add_trigger.call_args.args[0]
        assert trigger == "/mdt/statistics/stats_bias == complete"


class TestBuildSuiteVariables:
    """Requirement 4.4 — ecFlow variables on each task node."""

    def test_load_node_variables(self, _tracking_ecflow, multi_node_dag, _make_config):
        """Load node carries TASK_TYPE, TASK_NAME, DATASET_TYPE, etc."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(multi_node_dag, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["load_merra2"]._variables
        assert v["TASK_TYPE"] == "load_data"
        assert v["TASK_NAME"] == "merra2"
        assert v["DATASET_TYPE"] == "merra2"
        assert v["TASK_KWARGS"] == "{}"
        assert v["TASK_KWARGS_B64"]
        assert v["TASK_NAME_B64"]

    def test_wrapper_script_uses_encoded_user_values(self):
        """Quote-sensitive configuration is substituted through base64 tokens."""
        from mdt.ecflow_engine import _build_wrapper_script

        script = _build_wrapper_script("load_test", "/usr/bin/python")
        assert "decode('%TASK_NAME_B64%')" in script
        assert "json.loads(decode('%TASK_KWARGS_B64%'))" in script
        assert "decode('%URL_B64%')" in script

    def test_pair_node_kwargs_serialized(self, _tracking_ecflow, multi_node_dag, _make_config):
        """Pair node kwargs are JSON-serialized."""
        import json

        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(multi_node_dag, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["pair_merra2_aeronet"]._variables
        assert json.loads(v["TASK_KWARGS"]) == {"radius": 50}
        assert json.loads(__import__("base64").b64decode(v["TASK_KWARGS_B64"])) == {"radius": 50}
        assert json.loads(__import__("base64").b64decode(v["TASK_KWARGS_B64"])) == {"radius": 50}

    def test_metrics_and_plot_type_variables(self, _tracking_ecflow, full_pipeline_dag, _make_config):
        """Statistics node has METRICS; plot node has PLOT_TYPE."""
        import json

        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()

        stats_v = _tracking_ecflow["tasks"]["stats_bias"]._variables
        assert json.loads(stats_v["METRICS"]) == ["mb", "rmse"]

        plot_v = _tracking_ecflow["tasks"]["plot_spatial"]._variables
        assert plot_v["PLOT_TYPE"] == "spatial"

    def test_missing_optional_fields_default_to_empty(self, _tracking_ecflow, _make_config):
        """Nodes without dataset_type, metrics, plot_type get empty defaults."""
        from mdt.ecflow_engine import EcFlowEngine

        g = nx.DiGraph()
        g.add_node("load_source", task_type="load_data", name="source", dataset_type="fake", kwargs={})
        g.add_node("load_target", task_type="load_data", name="target", dataset_type="fake", kwargs={})
        g.add_node("pair_x", task_type="pair_data", name="x", source_name="source", target_name="target", kwargs={})
        g.add_edge("load_source", "pair_x")
        g.add_edge("load_target", "pair_x")
        engine = EcFlowEngine(g, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["pair_x"]._variables
        assert v["DATASET_TYPE"] == ""
        assert v["METRICS"] == "[]"
        assert v["PLOT_TYPE"] == ""


class TestBuildSuiteCluster:
    """Requirement 4.5 — cluster attribute mapping."""

    def test_cluster_set_from_dag_node(self, _tracking_ecflow, multi_node_dag, _make_config):
        """CLUSTER variable reflects the DAG node's cluster attribute."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(multi_node_dag, _make_config())
        engine.build_suite()

        v_load = _tracking_ecflow["tasks"]["load_merra2"]._variables
        assert v_load["CLUSTER"] == "service"

        v_pair = _tracking_ecflow["tasks"]["pair_merra2_aeronet"]._variables
        assert v_pair["CLUSTER"] == "compute"

    def test_cluster_defaults_to_empty_when_absent(self, _tracking_ecflow, _make_config):
        """Nodes without a cluster attribute get an empty CLUSTER variable."""
        from mdt.ecflow_engine import EcFlowEngine

        g = nx.DiGraph()
        g.add_node("load_x", task_type="load_data", name="x", dataset_type="obs", kwargs={})
        engine = EcFlowEngine(g, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["load_x"]._variables
        assert v["CLUSTER"] == ""


# ---------------------------------------------------------------------------
# Tests for generate_task_wrappers()
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 8.4
# ---------------------------------------------------------------------------


class TestGenerateTaskWrappersFileCreation:
    """Requirement 5.1, 5.5, 8.4 — one .ecf file per DAG node, output dir creation."""

    def test_one_script_per_dag_node_simple(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """One .ecf file is generated for a single-node DAG."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)

        paths = engine.generate_task_wrappers()

        assert len(paths) == 1
        assert all(os.path.isfile(p) for p in paths)

    def test_one_script_per_dag_node_multi(self, _fake_ecflow, multi_node_dag, _make_config, tmp_path):
        """One .ecf file is generated for each node in a multi-node DAG."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(multi_node_dag, cfg)

        paths = engine.generate_task_wrappers()

        assert len(paths) == len(multi_node_dag.nodes)

    def test_one_script_per_dag_node_full_pipeline(self, _fake_ecflow, full_pipeline_dag, _make_config, tmp_path):
        """One .ecf file per node across all five task types."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(full_pipeline_dag, cfg)

        paths = engine.generate_task_wrappers()

        assert len(paths) == len(full_pipeline_dag.nodes)
        filenames = {os.path.basename(p) for p in paths}
        expected = {f"{node}.ecf" for node in full_pipeline_dag.nodes}
        assert filenames == expected

    def test_output_directory_created(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """The task_script_dir is created if it doesn't already exist."""
        from mdt.ecflow_engine import EcFlowEngine

        target_dir = tmp_path / "nested" / "ecf_scripts"
        assert not target_dir.exists()

        cfg = _make_config({"task_script_dir": str(target_dir)})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        assert target_dir.is_dir()

    def test_returned_paths_are_correct(self, _fake_ecflow, multi_node_dag, _make_config, tmp_path):
        """Returned list contains the correct absolute file paths."""
        from mdt.ecflow_engine import EcFlowEngine

        script_dir = tmp_path / "scripts"
        cfg = _make_config({"task_script_dir": str(script_dir)})
        engine = EcFlowEngine(multi_node_dag, cfg)

        paths = engine.generate_task_wrappers()

        for p in paths:
            assert p.startswith(str(script_dir))
            assert p.endswith(".ecf")
            assert os.path.isfile(p)


class TestGenerateTaskWrappersScriptContent:
    """Requirement 5.2, 5.3, 5.4 — script content: init/complete/abort, dispatch, %VAR% tokens."""

    def _read_script(self, engine, node_id):
        """Helper to read the generated script for a given node."""
        from pathlib import Path

        matches = list(Path(engine.task_script_dir).rglob(f"{node_id}.ecf"))
        assert len(matches) == 1
        with matches[0].open() as fh:
            return fh.read()

    def test_script_contains_init_call(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Each wrapper calls client.child_init() at the start."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "client.child_init(" in content

    def test_script_contains_complete_call(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Each wrapper calls client.child_complete() on success."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "client.child_complete()" in content

    def test_script_contains_abort_call(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Each wrapper calls client.child_abort() on exception."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "client.child_abort(" in content

    def test_script_contains_ecflow_var_tokens(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Scripts use ecFlow %VAR% substitution tokens for task parameters."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "%TASK_TYPE_B64%" in content
        assert "%TASK_NAME_B64%" in content
        assert "%TASK_KWARGS_B64%" in content

    def test_script_contains_dispatch_for_load_data(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Script includes the dispatch import for load_data task type."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "from mdt.tasks.data import load_data" in content

    def test_script_contains_all_dispatch_blocks(self, _fake_ecflow, full_pipeline_dag, _make_config, tmp_path):
        """All five task-type dispatch blocks appear in every generated script."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(full_pipeline_dag, cfg)
        engine.generate_task_wrappers()

        # Every script has the same if/elif dispatch chain, so pick any node
        content = self._read_script(engine, "load_obs")
        assert "from mdt.tasks.data import load_data" in content
        assert "from mdt.tasks.pairing import pair_data" in content
        assert "from mdt.tasks.pairing import combine_paired_data" in content
        assert "from mdt.tasks.statistics import compute_statistics" in content
        assert "from mdt.tasks.plotting import generate_plot" in content

    def test_wrapper_has_durable_input_and_output_handoff(self, _fake_ecflow, full_pipeline_dag, _make_config):
        """Each wrapper reads upstream artifacts and writes its own result."""
        import re

        from mdt.ecflow_engine import EcFlowEngine, _build_wrapper_script

        engine = EcFlowEngine(full_pipeline_dag, _make_config())
        engine.build_suite()
        script = _build_wrapper_script("stats_bias", engine.python_executable)
        assert "load_artifact(data_dir, upstream_id)" in script
        assert "save_artifact(data_dir, node_id, result)" in script
        assert "source_inputs = load_artifact(data_dir, source_inputs)" in script
        assert "input_data=inputs['input']" in script
        compile(re.sub(r"%[A-Z0-9_]+%", "safe", script), "wrapper.ecf", "exec")
        assert engine._input_node_ids("pair_obs_model", full_pipeline_dag.nodes["pair_obs_model"]) == {
            "source": "load_obs",
            "target": "load_model",
        }

    def test_script_has_shebang_line(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Each wrapper starts with a Python shebang."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        first_line = content.splitlines()[0]
        assert first_line.startswith("#!")
        assert "python" in first_line.lower()

    def test_script_exits_with_code_1_on_error(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """Script calls sys.exit(1) after abort on exception."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "sys.exit(1)" in content


# ---------------------------------------------------------------------------
# Tests for EcFlowEngine.execute() flow
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------


class TestExecuteFlow:
    """Requirement 6.1–6.4 — execute() orchestration and server interaction."""

    def test_execute_calls_build_generate_load_in_order(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """execute() calls build_suite, generate_task_wrappers, _load_and_start in sequence."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(simple_dag, cfg)

        call_order = []

        orig_build = engine.build_suite
        orig_gen = engine.generate_task_wrappers

        def tracked_build():
            call_order.append("build_suite")
            return orig_build()

        def tracked_gen():
            call_order.append("generate_task_wrappers")
            return orig_gen()

        def tracked_load(defs):
            call_order.append("_load_and_start")
            # Skip actual client interaction
            return None

        engine.build_suite = tracked_build
        engine.generate_task_wrappers = tracked_gen
        engine._load_and_start = tracked_load
        engine._wait_for_suite = lambda: None

        engine.execute()

        assert call_order == ["build_suite", "generate_task_wrappers", "_load_and_start"]

    def test_load_and_start_uses_configured_host_port(self, _fake_ecflow, simple_dag, _make_config):
        """_load_and_start connects to the ecFlow server with custom host and port."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config(
            {
                "ecflow_host": "ecflow.hpc.local",
                "ecflow_port": "4500",
            }
        )
        engine = EcFlowEngine(simple_dag, cfg)

        mock_client = MagicMock(name="Client-instance")
        engine.ecflow.Client = MagicMock(return_value=mock_client)

        fake_defs = MagicMock(name="Defs")
        engine._load_and_start(fake_defs)

        engine.ecflow.Client.assert_called_once_with("ecflow.hpc.local", 4500)

    def test_load_and_start_loads_defs_then_begins_suite(self, _fake_ecflow, simple_dag, _make_config):
        """_load_and_start calls client.load(defs) then client.begin_suite(suite_name)."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        mock_client = MagicMock(name="Client-instance")
        engine.ecflow.Client = MagicMock(return_value=mock_client)

        fake_defs = MagicMock(name="Defs")
        engine._load_and_start(fake_defs)

        # Verify load is called before begin_suite via call order
        mock_client.load.assert_called_once_with(fake_defs)
        mock_client.begin_suite.assert_called_once_with("mdt")

        # Verify ordering: load was called before begin_suite
        load_call_idx = None
        begin_call_idx = None
        for i, c in enumerate(mock_client.method_calls):
            if c[0] == "load":
                load_call_idx = i
            elif c[0] == "begin_suite":
                begin_call_idx = i
        assert load_call_idx is not None
        assert begin_call_idx is not None
        assert load_call_idx < begin_call_idx

    def test_connection_failure_raises_runtime_error_with_host_port(self, _fake_ecflow, simple_dag, _make_config):
        """Connection failure raises RuntimeError mentioning the host and port."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config(
            {
                "ecflow_host": "bad-host.example.com",
                "ecflow_port": "9999",
            }
        )
        engine = EcFlowEngine(simple_dag, cfg)

        # Make Client constructor raise a connection error
        engine.ecflow.Client = MagicMock(side_effect=ConnectionRefusedError("Connection refused"))

        with pytest.raises(RuntimeError, match="bad-host.example.com"):
            engine._load_and_start(MagicMock())

    def test_connection_failure_includes_port_in_message(self, _fake_ecflow, simple_dag, _make_config):
        """RuntimeError message includes the port number."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config(
            {
                "ecflow_host": "myhost",
                "ecflow_port": "7777",
            }
        )
        engine = EcFlowEngine(simple_dag, cfg)

        engine.ecflow.Client = MagicMock(side_effect=OSError("Cannot connect"))

        with pytest.raises(RuntimeError, match="7777"):
            engine._load_and_start(MagicMock())

    def test_connection_failure_wraps_original_exception(self, _fake_ecflow, simple_dag, _make_config):
        """RuntimeError chains the original exception as __cause__."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        original = ConnectionRefusedError("Connection refused")
        engine.ecflow.Client = MagicMock(side_effect=original)

        with pytest.raises(RuntimeError) as exc_info:
            engine._load_and_start(MagicMock())

        assert exc_info.value.__cause__ is original

    def test_execute_returns_suite_name_and_completed_status(self, _fake_ecflow, simple_dag, _make_config, tmp_path):
        """execute() waits for terminal status and returns a completed result."""
        from mdt.ecflow_engine import EcFlowEngine

        cfg = _make_config(
            {
                "task_script_dir": str(tmp_path / "scripts"),
                "suite_name": "my_workflow",
            }
        )
        engine = EcFlowEngine(simple_dag, cfg)

        mock_client = MagicMock(name="Client-instance")
        engine.ecflow.Client = MagicMock(return_value=mock_client)
        engine._wait_for_suite = lambda: None

        result = engine.execute()

        assert result == {"suite": "my_workflow", "status": "complete"}
        mock_client.close.assert_called_once()

    def test_empty_workflow_completes_without_submitting_suite(self, _fake_ecflow, _make_config):
        """An empty DAG returns success without starting a suite that cannot complete."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(nx.DiGraph(), _make_config())
        engine.ecflow.Client = MagicMock()

        assert engine.execute() == {"suite": "mdt", "status": "complete"}
        engine.ecflow.Client.assert_not_called()

    def test_wait_for_suite_polls_until_complete(self, _fake_ecflow, simple_dag, _make_config):
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())
        suite = MagicMock()
        suite.get_state.side_effect = ["active", "complete"]
        defs = MagicMock()
        defs.find_suite.return_value = suite
        client = MagicMock()
        client.get_defs.return_value = defs
        engine.client = client
        engine.poll_interval = 0

        engine._wait_for_suite()

        assert client.sync_local.call_count == 2

    def test_wait_for_suite_reports_aborted_task(self, _fake_ecflow, simple_dag, _make_config):
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())
        failed_task = MagicMock()
        failed_task.get_state.return_value = "aborted"
        failed_task.get_abs_node_path.return_value = "/mdt/load/load_obs"
        suite = MagicMock()
        suite.get_state.return_value = "aborted"
        suite.get_all_nodes.return_value = [failed_task]
        defs = MagicMock()
        defs.find_suite.return_value = suite
        client = MagicMock()
        client.get_defs.return_value = defs
        engine.client = client

        with pytest.raises(RuntimeError, match="/mdt/load/load_obs"):
            engine._wait_for_suite()

    def test_load_failure_raises_runtime_error(self, _fake_ecflow, simple_dag, _make_config):
        """If client.load() fails, RuntimeError is raised with host/port."""
        from mdt.ecflow_engine import EcFlowEngine

        engine = EcFlowEngine(simple_dag, _make_config())

        mock_client = MagicMock(name="Client-instance")
        mock_client.load.side_effect = RuntimeError("load failed")
        engine.ecflow.Client = MagicMock(return_value=mock_client)

        with pytest.raises(RuntimeError, match="localhost"):
            engine._load_and_start(MagicMock())


@pytest.fixture
def run_generated_wrapper(_tracking_ecflow, _make_config, tmp_path):
    """Execute generated task code with ecFlow variable substitution and real artifacts."""
    import re
    from pathlib import Path

    from mdt.ecflow_engine import EcFlowEngine

    sys.modules["ecflow"].Client = MagicMock()
    engines = {}

    def run(dag, node_id):
        if id(dag) not in engines:
            engines[id(dag)] = EcFlowEngine(
                dag, _make_config({"task_script_dir": str(tmp_path / "scripts"), "ecflow_data_dir": str(tmp_path / "data")})
            )
            # All wrappers in this simulated run share one artifact namespace.
            engines[id(dag)].data_dir = tmp_path / "data"
        engine = engines[id(dag)]
        engine.build_suite()
        paths = engine.generate_task_wrappers()
        script_path = next(Path(path) for path in paths if Path(path).stem == node_id)
        variables = dict(_tracking_ecflow["tasks"][node_id]._variables)
        variables.update(ECF_NAME=f"/mdt/{node_id}", ECF_PASS="test", ECF_TRYNO="1")  # noqa: S106 — simulated ecFlow client
        script = re.sub(r"%([A-Z0-9_]+)%", lambda match: str(variables[match[1]]), script_path.read_text())
        client = sys.modules["ecflow"].Client.return_value
        client.reset_mock()
        exec(compile(script, str(script_path), "exec"), {})
        client.child_complete.assert_called_once()
        client.child_abort.assert_not_called()

    return run


@pytest.mark.parametrize("source_count", [1, 2])
@pytest.mark.parametrize("mask", [None, "land"])
def test_generated_pipeline_loads_pairing_artifacts(run_generated_wrapper, tmp_path, mocker, source_count, mask):
    """Single and multi-model wrappers pass datasets through pairing into statistics."""
    import xarray as xr

    from mdt.ecflow_io import load_artifact

    dag = nx.DiGraph()
    sources = [f"model{i}" for i in range(source_count)]
    for name in [*sources, "obs"]:
        dag.add_node(f"load_{name}", task_type="load_data", name=name, dataset_type="example", kwargs={})
    dag.add_node(
        "pair_eval", task_type="pair_data", name="eval", source_names=sources, target_name="obs", model_dim="experiment", mask=mask, kwargs={}
    )
    dag.add_edges_from((f"load_{name}", "pair_eval") for name in [*sources, "obs"])
    stats_kwargs = {"regions": ["North"], "savename": str(tmp_path / "stats_{region}.csv")} if mask else {}
    dag.add_node("stats_eval", task_type="compute_statistics", name="summary", metrics=["mb"], kwargs=stats_kwargs)
    dag.add_edge("pair_eval", "stats_eval")

    def load(name, **kwargs):
        return xr.Dataset({"value": ("x", [1.0, 3.0] if name == "obs" else [2.0, 5.0])})

    def pair(source_data, target_data, kwargs, **other):
        assert kwargs["model_dim"] == "experiment"
        assert other["mask"] == mask
        assert isinstance(target_data, xr.Dataset)
        if source_count == 1:
            assert isinstance(source_data, xr.Dataset)
            model = source_data.value
        else:
            assert list(source_data) == sources
            assert all(isinstance(value, xr.Dataset) for value in source_data.values())
            model = xr.concat([value.value for value in source_data.values()], dim="experiment")
        paired = xr.Dataset({"obs": target_data.value, "mod": model})
        return paired.assign(land=("x", ["North", "South"])) if mask else paired

    mocker.patch("mdt.tasks.data.load_data", side_effect=load)
    mocker.patch("mdt.tasks.pairing.pair_data", side_effect=pair)
    mocker.patch("mdt.tasks.statistics._find_metric", return_value=lambda obs, mod: (mod - obs).mean())
    for node_id in nx.topological_sort(dag):
        run_generated_wrapper(dag, node_id)
    result = load_artifact(tmp_path / "data", "stats_eval")
    assert float(result["North"]["mb"] if mask else result["mb"]) == (1.0 if mask else 1.5)


def test_generated_pairing_aborts_for_missing_source(run_generated_wrapper, multi_node_dag, tmp_path, mocker):
    """A missing source artifact aborts the child task with its upstream ID."""
    import xarray as xr

    from mdt.ecflow_io import save_artifact

    save_artifact(tmp_path / "data", "load_aeronet", xr.Dataset({"obs": ("x", [1.0])}))
    pair = mocker.patch("mdt.tasks.pairing.pair_data")
    with pytest.raises(SystemExit) as exc:
        run_generated_wrapper(multi_node_dag, "pair_merra2_aeronet")
    assert exc.value.code == 1
    client = sys.modules["ecflow"].Client.return_value
    assert "load_merra2" in client.child_abort.call_args.args[0]
    client.child_complete.assert_not_called()
    pair.assert_not_called()
