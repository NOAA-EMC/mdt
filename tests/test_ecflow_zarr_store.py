# Feature: virtualizarr-zarr-integration, Property 5: ecFlow variables reflect DAG node kwargs
"""Property-based tests for ecFlow ZARR_STORE_* variables.

Validates: Requirements 6.1
"""

import sys
import types
from unittest.mock import MagicMock

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Fixtures (adapted from tests/test_ecflow_engine.py)
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
def _make_config():
    """Factory that builds a mock config with a given execution dict."""

    def _factory(exec_dict=None):
        cfg = MagicMock()
        cfg.execution = exec_dict if exec_dict is not None else {}
        return cfg

    return _factory


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

VALID_BACKENDS = ["kerchunk_json", "kerchunk_parquet", "icechunk"]

backend_strategy = st.sampled_from(VALID_BACKENDS)

store_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), blacklist_characters=("\x00",)),
    min_size=1,
    max_size=100,
)

icechunk_repo_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), blacklist_characters=("\x00",)),
        min_size=1,
        max_size=100,
    ),
)


# ---------------------------------------------------------------------------
# Property 5: ecFlow variables reflect DAG node kwargs
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------


class TestEcFlowZarrStoreVariablesProperty:
    """Property 5: ecFlow variables reflect DAG node kwargs."""

    @given(
        backend=backend_strategy,
        store_path=store_path_strategy,
        icechunk_repo=icechunk_repo_strategy,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ecflow_variables_reflect_dag_kwargs(
        self,
        _tracking_ecflow,
        _make_config,
        backend,
        store_path,
        icechunk_repo,
    ):
        """**Validates: Requirements 6.1**.

        For any load_data DAG node whose kwargs contain use_virtualizarr=True,
        build_suite() sets ZARR_STORE_ENABLED, ZARR_STORE_BACKEND,
        ZARR_STORE_PATH, and ZARR_STORE_ICECHUNK_REPO on the ecFlow task node
        with values matching the kwargs.
        """
        # Feature: virtualizarr-zarr-integration, Property 5: ecFlow variables reflect DAG node kwargs
        from mdt.ecflow_engine import EcFlowEngine

        # Build kwargs with VirtualiZarr enabled
        node_kwargs = {
            "use_virtualizarr": True,
            "virtualizarr_backend": backend,
            "store_path": store_path,
        }
        if icechunk_repo is not None:
            node_kwargs["icechunk_repo"] = icechunk_repo

        # Build a DAG with a single load_data node
        dag = nx.DiGraph()
        dag.add_node(
            "load_test",
            task_type="load_data",
            name="test_dataset",
            dataset_type="test_type",
            kwargs=node_kwargs,
        )

        engine = EcFlowEngine(dag, _make_config())
        engine.build_suite()

        # Inspect the ecFlow variables set on the task node
        task_vars = _tracking_ecflow["tasks"]["load_test"]._variables

        assert task_vars["ZARR_STORE_ENABLED"] == "true"
        assert task_vars["ZARR_STORE_BACKEND"] == backend
        assert task_vars["ZARR_STORE_PATH"] == store_path

        if icechunk_repo is not None:
            assert task_vars["ZARR_STORE_ICECHUNK_REPO"] == icechunk_repo
        else:
            assert task_vars["ZARR_STORE_ICECHUNK_REPO"] == ""

        # Clean up tracking state for next Hypothesis example
        _tracking_ecflow["defs"].clear()
        _tracking_ecflow["suites"].clear()
        _tracking_ecflow["families"].clear()
        _tracking_ecflow["tasks"].clear()


# ---------------------------------------------------------------------------
# Fixture for wrapper-content tests (simpler than _tracking_ecflow)
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_ecflow(monkeypatch):
    """Inject a fake ``ecflow`` module so tests run without the real package."""
    fake = types.ModuleType("ecflow")
    fake.Defs = lambda *a, **kw: MagicMock(name="Defs")
    fake.Suite = lambda *a, **kw: MagicMock(name="Suite")
    fake.Family = lambda *a, **kw: MagicMock(name="Family")
    fake.Task = lambda *a, **kw: MagicMock(name="Task")
    fake.Client = MagicMock
    monkeypatch.setitem(sys.modules, "ecflow", fake)
    return fake


# ---------------------------------------------------------------------------
# Unit tests for ecFlow wrapper template and variables
# Validates: Requirements 6.1, 6.2, 6.3
# ---------------------------------------------------------------------------


class TestEcFlowWrapperScriptContent:
    """Requirement 6.2, 6.3 — wrapper script reads ZARR_STORE_* variables."""

    @staticmethod
    def _read_script(engine, node_id):
        """Read the generated .ecf script for *node_id*."""
        from pathlib import Path

        matches = list(Path(engine.task_script_dir).rglob(f"{node_id}.ecf"))
        assert len(matches) == 1
        with matches[0].open() as fh:
            return fh.read()

    def test_wrapper_contains_zarr_store_enabled_variable(self, _fake_ecflow, _make_config, tmp_path):
        """The generated wrapper reads %ZARR_STORE_ENABLED% ecFlow variable."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_obs",
            task_type="load_data",
            name="obs",
            dataset_type="aeronet",
            kwargs={},
        )
        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "decode('%ZARR_STORE_ENABLED_B64%')" in content

    def test_wrapper_contains_conditional_logic_for_disabled(self, _fake_ecflow, _make_config, tmp_path):
        """The wrapper has if/else logic that only injects VirtualiZarr params when enabled."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_obs",
            task_type="load_data",
            name="obs",
            dataset_type="aeronet",
            kwargs={},
        )
        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        # The script should check whether zarr is enabled before injecting params
        assert "zarr_enabled == 'true'" in content
        # When enabled, it should inject use_virtualizarr
        assert "kwargs['use_virtualizarr'] = True" in content

    def test_wrapper_reads_all_zarr_store_variables(self, _fake_ecflow, _make_config, tmp_path):
        """The wrapper decodes the Zarr backend, path, and Icechunk URL safely."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_obs",
            task_type="load_data",
            name="obs",
            dataset_type="aeronet",
            kwargs={},
        )
        cfg = _make_config({"task_script_dir": str(tmp_path / "scripts")})
        engine = EcFlowEngine(dag, cfg)
        engine.generate_task_wrappers()

        content = self._read_script(engine, "load_obs")
        assert "decode('%ZARR_STORE_BACKEND_B64%')" in content
        assert "decode('%ZARR_STORE_PATH_B64%')" in content
        assert "decode('%ZARR_STORE_ICECHUNK_URL_B64%')" in content


class TestEcFlowZarrStoreVariablesUnit:
    """Requirement 6.1 — ecFlow variables set correctly for each backend."""

    def test_variables_for_kerchunk_json(self, _tracking_ecflow, _make_config):
        """ZARR_STORE_* variables are correct for kerchunk_json backend."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_obs",
            task_type="load_data",
            name="obs",
            dataset_type="aeronet",
            kwargs={
                "use_virtualizarr": True,
                "virtualizarr_backend": "kerchunk_json",
                "store_path": "./zarr_stores/obs/",
            },
        )
        engine = EcFlowEngine(dag, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["load_obs"]._variables
        assert v["ZARR_STORE_ENABLED"] == "true"
        assert v["ZARR_STORE_BACKEND"] == "kerchunk_json"
        assert v["ZARR_STORE_PATH"] == "./zarr_stores/obs/"
        assert v["ZARR_STORE_ICECHUNK_REPO"] == ""

    def test_variables_for_kerchunk_parquet(self, _tracking_ecflow, _make_config):
        """ZARR_STORE_* variables are correct for kerchunk_parquet backend."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_model",
            task_type="load_data",
            name="model",
            dataset_type="merra2",
            kwargs={
                "use_virtualizarr": True,
                "virtualizarr_backend": "kerchunk_parquet",
                "store_path": "/data/zarr/model/",
            },
        )
        engine = EcFlowEngine(dag, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["load_model"]._variables
        assert v["ZARR_STORE_ENABLED"] == "true"
        assert v["ZARR_STORE_BACKEND"] == "kerchunk_parquet"
        assert v["ZARR_STORE_PATH"] == "/data/zarr/model/"
        assert v["ZARR_STORE_ICECHUNK_REPO"] == ""

    def test_variables_for_icechunk(self, _tracking_ecflow, _make_config):
        """ZARR_STORE_* variables are correct for icechunk backend with repo."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_sat",
            task_type="load_data",
            name="sat",
            dataset_type="satellite",
            kwargs={
                "use_virtualizarr": True,
                "virtualizarr_backend": "icechunk",
                "store_path": "/data/zarr/sat/",
                "icechunk_repo": "s3://bucket/repo",
            },
        )
        engine = EcFlowEngine(dag, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["load_sat"]._variables
        assert v["ZARR_STORE_ENABLED"] == "true"
        assert v["ZARR_STORE_BACKEND"] == "icechunk"
        assert v["ZARR_STORE_PATH"] == "/data/zarr/sat/"
        assert v["ZARR_STORE_ICECHUNK_REPO"] == "s3://bucket/repo"

    def test_icechunk_repo_empty_when_not_icechunk(self, _tracking_ecflow, _make_config):
        """ZARR_STORE_ICECHUNK_REPO is empty string for non-icechunk backends."""
        from mdt.ecflow_engine import EcFlowEngine

        for backend in ("kerchunk_json", "kerchunk_parquet"):
            dag = nx.DiGraph()
            dag.add_node(
                "load_ds",
                task_type="load_data",
                name="ds",
                dataset_type="obs",
                kwargs={
                    "use_virtualizarr": True,
                    "virtualizarr_backend": backend,
                    "store_path": "./zarr_stores/ds/",
                },
            )
            engine = EcFlowEngine(dag, _make_config())
            engine.build_suite()

            v = _tracking_ecflow["tasks"]["load_ds"]._variables
            assert v["ZARR_STORE_ICECHUNK_REPO"] == "", f"Expected empty ZARR_STORE_ICECHUNK_REPO for {backend}"

            # Clean up tracking state for next iteration
            _tracking_ecflow["defs"].clear()
            _tracking_ecflow["suites"].clear()
            _tracking_ecflow["families"].clear()
            _tracking_ecflow["tasks"].clear()

    def test_variables_disabled_when_no_virtualizarr(self, _tracking_ecflow, _make_config):
        """ZARR_STORE_ENABLED is 'false' when use_virtualizarr is not in kwargs."""
        from mdt.ecflow_engine import EcFlowEngine

        dag = nx.DiGraph()
        dag.add_node(
            "load_plain",
            task_type="load_data",
            name="plain",
            dataset_type="aeronet",
            kwargs={},
        )
        engine = EcFlowEngine(dag, _make_config())
        engine.build_suite()

        v = _tracking_ecflow["tasks"]["load_plain"]._variables
        assert v["ZARR_STORE_ENABLED"] == "false"
        assert v["ZARR_STORE_BACKEND"] == ""
        assert v["ZARR_STORE_PATH"] == ""
        assert v["ZARR_STORE_ICECHUNK_REPO"] == ""
