import sys
import types
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

# Create a fake prefect module before anything else
prefect_mock = types.ModuleType("prefect")
prefect_mock.task = MagicMock()
prefect_mock.flow = MagicMock()
prefect_mock.get_run_logger = MagicMock()

prefect_dask_mock = types.ModuleType("prefect_dask")
prefect_dask_task_runners_mock = types.ModuleType("prefect_dask.task_runners")
prefect_dask_mock.task_runners = prefect_dask_task_runners_mock
prefect_dask_task_runners_mock.DaskTaskRunner = MagicMock()

sys.modules["prefect"] = prefect_mock
sys.modules["prefect_dask"] = prefect_dask_mock
sys.modules["prefect_dask.task_runners"] = prefect_dask_task_runners_mock

# Don't mock dask at module level as it breaks xarray
# Instead, we will patch what we need inside the tests

from mdt.engine import PrefectEngine


class TestPrefectVirtualiZarrIntegration:
    """Test suite for VirtualiZarr integration in PrefectEngine."""

    def mock_task_decorator(self, *args, **kwargs):
        def wrapper(f):
            m = MagicMock(name=f"Task({kwargs.get('name', 'unknown')})")
            m.with_options.return_value = m
            m._original_func = f
            return m

        return wrapper

    def mock_flow_decorator(self, *args, **kwargs):
        def wrapper(f):
            def flow_runner(*a, **k):
                return f(*a, **k)

            flow_runner.with_options = MagicMock(return_value=flow_runner)
            return flow_runner

        return wrapper

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        """Setup Prefect and Dask mocks for engine tests."""
        # Reset mocks before each test
        prefect_mock.task.reset_mock()
        prefect_mock.flow.reset_mock()
        prefect_mock.get_run_logger.reset_mock()

        prefect_mock.task.side_effect = self.mock_task_decorator
        prefect_mock.flow.side_effect = self.mock_flow_decorator

        # Mock dask/distributed to avoid real cluster creation
        # We explicitly mock the distributed module in sys.modules to avoid AttributeError
        mock_distributed = types.ModuleType("dask.distributed")
        mock_distributed.LocalCluster = MagicMock()
        mock_distributed.Client = MagicMock()
        mocker.patch.dict(sys.modules, {"dask.distributed": mock_distributed})

        # We also need to make sure dask itself has the distributed attribute if it's already in sys.modules
        if "dask" in sys.modules:
            sys.modules["dask"].distributed = mock_distributed
        else:
            mock_dask = types.ModuleType("dask")
            mock_dask.distributed = mock_distributed
            mock_dask.annotate = MagicMock()
            mocker.patch.dict(sys.modules, {"dask": mock_dask})

        mocker.patch("dask.annotate", return_value=MagicMock())

    def test_execute_applies_virtualizarr_options(self, mocker):
        """Verify that with_options is called with correct tags and names for VirtualiZarr."""
        # Create a DAG with one VirtualiZarr node and one standard node
        dag = nx.DiGraph()
        dag.add_node(
            "load_vz",
            task_type="load_data",
            name="vz_ds",
            dataset_type="merra2",
            kwargs={"use_virtualizarr": True, "virtualizarr_backend": "kerchunk_json"},
            cluster="service",
        )
        dag.add_node("load_std", task_type="load_data", name="std_ds", dataset_type="cmaq", kwargs={}, cluster="service")

        config = MagicMock()
        config.execution = {"clusters": {"service": {"mode": "local"}}}

        # In the new code, task decorators are applied inside execute()
        # We need to capture the task object when it's created

        load_task_mock = MagicMock(name="LoadDataTask")
        load_task_mock.with_options.return_value = load_task_mock

        def side_effect(*args, **kwargs):
            if kwargs.get("name") == "Load Data":
                return lambda f: load_task_mock
            return self.mock_task_decorator(*args, **kwargs)

        with patch.object(prefect_mock, "task", side_effect=side_effect):
            engine = PrefectEngine(dag, config)
            engine.execute()

        assert load_task_mock.with_options.call_count == 2
        calls = load_task_mock.with_options.call_args_list

        vz_call = next((c for c in calls if c.kwargs.get("tags") == ["virtualizarr"]), None)
        assert vz_call is not None
        assert vz_call.kwargs["task_run_name"] == "Load Data: vz_ds [VirtualiZarr]"

        std_call = next((c for c in calls if not c.kwargs.get("tags")), None)
        assert std_call is not None
        assert std_call.kwargs == {}

    def test_prefect_load_data_logging(self, mocker):
        """Verify that the prefect_load_data task logs VirtualiZarr info when enabled."""
        import mdt.engine

        mock_logger = MagicMock()
        # In the new structure, get_run_logger is imported inside the functions
        # We need to mock it in prefect_mock
        prefect_mock.get_run_logger.return_value = mock_logger

        vz_kwargs = {
            "use_virtualizarr": True,
            "virtualizarr_backend": "icechunk",
            "store_path": "/tmp/vz",
            "icechunk_repo": "s3://repo",
            "files": ["fake.nc"],
        }
        # prefect_load_data is a function defined at module level
        # We MUST mock monetio because calling it triggers environment-dependent imports/validation
        mock_monetio = MagicMock()
        mock_monetio.load.return_value = MagicMock()

        with patch.dict("sys.modules", {"monetio": mock_monetio}):
            mdt.engine.prefect_load_data(name="my_vz", dataset_type="merra2", kwargs=vz_kwargs)
            assert mock_monetio.load.called

        log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("VirtualiZarr" in msg for msg in log_messages)
        assert any("backend: icechunk" in msg for msg in log_messages)
        assert any("store: /tmp/vz" in msg for msg in log_messages)
        assert any("icechunk_repo: s3://repo" in msg for msg in log_messages)

        mock_logger.info.reset_mock()
        mock_monetio.load.reset_mock()
        with patch.dict("sys.modules", {"monetio": mock_monetio}):
            mdt.engine.prefect_load_data(name="my_std", dataset_type="cmaq", kwargs={"files": ["fake.nc"]})
            assert mock_monetio.load.called

        log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Loading data: my_std of type cmaq" in msg for msg in log_messages)
        assert not any("VirtualiZarr" in msg for msg in log_messages)
