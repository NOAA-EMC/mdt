import pytest

from mdt.dag import DAGBuilder


class DummyConfig:
    """Mock MDT config object to inject mock dicts for the DAG builder testing."""

    def __init__(self, data_dict):
        self.data = data_dict
        self.execution = {"default_cluster": "compute"}
        self.pairing = {}
        self.combine = {}
        self.statistics = {}
        self.plots = {}
        self.reductions = {}
        self.save = {}
        self.combine = {}


def test_dag_builder_kerchunk_kwargs():
    """Test that DAGBuilder correctly merges use_kerchunk into kwargs."""
    config_dict = {
        "test_data": {
            "type": "cmaq",
            "use_kerchunk": True,
            "kerchunk_file": "ref.json",
            "kwargs": {"files": "dummy.nc"},
        }
    }
    config = DummyConfig(config_dict)
    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["load_test_data"]

    assert "use_kerchunk" in node["kwargs"]
    assert node["kwargs"]["use_kerchunk"] is True
    assert "kerchunk_file" in node["kwargs"]
    assert node["kwargs"]["kerchunk_file"] == "ref.json"
    assert node["kwargs"]["files"] == "dummy.nc"


def test_dag_builder_cluster_defaults():
    """Test that DAGBuilder assigns 'service' to data nodes and default_cluster to others."""
    config = DummyConfig(data_dict={"test_data": {"type": "cmaq"}, "test_target": {"type": "aeronet"}})
    config.pairing = {"my_pairing": {"source": "test_data", "target": "test_target"}}
    builder = DAGBuilder(config)
    dag = builder.build()

    # Data nodes should default to service cluster
    assert dag.nodes["load_test_data"]["cluster"] == "service"
    assert dag.nodes["load_test_target"]["cluster"] == "service"

    # Pairing nodes should default to the execution block's default cluster
    assert dag.nodes["pair_my_pairing"]["cluster"] == "compute"


def test_dag_builder_kerchunk_no_kwargs():
    """Test that DAGBuilder handles missing kwargs correctly with kerchunk."""
    config_dict = {
        "test_data": {
            "type": "cmaq",
            "use_kerchunk": True,
        }
    }
    config = DummyConfig(config_dict)
    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["load_test_data"]

    assert "use_kerchunk" in node["kwargs"]
    assert node["kwargs"]["use_kerchunk"] is True
    assert "kerchunk_file" not in node["kwargs"]


def test_dag_builder_kerchunk_none_kwargs():
    """Test that DAGBuilder handles kwargs=None correctly with kerchunk."""
    config_dict = {
        "test_data": {
            "type": "cmaq",
            "use_kerchunk": True,
            "kwargs": None,
        }
    }
    config = DummyConfig(config_dict)
    builder = DAGBuilder(config)
    dag = builder.build()

    node = dag.nodes["load_test_data"]

    assert "use_kerchunk" in node["kwargs"]
    assert node["kwargs"]["use_kerchunk"] is True


def test_dag_builder_collects_all_missing_references():
    """Aggregate missing source and target references with their fields."""
    config = DummyConfig({"model": {"type": "cmaq"}})
    config.pairing = {
        "first": {"source": "missing_model", "target": "missing_obs"},
        "second": {"source": "model", "target": "also_missing"},
    }

    with pytest.raises(ValueError, match="Workflow configuration is invalid") as exc:
        DAGBuilder(config).build()

    message = str(exc.value)
    assert "pairing.first.source" in message
    assert "pairing.first.target" in message
    assert "pairing.second.target" in message


def test_dag_builder_repeated_build_is_deterministic_and_does_not_mutate_config():
    """Building the same DAG twice is deterministic and leaves parsed kwargs intact."""
    config_data = {"model": {"type": "cmaq", "kwargs": {"fname": "model.nc"}, "use_kerchunk": True}}
    config = DummyConfig(config_data)
    builder = DAGBuilder(config)

    first_nodes = list(builder.build().nodes(data=True))
    second_nodes = list(builder.build().nodes(data=True))

    assert first_nodes == second_nodes
    assert "use_kerchunk" not in config_data["model"]["kwargs"]
