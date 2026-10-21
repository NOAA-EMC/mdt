# Feature: virtualizarr-zarr-integration, Property 2: Default store_path follows naming pattern
from hypothesis import given, settings
from hypothesis import strategies as st

from mdt.dag import DAGBuilder


class DummyConfig:
    """Mock MDT config object for DAG builder testing."""

    def __init__(self, data_dict):
        self.data = data_dict
        self.execution = {"default_cluster": "compute"}
        self.pairing = {}
        self.combine = {}
        self.statistics = {}
        self.plots = {}


# Strategy: valid Python identifiers — start with a letter, then alphanumeric/underscores
valid_dataset_names = st.from_regex(r"[a-zA-Z][a-zA-Z0-9_]{0,29}", fullmatch=True)


@settings(max_examples=100, deadline=None)
@given(name=valid_dataset_names)
def test_default_store_path_follows_naming_pattern(name):
    """Property 2: For any valid dataset name, when zarr_store.enabled is true.

    And store_path is not specified, the DAGBuilder sets store_path to
    './zarr_stores/{dataset_name}/'.

    **Validates: Requirements 1.7**
    """
    config_dict = {
        name: {
            "type": "cmaq",
            "zarr_store": {
                "enabled": True,
            },
        }
    }
    config = DummyConfig(config_dict)
    builder = DAGBuilder(config)
    dag = builder.build()

    node_id = f"load_{name}"
    assert node_id in dag.nodes, f"Expected node '{node_id}' in DAG"

    node = dag.nodes[node_id]
    assert node["kwargs"]["store_path"] == f"./zarr_stores/{name}/"


# --- Unit tests for DAGBuilder zarr_store param merging ---
# Requirements: 1.4, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5


class TestDAGBuilderZarrStoreAbsent:
    """Tests that zarr_store absence leaves kwargs unchanged."""

    def test_no_zarr_store_key(self):
        """When zarr_store is absent, no VirtualiZarr keys appear in kwargs."""
        config_dict = {
            "my_data": {
                "type": "cmaq",
                "kwargs": {"files": "data.nc"},
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_data"]["kwargs"]

        assert "use_virtualizarr" not in kw
        assert "virtualizarr_backend" not in kw
        assert "store_path" not in kw
        assert "icechunk_repo" not in kw
        assert kw["files"] == "data.nc"

    def test_zarr_store_enabled_false(self):
        """When zarr_store.enabled is false, no VirtualiZarr keys appear."""
        config_dict = {
            "my_data": {
                "type": "cmaq",
                "kwargs": {"files": "data.nc"},
                "zarr_store": {"enabled": False},
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_data"]["kwargs"]

        assert "use_virtualizarr" not in kw
        assert "virtualizarr_backend" not in kw
        assert "store_path" not in kw
        assert "icechunk_repo" not in kw
        assert kw["files"] == "data.nc"

    def test_zarr_store_empty_dict(self):
        """When zarr_store is an empty dict, no VirtualiZarr keys appear."""
        config_dict = {
            "my_data": {
                "type": "cmaq",
                "zarr_store": {},
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_data"]["kwargs"]

        assert "use_virtualizarr" not in kw
        assert "virtualizarr_backend" not in kw
        assert "store_path" not in kw
        assert "icechunk_repo" not in kw


class TestDAGBuilderZarrStoreEnabled:
    """Tests that zarr_store.enabled: true merges all expected params."""

    def test_enabled_with_all_params(self):
        """All zarr_store params are merged into kwargs when enabled."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "kwargs": {"files": "data.nc"},
                "zarr_store": {
                    "enabled": True,
                    "backend": "kerchunk_parquet",
                    "store_path": "/custom/path/",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["use_virtualizarr"] is True
        assert kw["virtualizarr_backend"] == "kerchunk_parquet"
        assert kw["store_path"] == "/custom/path/"
        assert kw["files"] == "data.nc"

    def test_enabled_defaults_backend_and_store_path(self):
        """When backend and store_path are omitted, defaults are applied."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {"enabled": True},
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["use_virtualizarr"] is True
        assert kw["virtualizarr_backend"] == "kerchunk_json"
        assert kw["store_path"] == "./zarr_stores/my_model/"
        assert "icechunk_repo" not in kw

    def test_enabled_kerchunk_json_backend(self):
        """Explicit kerchunk_json backend is forwarded correctly."""
        config_dict = {
            "obs": {
                "type": "aeronet",
                "zarr_store": {
                    "enabled": True,
                    "backend": "kerchunk_json",
                    "store_path": "./refs/obs/",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_obs"]["kwargs"]

        assert kw["use_virtualizarr"] is True
        assert kw["virtualizarr_backend"] == "kerchunk_json"
        assert kw["store_path"] == "./refs/obs/"


class TestDAGBuilderIcechunkRepo:
    """Tests that the icechunk path uses MonetIO-facing kwargs only."""

    def test_icechunk_params_included(self):
        """Icechunk-specific params are merged into MonetIO-facing kwargs."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "icechunk",
                    "icechunk_repo": "s3://bucket/repo",
                    "max_scan_attempts": 3,
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["use_icechunk"] is True
        assert kw["icechunk_url"] == "s3://bucket/repo"
        assert "use_virtualizarr" not in kw
        assert "virtualizarr_backend" not in kw
        assert "icechunk_repo" not in kw
        assert kw["max_scan_attempts"] == 3
        assert "store_path" not in kw

    def test_existing_zarr_included(self):
        """Existing and zarr_kwargs are merged into kwargs when provided."""
        config_dict = {
            "my_model": {
                "type": "aeronet",
                "zarr_store": {
                    "enabled": True,
                    "backend": "zarr",
                    "existing": True,
                    "store_path": "/path/to/zarr",
                    "zarr_kwargs": {"consolidated": True},
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["existing_zarr"] is True
        assert kw["virtualizarr_backend"] == "zarr"
        assert kw["store_path"] == "/path/to/zarr"
        assert kw["zarr_kwargs"] == {"consolidated": True}

    def test_icechunk_repo_included(self):
        """icechunk_repo alias is normalized to icechunk_url in kwargs."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "icechunk",
                    "icechunk_repo": "s3://bucket/repo",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["use_icechunk"] is True
        assert kw["icechunk_url"] == "s3://bucket/repo"
        assert "virtualizarr_backend" not in kw
        assert "icechunk_repo" not in kw
        assert "store_path" not in kw

    def test_icechunk_repo_absent_for_kerchunk(self):
        """icechunk_repo is NOT in kwargs when backend is not icechunk."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "kerchunk_parquet",
                    "store_path": "/data/refs/",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert "icechunk_repo" not in kw
        assert kw["virtualizarr_backend"] == "kerchunk_parquet"

    def test_icechunk_repo_empty_string_not_included(self):
        """An empty icechunk_repo string is treated as absent."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "kerchunk_json",
                    "icechunk_repo": "",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert "icechunk_repo" not in kw


class TestDAGBuilderZarrStoreKerchunkCoexistence:
    """Tests that zarr_store params coexist with existing use_kerchunk params."""

    def test_both_kerchunk_and_zarr_store(self):
        """Both use_kerchunk and zarr_store params appear in kwargs together."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "use_kerchunk": True,
                "kerchunk_file": "ref.json",
                "kwargs": {"files": "data.nc"},
                "zarr_store": {
                    "enabled": True,
                    "backend": "kerchunk_parquet",
                    "store_path": "/zarr/path/",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        # Kerchunk params present
        assert kw["use_kerchunk"] is True
        assert kw["kerchunk_file"] == "ref.json"

        # VirtualiZarr params present
        assert kw["use_virtualizarr"] is True
        assert kw["virtualizarr_backend"] == "kerchunk_parquet"
        assert kw["store_path"] == "/zarr/path/"

        # Original kwargs preserved
        assert kw["files"] == "data.nc"

    def test_kerchunk_without_zarr_store(self):
        """use_kerchunk alone does not introduce VirtualiZarr keys."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "use_kerchunk": True,
                "kerchunk_file": "ref.json",
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["use_kerchunk"] is True
        assert kw["kerchunk_file"] == "ref.json"
        assert "use_virtualizarr" not in kw
        assert "virtualizarr_backend" not in kw
        assert "store_path" not in kw
        assert "icechunk_repo" not in kw

    def test_zarr_store_without_kerchunk(self):
        """zarr_store alone does not introduce kerchunk keys."""
        config_dict = {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "kerchunk_json",
                },
            }
        }
        config = DummyConfig(config_dict)
        dag = DAGBuilder(config).build()
        kw = dag.nodes["load_my_model"]["kwargs"]

        assert kw["use_virtualizarr"] is True
        assert kw["virtualizarr_backend"] == "kerchunk_json"
        assert "use_kerchunk" not in kw
        assert "kerchunk_file" not in kw
