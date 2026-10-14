"""Tests for ConfigParser region masking validation."""

import pytest
import yaml

from mdt.config import ConfigParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path, config_dict):
    """Helper: write a config dict to a YAML file and return the path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_dict))
    return config_file


def _base_config_with_pairing(mask=None):
    """Return a minimal valid config with a pairing task, optionally with mask."""
    pairing = {"source": "model", "target": "obs"}
    if mask is not None:
        pairing["mask"] = mask
    return {
        "data": {
            "model": {"type": "cmaq"},
            "obs": {"type": "ish_lite"},
        },
        "pairing": {"pair_test": pairing},
    }


# ---------------------------------------------------------------------------
# Tests: mask key validation (Requirement 4.1, 4.5)
# ---------------------------------------------------------------------------


class TestMaskValidation:
    """Tests for mask key validation on pairing tasks."""

    def test_valid_mask_string(self, tmp_path):
        """A non-empty string mask value should be accepted."""
        config = _base_config_with_pairing(mask="land")
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert parser.pairing["pair_test"]["mask"] == "land"

    def test_mask_empty_string_raises(self, tmp_path):
        """An empty string mask value should raise ValueError."""
        config = _base_config_with_pairing(mask="")
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Pairing 'pair_test' has invalid 'mask' value"):
            ConfigParser(cfg)

    def test_mask_whitespace_only_raises(self, tmp_path):
        """A whitespace-only mask value should raise ValueError."""
        config = _base_config_with_pairing(mask="   ")
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Pairing 'pair_test' has invalid 'mask' value"):
            ConfigParser(cfg)

    def test_mask_integer_raises(self, tmp_path):
        """A non-string mask value (integer) should raise ValueError."""
        config = _base_config_with_pairing(mask=42)
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Pairing 'pair_test' has invalid 'mask' value"):
            ConfigParser(cfg)

    def test_mask_none_raises(self, tmp_path):
        """A None mask value should raise ValueError (key present but null)."""
        config = _base_config_with_pairing()
        config["pairing"]["pair_test"]["mask"] = None
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Pairing 'pair_test' has invalid 'mask' value"):
            ConfigParser(cfg)

    def test_mask_list_raises(self, tmp_path):
        """A list mask value should raise ValueError."""
        config = _base_config_with_pairing(mask=["land"])
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Pairing 'pair_test' has invalid 'mask' value"):
            ConfigParser(cfg)

    def test_no_mask_key_is_valid(self, tmp_path):
        """Absence of mask key should not raise (backward compatibility)."""
        config = _base_config_with_pairing()
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert "mask" not in parser.pairing["pair_test"]


# ---------------------------------------------------------------------------
# Tests: regions key validation (Requirement 4.2, 4.6)
# ---------------------------------------------------------------------------


class TestRegionsValidation:
    """Tests for regions key validation on plot/statistics tasks."""

    def test_valid_regions_in_plots(self, tmp_path):
        """A valid regions list in plots kwargs should be accepted."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": ["North America", "Europe"]},
            }
        }
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert parser.plots["my_plot"]["kwargs"]["regions"] == ["North America", "Europe"]

    def test_valid_regions_in_statistics(self, tmp_path):
        """A valid regions list in statistics kwargs should be accepted."""
        config = _base_config_with_pairing(mask="land")
        config["statistics"] = {
            "my_stats": {
                "input": "pair_test",
                "metrics": ["rmse"],
                "kwargs": {"regions": ["Asia"]},
            }
        }
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert parser.statistics["my_stats"]["kwargs"]["regions"] == ["Asia"]

    def test_regions_empty_list_raises(self, tmp_path):
        """An empty regions list should raise ValueError."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": []},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Plots task 'my_plot' has invalid 'regions'"):
            ConfigParser(cfg)

    def test_regions_with_empty_string_raises(self, tmp_path):
        """A regions list containing an empty string should raise ValueError."""
        config = _base_config_with_pairing(mask="land")
        config["statistics"] = {
            "my_stats": {
                "input": "pair_test",
                "metrics": ["rmse"],
                "kwargs": {"regions": ["valid", ""]},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Statistics task 'my_stats' has invalid 'regions'"):
            ConfigParser(cfg)

    def test_regions_with_whitespace_only_string_raises(self, tmp_path):
        """A regions list containing a whitespace-only string should raise ValueError."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": ["  "]},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Plots task 'my_plot' has invalid 'regions'"):
            ConfigParser(cfg)

    def test_regions_not_a_list_raises(self, tmp_path):
        """A non-list regions value should raise ValueError."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": "North America"},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Plots task 'my_plot' has invalid 'regions'"):
            ConfigParser(cfg)

    def test_regions_with_non_string_element_raises(self, tmp_path):
        """A regions list with a non-string element should raise ValueError."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": ["valid", 123]},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Plots task 'my_plot' has invalid 'regions'"):
            ConfigParser(cfg)

    def test_no_regions_key_is_valid(self, tmp_path):
        """Absence of regions key should not raise (backward compatibility)."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"savename": "output.png"},
            }
        }
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert "regions" not in parser.plots["my_plot"]["kwargs"]

    def test_null_kwargs_is_valid(self, tmp_path):
        """A null kwargs value should not raise."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": None,
            }
        }
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert parser.plots["my_plot"]["kwargs"] is None


# ---------------------------------------------------------------------------
# Tests: cross-validation (Requirement 4.3, 4.4)
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Tests for cross-validation between regions and mask."""

    def test_regions_without_mask_on_pairing_raises(self, tmp_path):
        """Regions referencing a pairing without mask should raise ValueError."""
        config = _base_config_with_pairing()  # No mask
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": ["North America"]},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(
            ValueError,
            match="Plots task 'my_plot' specifies 'regions' but its input pairing 'pair_test' does not define a 'mask' key",
        ):
            ConfigParser(cfg)

    def test_regions_with_mask_on_pairing_is_valid(self, tmp_path):
        """Regions referencing a pairing with mask should be accepted."""
        config = _base_config_with_pairing(mask="land")
        config["statistics"] = {
            "my_stats": {
                "input": "pair_test",
                "metrics": ["rmse"],
                "kwargs": {"regions": ["Europe"]},
            }
        }
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        assert parser.statistics["my_stats"]["kwargs"]["regions"] == ["Europe"]

    def test_regions_with_nonexistent_input_raises(self, tmp_path):
        """Regions referencing a non-existent pairing should raise ValueError."""
        config = _base_config_with_pairing(mask="land")
        config["plots"] = {
            "my_plot": {
                "input": "nonexistent_pairing",
                "type": "timeseries",
                "kwargs": {"regions": ["Asia"]},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(
            ValueError,
            match="Plots task 'my_plot' specifies 'regions' but its input pairing 'nonexistent_pairing' does not define a 'mask' key",
        ):
            ConfigParser(cfg)

    def test_statistics_regions_without_mask_raises(self, tmp_path):
        """Statistics regions referencing a pairing without mask should raise."""
        config = _base_config_with_pairing()  # No mask
        config["statistics"] = {
            "my_stats": {
                "input": "pair_test",
                "metrics": ["rmse"],
                "kwargs": {"regions": ["Europe"]},
            }
        }
        cfg = _write_config(tmp_path, config)
        with pytest.raises(
            ValueError,
            match="Statistics task 'my_stats' specifies 'regions' but its input pairing 'pair_test' does not define a 'mask' key",
        ):
            ConfigParser(cfg)

    def test_backward_compat_no_mask_no_regions(self, tmp_path):
        """Config without mask or regions should validate normally (backward compat)."""
        config = _base_config_with_pairing()  # No mask
        config["plots"] = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"savename": "output.png"},
            }
        }
        config["statistics"] = {
            "my_stats": {
                "input": "pair_test",
                "metrics": ["rmse"],
                "kwargs": {"obs_var": "t2m"},
            }
        }
        cfg = _write_config(tmp_path, config)
        parser = ConfigParser(cfg)
        # No mask on pairing
        assert "mask" not in parser.pairing["pair_test"]
        # No regions on plots or statistics
        assert "regions" not in parser.plots["my_plot"]["kwargs"]
        assert "regions" not in parser.statistics["my_stats"]["kwargs"]
