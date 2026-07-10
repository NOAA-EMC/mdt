"""Property-based tests for region masking configuration validation."""

import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mdt.config import ConfigParser
from mdt.dag import DAGBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(config_dict):
    """Write a config dict to a temporary YAML file and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config_dict, tmp)
    tmp.close()
    return Path(tmp.name)


def _config_with_mask(mask_value):
    """Build a minimal valid config with a pairing task using the given mask value."""
    return {
        "data": {
            "model": {"type": "cmaq"},
            "obs": {"type": "ish_lite"},
        },
        "pairing": {
            "pair_test": {
                "source": "model",
                "target": "obs",
                "mask": mask_value,
            }
        },
    }


def _config_with_regions(regions_value):
    """Build a minimal valid config with a pairing (mask set) and a plots task using the given regions value."""
    return {
        "data": {
            "model": {"type": "cmaq"},
            "obs": {"type": "ish_lite"},
        },
        "pairing": {
            "pair_test": {
                "source": "model",
                "target": "obs",
                "mask": "land",
            }
        },
        "plots": {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": regions_value},
            }
        },
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for valid non-empty strings (at least one non-whitespace character)
_non_empty_str = st.text(min_size=1).filter(lambda s: s.strip() != "")

# Strategy for valid mask values: non-empty strings with at least one non-whitespace char
_valid_mask = _non_empty_str

# Strategy for invalid mask values: anything that is not a non-empty string
_invalid_mask = st.one_of(
    # Empty string
    st.just(""),
    # Whitespace-only strings
    st.from_regex(r"^\s+$", fullmatch=True),
    # None
    st.just(None),
    # Integers
    st.integers(),
    # Lists
    st.lists(st.text(), min_size=0, max_size=5),
    # Floats
    st.floats(allow_nan=False, allow_infinity=False),
    # Booleans
    st.booleans(),
)

# Strategy for valid regions lists: at least one non-empty string
_valid_regions = st.lists(_non_empty_str, min_size=1, max_size=10)

# Strategy for invalid regions values
_invalid_regions = st.one_of(
    # Empty list
    st.just([]),
    # List containing at least one empty string
    st.lists(st.just(""), min_size=1, max_size=5),
    # List containing at least one whitespace-only string
    st.lists(st.just("   "), min_size=1, max_size=5),
    # List with a mix of valid and empty strings
    st.tuples(_non_empty_str, st.just("")).map(lambda t: [t[0], t[1]]),
    # List with a mix of valid and whitespace-only strings
    st.tuples(_non_empty_str, st.just("  ")).map(lambda t: [t[0], t[1]]),
    # Non-list types
    st.just("a single string"),
    st.integers(),
    st.just(None),
    st.just(42.5),
    # List with non-string elements
    st.lists(st.integers(), min_size=1, max_size=5),
    st.tuples(_non_empty_str, st.integers()).map(lambda t: [t[0], t[1]]),
)


# ---------------------------------------------------------------------------
# Feature: region-masking, Property 4: Config mask validation accepts valid strings and rejects invalid values
# ---------------------------------------------------------------------------


class TestConfigMaskValidationProperty:
    """Property 4: Config mask validation accepts valid strings and rejects invalid values.

    Validates: Requirements 4.1, 4.5
    """

    @given(mask=_valid_mask)
    @settings(max_examples=100)
    def test_valid_mask_accepted(self, mask):
        """For any non-empty string value, ConfigParser should accept it as a valid mask key."""
        config = _config_with_mask(mask)
        cfg_path = _write_config(config)
        try:
            parser = ConfigParser(cfg_path)
            assert parser.pairing["pair_test"]["mask"] == mask
        finally:
            cfg_path.unlink(missing_ok=True)

    @given(mask=_invalid_mask)
    @settings(max_examples=100)
    def test_invalid_mask_rejected(self, mask):
        """For any value that is not a non-empty string, ConfigParser should raise ValueError."""
        config = _config_with_mask(mask)
        cfg_path = _write_config(config)
        try:
            with pytest.raises(ValueError, match="has invalid 'mask' value"):
                ConfigParser(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Feature: region-masking, Property 5: Config regions validation accepts valid lists and rejects invalid values
# ---------------------------------------------------------------------------


class TestConfigRegionsValidationProperty:
    """Property 5: Config regions validation accepts valid lists and rejects invalid values.

    Validates: Requirements 4.2, 4.6
    """

    @given(regions=_valid_regions)
    @settings(max_examples=100)
    def test_valid_regions_accepted(self, regions):
        """For any list containing at least one non-empty string, ConfigParser should accept it."""
        config = _config_with_regions(regions)
        cfg_path = _write_config(config)
        try:
            parser = ConfigParser(cfg_path)
            assert parser.plots["my_plot"]["kwargs"]["regions"] == regions
        finally:
            cfg_path.unlink(missing_ok=True)

    @given(regions=_invalid_regions)
    @settings(max_examples=100)
    def test_invalid_regions_rejected(self, regions):
        """For any value that is not a valid regions list, ConfigParser should raise ValueError."""
        config = _config_with_regions(regions)
        cfg_path = _write_config(config)
        try:
            with pytest.raises(ValueError, match="has invalid 'regions'"):
                ConfigParser(cfg_path)
        finally:
            cfg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers for DAG tests
# ---------------------------------------------------------------------------


class _DummyConfig:
    """Minimal config object for DAGBuilder testing without YAML file I/O."""

    def __init__(self):
        self.data = {
            "model": {"type": "cmaq"},
            "obs": {"type": "ish_lite"},
        }
        self.execution = {"default_cluster": "compute"}
        self.pairing = {
            "pair_test": {
                "source": "model",
                "target": "obs",
                "mask": "land",
            }
        }
        self.combine = {}
        self.statistics = {}
        self.plots = {}


# ---------------------------------------------------------------------------
# Feature: region-masking, Property 6: DAG preserves regions list contents and order
# ---------------------------------------------------------------------------


class TestDAGRegionsPreservationProperty:
    """Property 6: DAG preserves regions list contents and order.

    Validates: Requirements 5.3, 5.4
    """

    @given(regions=st.lists(st.text(min_size=1).filter(lambda s: s.strip() != ""), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_plot_node_preserves_regions(self, regions):
        """For any list of region name strings in a plot task's kwargs, the DAGBuilder
        stores an identical list (same elements, same order) as a top-level 'regions'
        attribute on the corresponding plot node.
        """
        config = _DummyConfig()
        config.plots = {
            "my_plot": {
                "input": "pair_test",
                "type": "timeseries",
                "kwargs": {"regions": regions},
            }
        }

        builder = DAGBuilder(config)
        dag = builder.build()

        node = dag.nodes["plot_my_plot"]
        assert "regions" in node, "Plot node should have a 'regions' attribute"
        assert node["regions"] == regions, f"Plot node regions should match input exactly. Expected {regions!r}, got {node['regions']!r}"

    @given(regions=st.lists(st.text(min_size=1).filter(lambda s: s.strip() != ""), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_statistics_node_preserves_regions(self, regions):
        """For any list of region name strings in a statistics task's kwargs, the DAGBuilder
        stores an identical list (same elements, same order) as a top-level 'regions'
        attribute on the corresponding statistics node.
        """
        config = _DummyConfig()
        config.statistics = {
            "my_stats": {
                "input": "pair_test",
                "metrics": ["rmse", "mb"],
                "kwargs": {"regions": regions},
            }
        }

        builder = DAGBuilder(config)
        dag = builder.build()

        node = dag.nodes["stats_my_stats"]
        assert "regions" in node, "Statistics node should have a 'regions' attribute"
        assert node["regions"] == regions, f"Statistics node regions should match input exactly. Expected {regions!r}, got {node['regions']!r}"


# ---------------------------------------------------------------------------
# Feature: region-masking, Property 7: Radar metrics are within defined mathematical bounds
# ---------------------------------------------------------------------------

import os
import sys

import numpy as np

# Add the scripts directory to path so we can import the compute_radar_metrics function
# from the patch source directly (the function is defined in the RADAR_SOURCE string
# within patch_radar.py, but we import it from the installed monet_plots if available,
# or fall back to extracting it from the patch script).
try:
    from monet_plots.plots.radar import compute_radar_metrics
except ImportError:
    # If monet_plots.plots.radar is not available (patch not yet applied),
    # import directly from the patch script's module-level function definition.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    # The patch script embeds the source as a string; we need to exec it to get the function.
    import patch_radar as _patch_radar_mod

    # Extract compute_radar_metrics from the RADAR_SOURCE string
    _ns = {}
    exec(compile(_patch_radar_mod.RADAR_SOURCE, "<radar_source>", "exec"), _ns)
    compute_radar_metrics = _ns["compute_radar_metrics"]


# Strategy for generating numeric arrays suitable for radar metrics testing:
# - At least 2 elements
# - Non-NaN values
# - Non-zero variance (not all identical)
_finite_float = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

_numeric_array = st.lists(_finite_float, min_size=2, max_size=50).filter(lambda xs: np.std(xs, ddof=1) > 0)


class TestRadarMetricsBoundsProperty:
    """Property 7: Radar metrics are within defined mathematical bounds.

    For any pair of observed and predicted numeric arrays with at least 2 valid
    (non-NaN) data points and non-zero variance:
    - IOA should be in [0, 1]
    - KGE should be <= 1
    - CCC should be in [-1, 1]

    Validates: Requirements 8.1, 8.2, 8.3
    """

    @given(obs=_numeric_array, pred=_numeric_array)
    @settings(max_examples=100)
    def test_radar_metrics_bounds(self, obs, pred):
        """For any pair of observed and predicted arrays with at least 2 valid points
        and non-zero variance, IOA is in [0,1], KGE <= 1, and CCC is in [-1,1].
        """
        # Ensure both arrays are the same length (use the shorter length)
        min_len = min(len(obs), len(pred))
        assume(min_len >= 2)
        obs_arr = np.array(obs[:min_len], dtype=float)
        pred_arr = np.array(pred[:min_len], dtype=float)

        # Ensure non-zero variance in both arrays after truncation
        assume(np.std(obs_arr, ddof=1) > 0)
        assume(np.std(pred_arr, ddof=1) > 0)

        metrics = compute_radar_metrics(obs_arr, pred_arr)

        # IOA should be in [0, 1]
        ioa = metrics["IOA"]
        if not np.isnan(ioa):
            assert 0.0 <= ioa <= 1.0, f"IOA={ioa} is outside [0, 1]"

        # KGE should be <= 1
        kge = metrics["KGE"]
        if not np.isnan(kge):
            assert kge <= 1.0, f"KGE={kge} is greater than 1"

        # CCC should be in [-1, 1]
        ccc = metrics["CCC"]
        if not np.isnan(ccc):
            assert -1.0 <= ccc <= 1.0, f"CCC={ccc} is outside [-1, 1]"


# ---------------------------------------------------------------------------
# Feature: region-masking, Property 1: Region filtering preserves only matching data
# ---------------------------------------------------------------------------

import xarray as xr

from mdt.tasks.plotting import _filter_by_region

# Strategy for generating region labels (short non-empty strings)
_region_label = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")


@st.composite
def _dataset_with_regions(draw):
    """Generate an xarray Dataset with a region label variable and some data.

    Returns (dataset, region_var_name, list_of_unique_region_labels).
    """
    # Generate 2-5 unique region labels
    n_regions = draw(st.integers(min_value=2, max_value=5))
    region_labels = draw(
        st.lists(
            _region_label,
            min_size=n_regions,
            max_size=n_regions,
            unique=True,
        )
    )

    # Generate 5-30 data points
    n_points = draw(st.integers(min_value=5, max_value=30))

    # Assign each point a region from the available labels
    point_regions = draw(
        st.lists(
            st.sampled_from(region_labels),
            min_size=n_points,
            max_size=n_points,
        )
    )

    # Generate numeric data values for each point
    values = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n_points,
            max_size=n_points,
        )
    )

    # Build the xarray Dataset
    region_var_name = "mask_region"
    ds = xr.Dataset(
        {
            region_var_name: ("obs", point_regions),
            "temperature": ("obs", values),
        },
        coords={"obs": np.arange(n_points)},
    )

    return ds, region_var_name, region_labels


class TestRegionFilteringProperty:
    """Property 1: Region filtering preserves only matching data.

    For any xarray Dataset containing a region label variable and for any
    region name that exists in that variable, filtering the dataset by that
    region should produce a subset where every data point's region label
    equals the specified region name, and no data points with other region
    labels are present.

    Validates: Requirements 2.2, 3.2
    """

    @given(data=_dataset_with_regions())
    @settings(max_examples=100)
    def test_region_filter_preserves_only_matching(self, data):
        """Filtering by a region that exists in the dataset produces a subset
        containing only data points with that region label.
        """
        ds, region_var, region_labels = data

        # Pick a region that exists in the dataset
        # (all generated labels are assigned to at least the label list,
        # but not necessarily present in the data; pick one that IS present)
        present_labels = set(ds[region_var].values)
        # At least one label must be present since we generated points
        assume(len(present_labels) > 0)

        # Pick one of the labels actually present in the data
        target_region = list(present_labels)[0]

        # Apply the filter
        filtered = _filter_by_region(ds, region_var, target_region)

        # Property: every data point in the filtered result has the target region label
        filtered_labels = filtered[region_var].values
        for label in filtered_labels:
            assert label == target_region, f"Expected all labels to be '{target_region}', but found '{label}'"

        # Property: no data points with other region labels are present
        other_labels = present_labels - {target_region}
        for label in filtered_labels:
            assert label not in other_labels, f"Found label '{label}' from another region in filtered result"

    @given(data=_dataset_with_regions())
    @settings(max_examples=100)
    def test_region_filter_count_matches_original(self, data):
        """The number of points in the filtered result equals the number of
        points with that region label in the original dataset.
        """
        ds, region_var, region_labels = data

        present_labels = set(ds[region_var].values)
        assume(len(present_labels) > 0)

        target_region = list(present_labels)[0]

        # Count matching points in original
        original_count = int((ds[region_var] == target_region).sum().values)

        # Apply the filter
        filtered = _filter_by_region(ds, region_var, target_region)

        # The filtered dataset should have exactly that many points
        assert filtered.sizes["obs"] == original_count, (
            f"Expected {original_count} points for region '{target_region}', got {filtered.sizes['obs']}"
        )


# ---------------------------------------------------------------------------
# Feature: region-masking, Property 2: Plot file path sanitization produces safe filenames
# ---------------------------------------------------------------------------

from mdt.tasks.plotting import _sanitize_region_name


class TestPlotFilePathSanitizationProperty:
    """Property 2: Plot file path sanitization produces safe filenames.

    For any string used as a region name, after sanitization for plot file paths,
    the resulting string should contain only alphanumeric characters, hyphens, and
    periods — all other characters should be replaced with underscores.

    Validates: Requirements 2.4
    """

    @given(region_name=st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_sanitized_name_contains_only_safe_characters(self, region_name):
        """For any string used as a region name, the sanitized result contains only
        alphanumeric characters, hyphens, periods, and underscores (the replacement char).
        """
        sanitized = _sanitize_region_name(region_name)

        import re as _re

        # Every character in the result must be alphanumeric, hyphen, period, or underscore
        # (underscore is the replacement character for unsafe chars)
        assert _re.fullmatch(r"[a-zA-Z0-9\-._]*", sanitized) is not None, (
            f"Sanitized name {sanitized!r} contains characters outside [a-zA-Z0-9\\-._ ]"
        )

    @given(region_name=st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_safe_characters_preserved(self, region_name):
        """Characters that are already safe (alphanumeric, hyphens, periods) are
        preserved in their original positions.
        """
        sanitized = _sanitize_region_name(region_name)

        import re as _re

        # The sanitized string must have the same length as the input
        assert len(sanitized) == len(region_name), f"Sanitized length {len(sanitized)} != input length {len(region_name)}"

        # Each safe character in the input must appear unchanged at the same position
        for i, ch in enumerate(region_name):
            if _re.match(r"[a-zA-Z0-9\-.]", ch):
                assert sanitized[i] == ch, f"Safe character {ch!r} at position {i} was changed to {sanitized[i]!r}"

    @given(region_name=st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_unsafe_characters_replaced_with_underscore(self, region_name):
        """Characters that are not alphanumeric, hyphens, or periods are replaced
        with underscores.
        """
        sanitized = _sanitize_region_name(region_name)

        import re as _re

        for i, ch in enumerate(region_name):
            if not _re.match(r"[a-zA-Z0-9\-.]", ch):
                assert sanitized[i] == "_", f"Unsafe character {ch!r} at position {i} was not replaced with underscore, got {sanitized[i]!r}"
