"""Unit tests for region-filtered plotting in generate_plot.

Tests verify that:
- One plot file is produced per region
- {region} placeholder is substituted with sanitized names
- ValueError raised when {region} placeholder is missing
- Warning logged and region skipped when zero data points match
- Backward compatibility: no regions produces single plot

Requirements: 2.1, 2.3, 2.5, 2.6, 2.7, 2.8
"""

import logging

import numpy as np
import pytest
import xarray as xr

from mdt.tasks.plotting import (
    _find_region_variable,
    _sanitize_region_name,
    generate_plot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_with_regions():
    """Create a dataset with a region label variable (simulating query_mask output)."""
    n = 12
    regions = ["North America", "Europe", "Asia"] * 4
    return xr.Dataset(
        {
            "temperature": ("x", np.random.rand(n)),
            "land": ("x", regions),
        },
        coords={
            "lat": ("x", np.linspace(-60, 60, n)),
            "lon": ("x", np.linspace(-180, 180, n)),
        },
    )


@pytest.fixture
def dataset_no_region_var():
    """Create a dataset without a region label variable."""
    n = 10
    return xr.Dataset(
        {
            "temperature": ("x", np.random.rand(n)),
            "pressure": ("x", np.random.rand(n)),
        },
        coords={
            "lat": ("x", np.linspace(-60, 60, n)),
            "lon": ("x", np.linspace(-180, 180, n)),
        },
    )


@pytest.fixture
def dataset_with_empty_region():
    """Create a dataset where one region has no matching data points."""
    n = 8
    # Only "North America" and "Europe" have data; "Antarctica" does not
    regions = ["North America", "Europe", "North America", "Europe", "North America", "Europe", "North America", "Europe"]
    return xr.Dataset(
        {
            "temperature": ("x", np.random.rand(n)),
            "land": ("x", regions),
        },
        coords={
            "lat": ("x", np.linspace(-60, 60, n)),
            "lon": ("x", np.linspace(-180, 180, n)),
        },
    )


# ---------------------------------------------------------------------------
# Tests: One plot file produced per region (Requirement 2.1)
# ---------------------------------------------------------------------------


class TestOnePlotPerRegion:
    """Test that generate_plot produces one plot per region."""

    def test_produces_one_result_per_region(self, mocker, dataset_with_regions):
        """When regions are specified, generate_plot returns one result per region."""
        mock_single = mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        results = generate_plot(
            name="test_plot",
            plot_type="timeseries",
            input_data=dataset_with_regions,
            kwargs={
                "savename": "output_{region}.png",
                "regions": ["North America", "Europe", "Asia"],
            },
        )

        assert isinstance(results, list)
        assert len(results) == 3
        assert mock_single.call_count == 3

    def test_produces_two_results_for_two_regions(self, mocker, dataset_with_regions):
        """When 2 regions are specified, exactly 2 plots are generated."""
        mock_single = mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        results = generate_plot(
            name="test_plot",
            plot_type="scatter",
            input_data=dataset_with_regions,
            kwargs={
                "savename": "scatter_{region}.png",
                "regions": ["North America", "Europe"],
            },
        )

        assert len(results) == 2
        assert mock_single.call_count == 2


# ---------------------------------------------------------------------------
# Tests: {region} placeholder substitution with sanitized names (Requirement 2.3, 2.4)
# ---------------------------------------------------------------------------


class TestRegionPlaceholderSubstitution:
    """Test that {region} placeholder is substituted with sanitized region names."""

    def test_placeholder_substituted_with_sanitized_name(self, mocker, dataset_with_regions):
        """The {region} placeholder is replaced with the sanitized region name."""
        call_kwargs = []

        def capture_kwargs(name, plot_type, data, kwargs, track):
            call_kwargs.append(kwargs.copy())
            return "plot_obj"

        mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            side_effect=capture_kwargs,
        )

        generate_plot(
            name="test_plot",
            plot_type="timeseries",
            input_data=dataset_with_regions,
            kwargs={
                "savename": "output_{region}.png",
                "regions": ["North America", "Europe"],
            },
        )

        # "North America" -> "North_America" (space replaced with underscore)
        assert call_kwargs[0]["savename"] == "output_North_America.png"
        # "Europe" stays as "Europe" (no special chars)
        assert call_kwargs[1]["savename"] == "output_Europe.png"

    def test_special_characters_sanitized(self, mocker, dataset_with_regions):
        """Region names with special characters are sanitized in the savename."""
        # Add a region with special characters to the dataset
        n = 4
        ds = xr.Dataset(
            {
                "temperature": ("x", np.random.rand(n)),
                "land": ("x", ["US/Canada (East)", "US/Canada (East)", "US/Canada (East)", "US/Canada (East)"]),
            },
            coords={
                "lat": ("x", np.linspace(30, 50, n)),
                "lon": ("x", np.linspace(-100, -70, n)),
            },
        )

        call_kwargs = []

        def capture_kwargs(name, plot_type, data, kwargs, track):
            call_kwargs.append(kwargs.copy())
            return "plot_obj"

        mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            side_effect=capture_kwargs,
        )

        generate_plot(
            name="test_plot",
            plot_type="timeseries",
            input_data=ds,
            kwargs={
                "savename": "plot_{region}.png",
                "regions": ["US/Canada (East)"],
            },
        )

        # "/" and "(" and ")" and space should all become underscores
        assert call_kwargs[0]["savename"] == "plot_US_Canada__East_.png"

    def test_sanitize_preserves_hyphens_and_periods(self):
        """Hyphens and periods are preserved in sanitized region names."""
        assert _sanitize_region_name("US-East.v2") == "US-East.v2"

    def test_sanitize_replaces_spaces_and_special_chars(self):
        """Spaces and special characters are replaced with underscores."""
        assert _sanitize_region_name("North America") == "North_America"
        assert _sanitize_region_name("Asia (South)") == "Asia__South_"


# ---------------------------------------------------------------------------
# Tests: ValueError when {region} placeholder missing (Requirement 2.8)
# ---------------------------------------------------------------------------


class TestMissingPlaceholderRaisesError:
    """Test that ValueError is raised when {region} placeholder is missing."""

    def test_raises_valueerror_without_placeholder(self, dataset_with_regions):
        """When regions are specified but savename lacks {region}, raise ValueError."""
        with pytest.raises(ValueError, match="savename must contain.*\\{region\\}.*placeholder"):
            generate_plot(
                name="test_plot",
                plot_type="timeseries",
                input_data=dataset_with_regions,
                kwargs={
                    "savename": "output.png",
                    "regions": ["North America", "Europe"],
                },
            )

    def test_raises_valueerror_default_savename_without_placeholder(self, dataset_with_regions):
        """When regions are specified and no savename is provided, raise ValueError."""
        with pytest.raises(ValueError, match="savename must contain.*\\{region\\}.*placeholder"):
            generate_plot(
                name="test_plot",
                plot_type="timeseries",
                input_data=dataset_with_regions,
                kwargs={
                    "regions": ["North America"],
                },
            )

    def test_no_error_when_placeholder_present(self, mocker, dataset_with_regions):
        """When {region} placeholder is present, no ValueError is raised."""
        mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        # Should not raise
        results = generate_plot(
            name="test_plot",
            plot_type="timeseries",
            input_data=dataset_with_regions,
            kwargs={
                "savename": "output_{region}.png",
                "regions": ["North America"],
            },
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests: Warning logged and region skipped when zero data points (Requirement 2.6)
# ---------------------------------------------------------------------------


class TestEmptyRegionSkipped:
    """Test that regions with zero data points are skipped with a warning."""

    def test_warning_logged_for_empty_region(self, mocker, dataset_with_empty_region, caplog):
        """When a region has no matching data, a warning is logged."""
        mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        with caplog.at_level(logging.WARNING, logger="mdt.tasks.plotting"):
            results = generate_plot(
                name="test_plot",
                plot_type="timeseries",
                input_data=dataset_with_empty_region,
                kwargs={
                    "savename": "output_{region}.png",
                    "regions": ["North America", "Antarctica", "Europe"],
                },
            )

        # Antarctica has no data, so only 2 plots should be produced
        assert len(results) == 2

        # Verify warning was logged for Antarctica
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Antarctica" in r.message and "no data" in r.message for r in warning_records)

    def test_all_regions_empty_returns_empty_list(self, mocker, caplog):
        """When all regions have no matching data, an empty list is returned."""
        # Dataset with only "Ocean" labels
        n = 4
        ds = xr.Dataset(
            {
                "temperature": ("x", np.random.rand(n)),
                "land": ("x", ["Ocean"] * n),
            },
            coords={
                "lat": ("x", np.linspace(-10, 10, n)),
                "lon": ("x", np.linspace(-20, 20, n)),
            },
        )

        mock_single = mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        with caplog.at_level(logging.WARNING, logger="mdt.tasks.plotting"):
            results = generate_plot(
                name="test_plot",
                plot_type="timeseries",
                input_data=ds,
                kwargs={
                    "savename": "output_{region}.png",
                    "regions": ["North America", "Europe"],
                },
            )

        assert results == []
        mock_single.assert_not_called()

        # Warnings should be logged for both regions
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 2


# ---------------------------------------------------------------------------
# Tests: Backward compatibility - no regions produces single plot (Requirement 2.7)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Test that no regions produces a single plot (backward compatibility)."""

    def test_no_regions_calls_single_plot(self, mocker, dataset_with_regions):
        """When no regions are specified, _generate_single_plot is called once."""
        mock_single = mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        result = generate_plot(
            name="test_plot",
            plot_type="timeseries",
            input_data=dataset_with_regions,
            kwargs={"savename": "output.png"},
        )

        # Should return a single result, not a list
        assert result == "plot_obj"
        mock_single.assert_called_once()

    def test_no_regions_key_in_kwargs(self, mocker, dataset_with_regions):
        """When kwargs has no 'regions' key at all, single plot behavior."""
        mock_single = mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        result = generate_plot(
            name="test_plot",
            plot_type="scatter",
            input_data=dataset_with_regions,
            kwargs={"savename": "scatter.png", "x": "temperature"},
        )

        assert result == "plot_obj"
        mock_single.assert_called_once()

    def test_regions_none_treated_as_no_regions(self, mocker, dataset_with_regions):
        """When regions is explicitly None, single plot behavior."""
        mock_single = mocker.patch(
            "mdt.tasks.plotting._generate_single_plot",
            return_value="plot_obj",
        )

        result = generate_plot(
            name="test_plot",
            plot_type="timeseries",
            input_data=dataset_with_regions,
            kwargs={"savename": "output.png", "regions": None},
        )

        assert result == "plot_obj"
        mock_single.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Region variable detection (Requirement 2.5)
# ---------------------------------------------------------------------------


class TestRegionVariableDetection:
    """Test that _find_region_variable raises ValueError when no region var exists."""

    def test_raises_when_no_region_variable(self, dataset_no_region_var):
        """ValueError raised when dataset has no string-typed data variable."""
        with pytest.raises(ValueError, match="No region label variable found"):
            _find_region_variable(dataset_no_region_var)

    def test_raises_when_not_dataset(self):
        """ValueError raised when input is not an xarray Dataset."""
        with pytest.raises(ValueError, match="Region filtering requires an xarray Dataset"):
            _find_region_variable(np.array([1, 2, 3]))

    def test_finds_region_variable(self, dataset_with_regions):
        """Correctly identifies the string-typed region variable."""
        var = _find_region_variable(dataset_with_regions)
        assert var == "land"

    def test_regions_specified_but_no_region_var_raises(self, mocker, dataset_no_region_var):
        """When regions are specified but dataset has no region var, ValueError is raised."""
        with pytest.raises(ValueError, match="No region label variable found"):
            generate_plot(
                name="test_plot",
                plot_type="timeseries",
                input_data=dataset_no_region_var,
                kwargs={
                    "savename": "output_{region}.png",
                    "regions": ["North America"],
                },
            )
