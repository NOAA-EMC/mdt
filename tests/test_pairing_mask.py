"""Unit tests for pairing mask application in pair_data.

Tests verify that:
- query_mask is called with correct arguments when mask is present
- query_mask is NOT called when mask is absent
- Exceptions from query_mask are logged at ERROR level and re-raised

Requirements: 1.1, 1.2, 1.3, 1.4
"""

import logging

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def dummy_paired_dataset():
    """Create a dummy xr.Dataset that monet.util.combinetool.pair would return."""
    lat = np.linspace(-10, 10, 5)
    lon = np.linspace(-20, 20, 5)
    return xr.Dataset(
        {"temperature": (("x",), np.random.rand(5))},
        coords={"lat": ("x", lat), "lon": ("x", lon)},
    )


@pytest.fixture
def mock_pair(mocker, dummy_paired_dataset):
    """Mock monet.util.combinetool.pair to return the dummy dataset."""
    return mocker.patch(
        "monet.util.combinetool.pair",
        return_value=dummy_paired_dataset,
    )


class TestPairingMaskApplication:
    """Tests for mask application in pair_data (Requirement 1.1, 1.2)."""

    def test_query_mask_called_with_correct_args(self, mocker, mock_pair, dummy_paired_dataset):
        """When mask='land', query_mask is called with (paired_data, 'land')."""
        from mdt.tasks.pairing import pair_data

        # Create a masked dataset (simulating what query_mask returns)
        masked_ds = dummy_paired_dataset.copy()
        masked_ds["land"] = ("x", ["North America"] * 5)

        mock_query_mask = mocker.patch(
            "monet.util.mask.query_mask",
            return_value=masked_ds,
        )

        result = pair_data(
            name="test_pair",
            method="bilinear",
            source_data=dummy_paired_dataset,
            target_data=dummy_paired_dataset,
            kwargs={},
            mask="land",
        )

        # Verify query_mask was called with the paired data and mask name
        mock_query_mask.assert_called_once_with(dummy_paired_dataset, "land")

        # Verify the result contains the region label variable
        assert "land" in result.data_vars

    def test_query_mask_not_called_when_mask_absent(self, mocker, mock_pair, dummy_paired_dataset):
        """When mask=None (default), query_mask is NOT called."""
        from mdt.tasks.pairing import pair_data

        mock_query_mask = mocker.patch(
            "monet.util.mask.query_mask",
        )

        pair_data(
            name="test_pair",
            method="bilinear",
            source_data=dummy_paired_dataset,
            target_data=dummy_paired_dataset,
            kwargs={},
            mask=None,
        )

        mock_query_mask.assert_not_called()

    def test_query_mask_not_called_when_mask_not_provided(self, mocker, mock_pair, dummy_paired_dataset):
        """When mask parameter is not provided (defaults to None), query_mask is NOT called."""
        from mdt.tasks.pairing import pair_data

        mock_query_mask = mocker.patch(
            "monet.util.mask.query_mask",
        )

        pair_data(
            name="test_pair",
            method="bilinear",
            source_data=dummy_paired_dataset,
            target_data=dummy_paired_dataset,
            kwargs={},
        )

        mock_query_mask.assert_not_called()


class TestPairingMaskExceptionHandling:
    """Tests for exception handling when query_mask fails (Requirement 1.3)."""

    def test_exception_logged_and_reraised(self, mocker, mock_pair, dummy_paired_dataset, caplog):
        """When query_mask raises, the error is logged at ERROR level and re-raised."""
        from mdt.tasks.pairing import pair_data

        error_msg = "Unknown mask name: 'invalid_mask'"
        mock_query_mask = mocker.patch(
            "monet.util.mask.query_mask",
            side_effect=ValueError(error_msg),
        )

        with caplog.at_level(logging.ERROR, logger="mdt.tasks.pairing"):
            with pytest.raises(ValueError, match="Unknown mask name"):
                pair_data(
                    name="test_pair",
                    method="bilinear",
                    source_data=dummy_paired_dataset,
                    target_data=dummy_paired_dataset,
                    kwargs={},
                    mask="invalid_mask",
                )

        # Verify error was logged
        assert any("Failed to apply mask" in record.message for record in caplog.records)
        assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_source_and_target_duplicate_time_values_are_deduplicated_before_pairing(mocker):
    """Duplicate time values in source/target are dropped before monet pairing."""
    from mdt.tasks.pairing import pair_data

    source = xr.Dataset(
        {"mod": ("time", [1.0, 1.5, 2.0])},
        coords={"time": ["2023-08-01T00:00:00", "2023-08-01T00:00:00", "2023-08-01T01:00:00"]},
    )
    target = xr.Dataset(
        {"obs": ("time", [10.0, 11.0, 12.0])},
        coords={"time": ["2023-08-01T00:00:00", "2023-08-01T00:00:00", "2023-08-01T01:00:00"]},
    )

    mock_pair = mocker.patch("monet.util.combinetool.pair", return_value=source)

    pair_data(
        name="test_pair_dupe_time",
        method="nearest",
        source_data=source,
        target_data=target,
        kwargs={},
    )

    called_source = mock_pair.call_args[0][0]
    called_target = mock_pair.call_args[0][1]

    assert isinstance(called_source, xr.Dataset)
    assert called_source.indexes["time"].is_unique
    assert called_source.sizes["time"] == 2

    assert isinstance(called_target, xr.Dataset)
    assert called_target.indexes["time"].is_unique
    assert called_target.sizes["time"] == 2

    def test_runtime_error_logged_and_reraised(self, mocker, mock_pair, dummy_paired_dataset, caplog):
        """Non-ValueError exceptions from query_mask are also logged and re-raised."""
        from mdt.tasks.pairing import pair_data

        mock_query_mask = mocker.patch(
            "monet.util.mask.query_mask",
            side_effect=RuntimeError("Network error fetching mask data"),
        )

        with caplog.at_level(logging.ERROR, logger="mdt.tasks.pairing"):
            with pytest.raises(RuntimeError, match="Network error"):
                pair_data(
                    name="test_pair",
                    method="bilinear",
                    source_data=dummy_paired_dataset,
                    target_data=dummy_paired_dataset,
                    kwargs={},
                    mask="land",
                )

        # Verify error was logged
        assert any("Failed to apply mask" in record.message for record in caplog.records)
        assert any(record.levelno == logging.ERROR for record in caplog.records)
