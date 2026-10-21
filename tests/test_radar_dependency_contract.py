"""The pinned plotting library exposes normalized Dataset radar metrics."""

import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.radar import compute_radar_metrics


def test_current_radar_dataset_contract():
    """Perfect predictions score one with explicit current metric names."""
    observations = np.arange(1.0, 6.0)
    result = compute_radar_metrics(observations, observations, metrics=["R", "D1", "KGE", "CCC"])
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"R", "D1", "KGE", "CCC"}
    for metric in result.data_vars:
        assert float(result[metric]) == pytest.approx(1.0)
