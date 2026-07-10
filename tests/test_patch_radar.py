"""Unit tests for scripts/patch_radar.py — RadarPlot metric computations.

Tests IOA, KGE, CCC correctness for known inputs, NaN for degenerate cases,
and idempotency of the patch script.

Requirements: 8.1, 8.2, 8.3, 8.5, 8.6
"""

import math
import os
import sys

import numpy as np
import pytest

# Import compute_radar_metrics from the patched monet_plots installation
from monet_plots.plots.radar import compute_radar_metrics

# Import patch utilities from the scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from patch_radar import RADAR_SOURCE, is_already_patched

# ---------------------------------------------------------------------------
# Test IOA, KGE, CCC computed correctly for known inputs (Req 8.1, 8.2, 8.3)
# ---------------------------------------------------------------------------


class TestPerfectPrediction:
    """When obs == pred, all metrics should be at their optimal values."""

    def test_perfect_prediction_ioa(self):
        """IOA should be 1.0 for perfect prediction."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_radar_metrics(obs, pred)
        assert result["IOA"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_prediction_kge(self):
        """KGE should be 1.0 for perfect prediction."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_radar_metrics(obs, pred)
        assert result["KGE"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_prediction_ccc(self):
        """CCC should be 1.0 for perfect prediction."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_radar_metrics(obs, pred)
        assert result["CCC"] == pytest.approx(1.0, abs=1e-10)


class TestKnownValues:
    """Test metrics against hand-computed known values."""

    def test_ioa_known_case(self):
        """IOA for a known obs/pred pair computed by hand.

        obs = [2, 4, 6, 8], pred = [3, 5, 7, 9]
        mean_obs = 5.0
        ss_res = (3-2)^2 + (5-4)^2 + (7-6)^2 + (9-8)^2 = 4
        denom terms: (|3-5|+|2-5|)^2 + (|5-5|+|4-5|)^2 + (|7-5|+|6-5|)^2 + (|9-5|+|8-5|)^2
                   = (2+3)^2 + (0+1)^2 + (2+1)^2 + (4+3)^2
                   = 25 + 1 + 9 + 49 = 84
        IOA = 1 - 4/84 = 1 - 1/21 ≈ 0.95238
        """
        obs = np.array([2.0, 4.0, 6.0, 8.0])
        pred = np.array([3.0, 5.0, 7.0, 9.0])
        result = compute_radar_metrics(obs, pred)
        assert result["IOA"] == pytest.approx(1.0 - 4.0 / 84.0, abs=1e-10)

    def test_kge_known_case(self):
        """KGE for a known obs/pred pair.

        obs = [1, 2, 3, 4, 5], pred = [2, 4, 6, 8, 10]
        r = 1.0 (perfect correlation)
        std_obs = sqrt(2.5) (ddof=1), std_pred = sqrt(10) (ddof=1)
        alpha = std_pred/std_obs = sqrt(10)/sqrt(2.5) = 2.0
        mean_obs = 3.0, mean_pred = 6.0
        beta = 6/3 = 2.0
        KGE = 1 - sqrt((1-1)^2 + (2-1)^2 + (2-1)^2) = 1 - sqrt(2) ≈ -0.41421
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = compute_radar_metrics(obs, pred)
        expected_kge = 1.0 - math.sqrt(2.0)
        assert result["KGE"] == pytest.approx(expected_kge, abs=1e-10)

    def test_ccc_known_case(self):
        """CCC for a known obs/pred pair.

        obs = [1, 2, 3, 4, 5], pred = [2, 4, 6, 8, 10]
        r = 1.0
        std_obs = sqrt(2.5), std_pred = sqrt(10)
        var_obs = 2.5, var_pred = 10.0
        mean_obs = 3.0, mean_pred = 6.0
        CCC = 2*1*sqrt(2.5)*sqrt(10) / (2.5 + 10 + (3-6)^2)
            = 2*sqrt(25) / (2.5 + 10 + 9)
            = 10 / 21.5 ≈ 0.46512
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = compute_radar_metrics(obs, pred)
        std_obs = np.std(obs, ddof=1)
        std_pred = np.std(pred, ddof=1)
        var_obs = std_obs**2
        var_pred = std_pred**2
        expected_ccc = 2.0 * 1.0 * std_obs * std_pred / (var_obs + var_pred + (3.0 - 6.0) ** 2)
        assert result["CCC"] == pytest.approx(expected_ccc, abs=1e-10)

    def test_all_three_keys_present(self):
        """Result dictionary always contains IOA, KGE, CCC keys."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.5, 2.5, 3.5, 4.5])
        result = compute_radar_metrics(obs, pred)
        assert set(result.keys()) == {"IOA", "KGE", "CCC"}

    def test_ioa_bounded_zero_one(self):
        """IOA should be in [0, 1] for non-degenerate inputs."""
        obs = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        pred = np.array([2.0, 3.0, 8.0, 1.0, 6.0])
        result = compute_radar_metrics(obs, pred)
        assert 0.0 <= result["IOA"] <= 1.0

    def test_ccc_bounded_neg1_pos1(self):
        """CCC should be in [-1, 1] for non-degenerate inputs."""
        obs = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        pred = np.array([2.0, 3.0, 8.0, 1.0, 6.0])
        result = compute_radar_metrics(obs, pred)
        assert -1.0 <= result["CCC"] <= 1.0

    def test_kge_at_most_one(self):
        """KGE should be <= 1 for non-degenerate inputs."""
        obs = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        pred = np.array([2.0, 3.0, 8.0, 1.0, 6.0])
        result = compute_radar_metrics(obs, pred)
        assert result["KGE"] <= 1.0


# ---------------------------------------------------------------------------
# Test NaN returned for degenerate inputs (Req 8.5)
# ---------------------------------------------------------------------------


class TestDegenerateCases:
    """Degenerate inputs should return NaN for all metrics."""

    def test_single_element_returns_nan(self):
        """Fewer than 2 valid points → all NaN."""
        obs = np.array([5.0])
        pred = np.array([3.0])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_empty_arrays_returns_nan(self):
        """Empty arrays → all NaN."""
        obs = np.array([])
        pred = np.array([])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_all_nan_returns_nan(self):
        """All NaN values → fewer than 2 valid points → all NaN."""
        obs = np.array([np.nan, np.nan, np.nan])
        pred = np.array([np.nan, np.nan, np.nan])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_one_valid_point_after_nan_removal(self):
        """Only 1 valid point after NaN removal → all NaN."""
        obs = np.array([1.0, np.nan, np.nan])
        pred = np.array([2.0, np.nan, np.nan])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_zero_variance_obs_returns_nan(self):
        """Zero variance in observed array → all NaN."""
        obs = np.array([3.0, 3.0, 3.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_zero_variance_pred_returns_nan(self):
        """Zero variance in predicted array → all NaN."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([5.0, 5.0, 5.0, 5.0])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_zero_variance_both_returns_nan(self):
        """Zero variance in both arrays → all NaN."""
        obs = np.array([7.0, 7.0, 7.0])
        pred = np.array([7.0, 7.0, 7.0])
        result = compute_radar_metrics(obs, pred)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])

    def test_nan_in_one_array_reduces_valid_count(self):
        """NaN in pred reduces valid count; if < 2 remain, return NaN."""
        obs = np.array([1.0, 2.0])
        pred = np.array([np.nan, 3.0])
        result = compute_radar_metrics(obs, pred)
        # Only 1 valid pair: (2.0, 3.0)
        assert math.isnan(result["IOA"])
        assert math.isnan(result["KGE"])
        assert math.isnan(result["CCC"])


# ---------------------------------------------------------------------------
# Test idempotency: running patch twice does not duplicate metrics (Req 8.6)
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Patch script should detect already-patched state and skip."""

    def test_is_already_patched_detects_all_markers(self, tmp_path):
        """is_already_patched returns True when IOA, KGE, CCC markers present."""
        radar_file = tmp_path / "radar.py"
        radar_file.write_text(RADAR_SOURCE)
        assert is_already_patched(str(radar_file)) is True

    def test_is_already_patched_false_when_missing(self, tmp_path):
        """is_already_patched returns False when file doesn't contain all markers."""
        radar_file = tmp_path / "radar.py"
        radar_file.write_text("# empty radar module\n")
        assert is_already_patched(str(radar_file)) is False

    def test_is_already_patched_false_when_file_missing(self, tmp_path):
        """is_already_patched returns False when file doesn't exist."""
        assert is_already_patched(str(tmp_path / "nonexistent.py")) is False

    def test_is_already_patched_partial_markers(self, tmp_path):
        """is_already_patched returns False when only some markers present."""
        radar_file = tmp_path / "radar.py"
        radar_file.write_text('"IOA"\n"KGE"\n# no CCC marker\n')
        assert is_already_patched(str(radar_file)) is False

    def test_radar_source_contains_all_metric_keys(self):
        """RADAR_SOURCE template contains all three metric keys."""
        assert '"IOA"' in RADAR_SOURCE
        assert '"KGE"' in RADAR_SOURCE
        assert '"CCC"' in RADAR_SOURCE

    def test_compute_radar_metrics_idempotent_result(self):
        """Calling compute_radar_metrics twice with same input gives same result."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result1 = compute_radar_metrics(obs, pred)
        result2 = compute_radar_metrics(obs, pred)
        assert result1 == result2
