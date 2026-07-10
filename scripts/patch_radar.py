#!/usr/bin/env python
"""Patch monet-plots to add RadarPlot with IOA, KGE, and CCC metrics.

Creates a RadarPlot class in monet_plots/plots/radar.py that provides
compute_radar_metrics with:
- IOA (Index of Agreement, Willmott 1981)
- KGE (Kling-Gupta Efficiency, Gupta 2009)
- CCC (Concordance Correlation Coefficient, Lin 1989)

Returns NaN for degenerate inputs (zero variance, < 2 valid points).
Idempotent: skips if radar.py already contains IOA/KGE/CCC.
"""

import os
import sys


def get_radar_path() -> str:
    """Locate the monet_plots/plots/ directory via the installed package."""
    try:
        import monet_plots
    except ImportError:
        print("ERROR: monet_plots is not installed in this environment.")
        sys.exit(1)

    plots_dir = os.path.join(os.path.dirname(monet_plots.__file__), "plots")
    if not os.path.isdir(plots_dir):
        print(f"ERROR: plots directory not found at {plots_dir}")
        sys.exit(1)

    return os.path.join(plots_dir, "radar.py")


def is_already_patched(radar_path: str) -> bool:
    """Check if IOA/KGE/CCC metrics already exist in the radar module."""
    if not os.path.exists(radar_path):
        return False
    with open(radar_path) as f:
        content = f.read()
    return all(marker in content for marker in ['"IOA"', '"KGE"', '"CCC"'])


RADAR_SOURCE = '''\
# src/monet_plots/plots/radar.py
"""Radar (spider) plot for multi-metric model evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BasePlot


def compute_radar_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """Compute verification metrics for radar plot display.

    Computes IOA, KGE, and CCC from paired observed/predicted arrays.
    Returns NaN for degenerate inputs (zero variance or < 2 valid points).

    Parameters
    ----------
    obs : np.ndarray
        Observed values (may contain NaN).
    pred : np.ndarray
        Predicted/model values (may contain NaN).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys "IOA", "KGE", "CCC" and their computed values.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # Mask out NaN pairs
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs_v = obs[valid]
    pred_v = pred[valid]

    n = len(obs_v)

    # Degenerate case: fewer than 2 valid points
    if n < 2:
        return {"IOA": float("nan"), "KGE": float("nan"), "CCC": float("nan")}

    mean_obs = np.mean(obs_v)
    mean_pred = np.mean(pred_v)
    std_obs = np.std(obs_v, ddof=1)
    std_pred = np.std(pred_v, ddof=1)
    var_obs = std_obs ** 2
    var_pred = std_pred ** 2

    # Degenerate case: zero variance in either array
    if std_obs == 0.0 or std_pred == 0.0:
        return {"IOA": float("nan"), "KGE": float("nan"), "CCC": float("nan")}

    # Pearson correlation coefficient
    r = np.corrcoef(obs_v, pred_v)[0, 1]

    # --- IOA (Index of Agreement, Willmott 1981) ---
    # IOA = 1 - sum((pred - obs)^2) / sum((|pred - mean_obs| + |obs - mean_obs|)^2)
    ss_res = np.sum((pred_v - obs_v) ** 2)
    ss_denom = np.sum((np.abs(pred_v - mean_obs) + np.abs(obs_v - mean_obs)) ** 2)
    if ss_denom == 0.0:
        ioa = float("nan")
    else:
        ioa = 1.0 - ss_res / ss_denom
        # Bound to [0, 1]
        ioa = float(np.clip(ioa, 0.0, 1.0))

    # --- KGE (Kling-Gupta Efficiency, Gupta 2009) ---
    # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    # alpha = std_pred / std_obs, beta = mean_pred / mean_obs
    alpha = std_pred / std_obs
    if mean_obs == 0.0:
        beta = float("nan")
        kge = float("nan")
    else:
        beta = mean_pred / mean_obs
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    # --- CCC (Concordance Correlation Coefficient, Lin 1989) ---
    # CCC = 2*r*std_obs*std_pred / (var_obs + var_pred + (mean_obs - mean_pred)^2)
    ccc_denom = var_obs + var_pred + (mean_obs - mean_pred) ** 2
    if ccc_denom == 0.0:
        ccc = float("nan")
    else:
        ccc = float(2.0 * r * std_obs * std_pred / ccc_denom)
        # Bound to [-1, 1]
        ccc = float(np.clip(ccc, -1.0, 1.0))

    return {"IOA": ioa, "KGE": kge, "CCC": ccc}


class RadarPlot(BasePlot):
    """Radar (spider) plot for multi-metric model evaluation.

    Displays multiple verification metrics on a polar axis, allowing
    visual comparison of model performance across different measures.
    """

    def __init__(
        self,
        data: Any,
        *,
        obs_col: Optional[str] = None,
        mod_col: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        """Initialize RadarPlot.

        Parameters
        ----------
        data : Any
            Input data (DataFrame, DataArray, Dataset, or ndarray).
        obs_col : str, optional
            Column/variable name for observations.
        mod_col : str, optional
            Column/variable name for model predictions.
        metrics : Dict[str, float], optional
            Pre-computed metrics dictionary. If provided, obs_col/mod_col
            are not required.
        **kwargs : Any
            Arguments passed to BasePlot.
        """
        if "subplot_kw" not in kwargs:
            kwargs["subplot_kw"] = {"projection": "polar"}
        elif "projection" not in kwargs["subplot_kw"]:
            kwargs["subplot_kw"]["projection"] = "polar"

        super().__init__(**kwargs)
        self.data = data
        self.obs_col = obs_col
        self.mod_col = mod_col

        if metrics is not None:
            self.metrics = metrics
        elif obs_col is not None and mod_col is not None:
            self.metrics = self._compute_metrics()
        else:
            self.metrics = {}

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute radar metrics from data columns."""
        import pandas as pd

        if hasattr(self.data, "to_dataframe"):
            df = self.data.to_dataframe().reset_index()
        elif isinstance(self.data, pd.DataFrame):
            df = self.data
        else:
            raise ValueError("Data must be a DataFrame or xarray object.")

        obs = df[self.obs_col].values
        pred = df[self.mod_col].values
        return compute_radar_metrics(obs, pred)

    def plot(self, **kwargs: Any):
        """Generate the radar plot.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments for styling.

        Returns
        -------
        matplotlib.axes.Axes
            The polar axes with the radar plot.
        """
        if not self.metrics:
            return self.ax

        labels = list(self.metrics.keys())
        values = [self.metrics[k] for k in labels]
        n = len(labels)

        # Compute angles for each metric
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        # Close the polygon
        values = values + [values[0]]
        angles = angles + [angles[0]]

        self.ax.plot(angles, values, "o-", linewidth=2, **kwargs)
        self.ax.fill(angles, values, alpha=0.25)
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(labels)
        self.ax.set_ylim(0, 1)

        return self.ax
'''


def apply_patch() -> None:
    """Create or update the radar.py module in monet_plots."""
    radar_path = get_radar_path()

    if is_already_patched(radar_path):
        print("Already patched")
        return

    with open(radar_path, "w") as f:
        f.write(RADAR_SOURCE)

    print(f"✓ Created {radar_path} with RadarPlot (IOA, KGE, CCC metrics)")


if __name__ == "__main__":
    apply_patch()
