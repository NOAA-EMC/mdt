import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import pandas as pd
import xarray as xr

from mdt.tasks.plotting import _filter_by_region, _find_region_variable, _is_empty
from mdt.utils import discover_spatial_dims, update_history

logger = logging.getLogger(__name__)


def compute_statistics(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute statistics on paired data using monet-stats with Dask support.

    Adheres to the Aero Protocol by calling metrics directly on Xarray
    objects, allowing the backend (NumPy or Dask) to handle the data
    without forcing re-chunking or memory-intensive operations.

    When a ``regions`` list is present in kwargs, computes metrics separately
    for each region by filtering the input data and substituting ``{region}``
    in the savename. Spaces in region names are replaced with underscores.

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : List[str]
        A list of metric names to compute (e.g., ['rmse', 'bias', 'corr']).
    input_data : xr.Dataset or xr.DataArray or pd.DataFrame
        The paired dataset containing both model and observational data.
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to the monet_stats metric functions.
        Common keys: `obs_var`, `mod_var`, `weights`, `dim`.

    Returns
    -------
    Dict[str, Any]
        When regions are specified, a dictionary mapping region names to their
        per-region results dictionaries. Otherwise, a dictionary mapping metric
        names to their computed results.

    Raises
    ------
    ImportError
        If monet-stats is not installed.
    ValueError
        If regions are specified but savename does not contain ``{region}``
        placeholder, or if no region label variable is found in the dataset.

    Examples
    --------
    >>> metrics = ['rmse', 'mb']
    >>> kwargs = {'obs_var': 'obs', 'mod_var': 'mod', 'weights': 'w'}
    >>> results = compute_statistics('test_stats', metrics, ds, kwargs)
    """
    regions = kwargs.pop("regions", None)

    if regions:
        savename_template = kwargs.get("savename", "")
        if "{region}" not in savename_template:
            raise ValueError(f"Statistics '{name}': savename must contain '{{region}}' placeholder when regions are specified.")
        region_var = _find_region_variable(input_data)
        all_results = {}
        for region in regions:
            filtered = _filter_by_region(input_data, region_var, region)
            if _is_empty(filtered):
                logger.warning("Statistics '%s': region '%s' has no data, skipping.", name, region)
                continue
            region_savename = savename_template.replace("{region}", region.replace(" ", "_"))
            region_kwargs = {**kwargs, "savename": region_savename}
            region_kwargs.pop("regions", None)
            results = _compute_single(name, metrics, filtered, region_kwargs)
            all_results[region] = results
        return all_results
    else:
        return _compute_single(name, metrics, input_data, kwargs)


def _compute_single(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: Dict[str, Any],
) -> Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]:
    """
    Compute statistics for a single dataset (no region filtering).

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : List[str]
        A list of metric names to compute.
    input_data : xr.Dataset or xr.DataArray or pd.DataFrame
        The data to compute metrics on.
    kwargs : Dict[str, Any]
        Additional keyword arguments for metric functions.

    Returns
    -------
    Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]
        A dictionary mapping metric names to their results.
    """
    logger.info("Computing statistics '%s' for metrics: %s", name, metrics)

    try:
        import monet_stats

        results = {}
        for metric_name in metrics:
            logger.debug("Processing metric: %s", metric_name)
            metric_func = _find_metric(monet_stats, metric_name)

            if not metric_func:
                logger.warning("Metric '%s' not found in monet_stats. Skipping.", metric_name)
                continue

            try:
                result = _execute_metric(input_data, metric_func, kwargs)

                # Provenance Tracking (Aero Protocol Rule 2.3)
                msg = f"Computed {metric_name} with params: {kwargs}"
                result = update_history(result, msg)

                results[metric_name] = cast(Union[xr.Dataset, xr.DataArray, pd.Series], result)
            except Exception as e:
                logger.error("Failed to compute %s: %s", metric_name, e)
                raise

        logger.info("Successfully computed statistics '%s'", name)

        # Save results to file if savename is specified
        savename = kwargs.get("savename")
        if savename:
            _save_statistics(name, results, savename)

        return results

    except ImportError:
        logger.error("monet-stats is not installed. Please install it to use this task.")
        raise


def _find_metric(module: Any, metric_name: str) -> Any:
    """
    Robust discovery of metric functions in monet-stats.

    Parameters
    ----------
    module : Any
        The module to search for the metric (usually monet_stats).
    metric_name : str
        The name of the metric to find (case-insensitive).

    Returns
    -------
    Any or None
        The metric function if found, else None.
    """
    # 1. Alias Handling (Aero Protocol Standard)
    aliases = {"BIAS": "mb", "MBIAS": "mb"}
    upper_name = metric_name.upper()
    if upper_name in aliases:
        target_name = aliases[upper_name]
        logger.debug(f"Alias found: mapping '{metric_name}' to '{target_name}'")
    else:
        target_name = metric_name

    # 2. Direct lookup
    if hasattr(module, target_name):
        logger.debug(f"Found metric '{target_name}' via direct lookup.")
        return getattr(module, target_name)

    # 3. Case-insensitive search
    for attr in dir(module):
        if attr.lower() == target_name.lower():
            logger.debug(f"Found metric '{target_name}' via case-insensitive search as '{attr}'.")
            return getattr(module, attr)

    # 4. Search in sub-module
    if hasattr(module, "stats"):
        for attr in dir(module.stats):
            if attr.lower() == target_name.lower():
                logger.debug(f"Found metric '{target_name}' in 'stats' submodule as '{attr}'.")
                return getattr(module.stats, attr)

    try:
        mod_name = module.__name__
    except AttributeError:
        mod_name = str(module)

    logger.warning(f"Metric '{target_name}' could not be found in {mod_name}")
    return None


def _execute_metric(
    data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    func: Any,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, xr.DataArray, pd.Series]:
    """
    Execute metric from monet-stats with native Xarray/Dask support.

    This function orchestrates the execution by resolving weights and
    dimensions, and checking for native backend support via inspection.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray or pd.DataFrame
        The data object to process.
    func : Any
        The metric function to execute.
    kwargs : Dict[str, Any]
        Parameters for the metric execution, including obs/mod variable names
        and weighting options.

    Returns
    -------
    xr.Dataset or xr.DataArray or pd.Series
        The computed metric result.

    Notes
    -----
    Aero Protocol: Preserves Dask laziness and ensures backend-agnostic behavior.
    """
    obs_var = kwargs.get("obs_var", "obs")
    mod_var = kwargs.get("mod_var", "mod")
    weights = kwargs.get("weights")
    dim = kwargs.get("dim")

    # Filter out MDT-specific keys for the standard metric call
    call_kwargs = {k: v for k, v in kwargs.items() if k not in ["obs_var", "mod_var", "weights", "dim", "savename"]}

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        # 1. Extract Target Data
        if isinstance(data, xr.Dataset):
            target_obs = data[obs_var]
            target_mod = data[mod_var]
        else:
            target_obs = None  # Not used for single DataArray metrics
            target_mod = data

        # 2. Resolve Weights and Dimensions
        w = None
        if weights is not None:
            if isinstance(weights, str):
                if isinstance(data, xr.Dataset) and weights in data:
                    w = data[weights]
                elif isinstance(data, xr.DataArray) and weights in data.coords:
                    w = data.coords[weights]
            else:
                w = weights

        # Resolve axis/dim for monet-stats (standardizing on 'axis' for metrics)
        axis = dim
        if axis is None:
            # Try to discover spatial dims if none provided
            lat_dim, lon_dim = discover_spatial_dims(data)
            axis = [d for d in [lat_dim, lon_dim] if d is not None]
            if not axis:
                axis = None

        # Check function signature for native support
        try:
            sig = inspect.signature(func)
            has_weights = "weights" in sig.parameters
            axis_param = "axis" if "axis" in sig.parameters else ("dim" if "dim" in sig.parameters else None)
        except (ValueError, TypeError):
            # Fallback for built-ins or functions without signatures
            has_weights = False
            axis_param = None

        # 3. Weighted Logic (Aero Protocol + monet-stats backend)
        if w is not None:
            if has_weights:
                # Direct call if metric supports weights natively
                if axis_param:
                    call_kwargs[axis_param] = axis
                if target_obs is not None:
                    return cast(Union[xr.Dataset, xr.DataArray], func(target_obs, target_mod, weights=w, **call_kwargs))
                return cast(Union[xr.Dataset, xr.DataArray], func(target_mod, weights=w, **call_kwargs))

            # MDT Architecture Rule: Rely solely on monet-stats for statistical computations.
            # Manual fallbacks have been removed to ensure scientific consistency
            # and to minimize maintenance overhead.
            metric_name = getattr(func, "__name__", "")
            logger.warning(
                "Metric '%s' does not support weights natively. Using unweighted.",
                metric_name,
            )

        # 4. Standard Fallback (Unweighted or natively supported axis)
        if axis_param:
            call_kwargs[axis_param] = axis

        if target_obs is not None:
            return cast(Union[xr.Dataset, xr.DataArray], func(target_obs, target_mod, **call_kwargs))
        return cast(Union[xr.Dataset, xr.DataArray], func(target_mod, **call_kwargs))

    # data must be pd.DataFrame based on the Union type hint
    obs = data[obs_var]
    mod = data[mod_var]

    if weights is not None:
        # Check if the function supports weights (Pandas-compatible)
        try:
            sig = inspect.signature(func)
            if "weights" in sig.parameters:
                weights_val = data[weights] if isinstance(weights, str) else weights
                return cast(pd.Series, func(obs, mod, weights=weights_val, **call_kwargs))
        except (ValueError, TypeError):
            pass

    return cast(pd.Series, func(obs, mod, **call_kwargs))


def _save_statistics(name: str, results: Dict[str, Any], savename: str) -> None:
    """Save computed statistics to a file (CSV or Markdown).

    Parameters
    ----------
    name : str
        The statistics task name.
    results : dict
        Mapping of metric names to computed values.
    savename : str
        Output file path. Extension determines format:
        - .csv → CSV file
        - .md → Markdown table
        - .json → JSON file
        - anything else → Markdown table
    """
    import numpy as np

    # Extract scalar values from results
    rows = {}
    for metric, value in results.items():
        if hasattr(value, "values"):
            v = value.values
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            elif isinstance(v, np.ndarray):
                v = float(v.mean())
        elif hasattr(value, "item"):
            v = value.item()
        else:
            v = value
        rows[metric] = v

    df = pd.DataFrame([rows], index=[name])

    if savename.endswith(".csv"):
        df.to_csv(savename)
        logger.info("Saved statistics to %s (CSV)", savename)
    elif savename.endswith(".json"):
        df.to_json(savename, indent=2)
        logger.info("Saved statistics to %s (JSON)", savename)
    else:
        # Default: Markdown table
        md = f"# Statistics: {name}\n\n"
        md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in rows.items():
            md += f"| {metric} | {value:.4f} |\n"
        with Path(savename).open("w") as f:
            f.write(md)
        logger.info("Saved statistics to %s (Markdown)", savename)
