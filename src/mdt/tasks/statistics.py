import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import pandas as pd
import xarray as xr

from mdt.tasks.plotting import _filter_by_region, _find_region_variable, _is_empty
from mdt.utils import discover_spatial_dims, to_xarray, update_history

logger = logging.getLogger(__name__)


def compute_statistics(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray],
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
    input_data : xr.Dataset or xr.DataArray
        The paired dataset containing both model and observational data.
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to the monet_stats metric functions.
        Common keys: `obs_var`, `mod_var`, `weights`, `dim`.
        Requested weights are required unless `allow_unweighted_fallback=True`
        explicitly permits an unsupported metric to run unweighted. Missing
        named weight variables always raise an error.

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
    kwargs = dict(kwargs)
    input_data = to_xarray(input_data)
    regions = kwargs.pop("regions", None)

    model_dim = kwargs.pop("model_dim", input_data.attrs.get("mdt_model_dim", "model") if isinstance(input_data, xr.Dataset) else "model")
    if isinstance(input_data, xr.Dataset) and model_dim in input_data.dims:
        results = {}
        for model in input_data[model_dim].values.tolist():
            model_kwargs = dict(kwargs)
            if model_kwargs.get("savename"):
                path = Path(model_kwargs["savename"])
                model_kwargs["savename"] = str(path.with_name(f"{path.stem}_{model}{path.suffix}"))
            model_data = input_data.sel({model_dim: model})
            results[str(model)] = compute_statistics(
                name=f"{name}_{model}", metrics=metrics, input_data=model_data, kwargs={**model_kwargs, **({"regions": regions} if regions else {})}
            )
        return results

    if regions:
        if not isinstance(input_data, xr.Dataset):
            raise ValueError("Region filtering requires an xarray Dataset.")
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
            region_results = _compute_single(name, metrics, filtered, region_kwargs)
            all_results[region] = region_results
        return all_results
    else:
        return _compute_single(name, metrics, input_data, kwargs)


def _compute_single(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray],
    kwargs: Dict[str, Any],
) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
    """
    Compute statistics for a single dataset (no region filtering).

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : List[str]
        A list of metric names to compute.
    input_data : xr.Dataset or xr.DataArray
        The data to compute metrics on.
    kwargs : Dict[str, Any]
        Additional keyword arguments for metric functions.

    Returns
    -------
    Dict[str, Union[xr.Dataset, xr.DataArray]]
        A dictionary mapping metric names to their results.
    """
    logger.info("Computing statistics '%s' for metrics: %s", name, metrics)
    input_data = to_xarray(input_data)

    try:
        import monet_stats

        resolved_metrics = [(metric_name, _find_metric(monet_stats, metric_name)) for metric_name in metrics]
        missing = [metric_name for metric_name, metric_func in resolved_metrics if not metric_func]
        if missing:
            raise ValueError(f"Unknown monet_stats metric(s): {', '.join(missing)}")

        results = {}
        for metric_name, metric_func in resolved_metrics:
            logger.debug("Processing metric: %s", metric_name)

            try:
                result = _execute_metric(input_data, metric_func, kwargs)
                result = to_xarray(result)

                # Provenance Tracking (Aero Protocol Rule 2.3)
                msg = f"Computed {metric_name} with params: {kwargs}"
                result = update_history(result, msg)

                results[metric_name] = cast(Union[xr.Dataset, xr.DataArray], result)
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
    data: Union[xr.Dataset, xr.DataArray],
    func: Any,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Execute metric from monet-stats with native Xarray/Dask support.

    This function orchestrates the execution by resolving weights and
    dimensions, and checking for native backend support via inspection.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The data object to process.
    func : Any
        The metric function to execute.
    kwargs : Dict[str, Any]
        Parameters for the metric execution, including obs/mod variable names
        and weighting options.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The computed metric result.

    Notes
    -----
    Aero Protocol: Preserves Dask laziness and ensures backend-agnostic behavior.
    """
    obs_var = kwargs.get("obs_var", "obs")
    mod_var = kwargs.get("mod_var", "mod")
    weights = kwargs.get("weights")
    dim = kwargs.get("dim")
    allow_unweighted_fallback = kwargs.get("allow_unweighted_fallback", False)
    if not isinstance(allow_unweighted_fallback, bool):
        raise ValueError("allow_unweighted_fallback must be a boolean.")

    # Filter out MDT-specific keys for the standard metric call
    call_kwargs = {k: v for k, v in kwargs.items() if k not in ["obs_var", "mod_var", "weights", "dim", "savename", "allow_unweighted_fallback"]}

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
                    raise ValueError(f"Requested weight variable '{weights}' was not found in the input data.")
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
        fallback_message = None
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
            if not allow_unweighted_fallback:
                raise ValueError(
                    f"Metric '{metric_name}' does not support requested weights. "
                    "Choose a weighted metric or explicitly set allow_unweighted_fallback: true."
                )
            fallback_message = (
                f"Metric '{metric_name}' computed unweighted because it does not support requested weights; allow_unweighted_fallback=True."
            )
            logger.warning(fallback_message)

        # 4. Standard Fallback (Unweighted or natively supported axis)
        if axis_param:
            call_kwargs[axis_param] = axis

        if target_obs is not None:
            result = func(target_obs, target_mod, **call_kwargs)
        else:
            result = func(target_mod, **call_kwargs)
        if fallback_message:
            result = update_history(to_xarray(result), fallback_message)
        return cast(Union[xr.Dataset, xr.DataArray], result)

    raise TypeError(f"Statistics require xarray input, got {type(data).__name__}.")


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
        if isinstance(value, xr.Dataset):
            value = value.to_array().mean(skipna=True)
        if isinstance(value, xr.DataArray):
            v = value.mean(skipna=True).compute().item()
        elif hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
            v = value.item()
        elif hasattr(value, "item"):
            v = np.asarray(value).mean().item()
        elif isinstance(value, np.ndarray):
            v = value.mean().item()
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
