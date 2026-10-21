import logging
import threading
from typing import Any, Dict, Optional, Union, cast

import pandas as pd
import xarray as xr

from mdt.utils import to_xarray, update_history

# Monet pairing is usually handled by `monet.models.*` or `monet.obs.*` depending on the object,
# or through `monet.util.interp_util` regridding.

logger = logging.getLogger(__name__)
_esmf_thread_state = threading.local()


def _ensure_esmf_manager() -> None:
    """Initialize ESMF manager in the current worker thread when available."""
    if getattr(_esmf_thread_state, "manager_ready", False):
        return

    try:
        import esmpy
    except ImportError:
        return

    esmpy.Manager(debug=False)
    _esmf_thread_state.manager_ready = True


def _harmonize_spatial_coordinates(
    data: Union[xr.Dataset, xr.DataArray],
) -> Union[xr.Dataset, xr.DataArray]:
    """Normalize spatial coordinate aliases on paired xarray outputs."""
    out: Union[xr.Dataset, xr.DataArray] = data

    has_lat_lon = "lat" in out.coords and "lon" in out.coords
    has_latitude_longitude = "latitude" in out.coords and "longitude" in out.coords

    # Prefer canonical 1D lat/lon axes when available.
    if has_lat_lon and has_latitude_longitude:
        lat = out.coords["lat"]
        lon = out.coords["lon"]
        latitude = out.coords["latitude"]
        longitude = out.coords["longitude"]
        if lat.ndim == 1 and lon.ndim == 1 and latitude.ndim == 2 and longitude.ndim == 2:
            out = out.drop_vars(["latitude", "longitude"])
            return out

    # Provide lat/lon aliases when only latitude/longitude exist as 1D coords.
    if not has_lat_lon and has_latitude_longitude:
        latitude = out.coords["latitude"]
        longitude = out.coords["longitude"]
        if latitude.ndim == 1 and longitude.ndim == 1:
            out = out.assign_coords(lat=latitude, lon=longitude)

    return out


def _drop_duplicate_time_entries(
    data: Union[xr.Dataset, xr.DataArray],
    name: str,
    role: str,
) -> Union[xr.Dataset, xr.DataArray]:
    """Drop duplicate time entries that break xarray alignment during pairing."""
    # Never drop duplicate time entries for point observation datasets (non-gridded)
    if "y" not in data.dims and "x" not in data.dims:
        return data

    if "time" in data.indexes:
        time_index = data.indexes["time"]
        dup_count = int(time_index.duplicated().sum())
        if dup_count > 0:
            if isinstance(data, xr.Dataset) and {"forecast_reference_time", "lead_time"}.issubset(data.coords):
                logger.info(
                    "Dataset '%s' has %d repeated valid times from distinct forecast cycles/leads; preserving all forecast records.",
                    name,
                    dup_count,
                )
                return data
            logger.warning(
                "Dataset '%s' %s contains %d duplicate time values; keeping first occurrence.",
                name,
                role,
                dup_count,
            )
            keep_mask = ~time_index.duplicated(keep="first")
            return data.isel(time=keep_mask)
    return data


def pair_data(
    name: str,
    method: str,
    source_data: Union[xr.Dataset, xr.DataArray, Dict[str, Union[xr.Dataset, xr.DataArray]]],
    target_data: Union[xr.Dataset, xr.DataArray],
    kwargs: Dict[str, Any],
    mask: Optional[str] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Dynamically pair two datasets using monet.

    Parameters
    ----------
    name : str
        The identifier for this pairing task.
    method : str
        Spatial interpolation method (e.g., 'nearest', 'bilinear', 'conservative').
    source_data : xarray.Dataset or xarray.DataArray
        The source data object (typically a model).
    target_data : xarray.Dataset or xarray.DataArray
        The target data object or grid (typically observations).
    kwargs : dict
        Additional keyword arguments for the pairing function (e.g., 'interp_time', 'suffix').
    mask : str, optional
        Name of a geographic region mask to apply after pairing. When provided,
        ``monet.util.mask.query_mask`` is called to add a region label variable
        to the paired dataset. Default is None (no masking).

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The resulting paired dataset.

    Raises
    ------
    ImportError
        If required pairing backend is not installed.
    ValueError
        If the mask name is not recognized by ``monet.util.mask.get_mask``.

    Examples
    --------
    >>> paired = pair_data(
    ...     "my_pairing", "bilinear", model_ds, obs_df, {"interp_time": True}
    ... )
    >>> paired_masked = pair_data(
    ...     "my_pairing", "bilinear", model_ds, obs_df, {"interp_time": True}, mask="land"
    ... )
    """
    kwargs = dict(kwargs)
    dim = kwargs.pop("model_dim", "model")
    if isinstance(source_data, dict):
        if not source_data:
            raise ValueError("Multi-model pairing requires at least one source dataset.")
        paired = {model: pair_data(name, method, data, target_data, dict(kwargs), mask=mask) for model, data in source_data.items()}
        combined = combine_paired_data(paired, dim=dim)
        # Observation fields are shared across models. Collapse fields whose
        # values are identical for every model; model fields remain model-indexed.
        if isinstance(combined, xr.Dataset):
            combined.attrs["mdt_model_dim"] = dim
            for var in list(combined.data_vars):
                if dim not in combined[var].dims:
                    continue
                first = combined[var].isel({dim: 0}, drop=True)
                if all(combined[var].isel({dim: i}, drop=True).identical(first) for i in range(1, combined.sizes[dim])):
                    combined[var] = first
        return combined

    logger.info("Pairing data '%s' using method '%s'", name, method)

    source_data = to_xarray(source_data)
    target_data = to_xarray(target_data)

    import monet

    try:
        _ensure_esmf_manager()

        # Drop UGRID 'mesh' variable if present — it causes MergeError in xr.merge
        # when monet tries to merge paired model data with observations.
        # This is a known incompatibility between UGRID-convention datasets and xarray.merge.
        if isinstance(target_data, xr.Dataset) and "mesh" in target_data:
            target_data = target_data.drop_vars("mesh")

        source_xr = _drop_duplicate_time_entries(source_data, name, "source")
        target_data = _drop_duplicate_time_entries(target_data, name, "target")

        # Use the unified monet.pair interface
        # Preserve xarray inputs through pairing to maintain Aero Protocol laziness.
        paired_data = monet.util.combinetool.pair(source_xr, target_data, method=method, **kwargs)
        paired_data = to_xarray(paired_data)
        paired_data = _harmonize_spatial_coordinates(paired_data)

        # Apply region mask if configured
        if mask is not None:
            try:
                from monet.util.mask import query_mask

                paired_data = query_mask(paired_data, mask)
                logger.info("Applied region mask '%s' to paired data '%s'", mask, name)
            except Exception as e:
                logger.error("Failed to apply mask '%s' to '%s': %s", mask, name, e)
                raise

        # Provenance Tracking
        msg = f"Paired using method '{method}' with params {kwargs}."
        paired_data = update_history(paired_data, msg)

        # Sort by time to ensure monotonicity (important for subsequent resampling/plotting)
        if isinstance(paired_data, xr.Dataset):
            if "time" in paired_data.dims:
                paired_data = paired_data.sortby("time")
                logger.info("Sorted paired xarray dataset by 'time' dimension.")
        logger.info("Successfully paired data '%s'", name)
        return cast(Union[xr.Dataset, xr.DataArray], paired_data)

    except ImportError as e:
        logger.error("Required package for pairing not found: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to pair data '%s': %s", name, e)
        raise


def combine_paired_data(
    paired_data: Dict[str, Union[xr.Dataset, xr.DataArray]],
    dim: str = "model",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Combines multiple paired datasets into a single dataset.

    This is useful for creating a single dataset containing multiple model runs
    alongside observations, facilitating comparative analysis and plotting.

    Parameters
    ----------
    paired_data : dict
        A dictionary mapping model names/identifiers to their corresponding paired data objects.
    dim : str, optional
        The name of the dimension or column to use for the model identifier. Default is 'model'.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The combined dataset.

    Raises
    ------
    ValueError
        If no data is provided.
    TypeError
        If the data types are inconsistent or unsupported.
    """
    if not paired_data:
        raise ValueError("No paired data provided for combination.")

    paired_data = {name: to_xarray(data) for name, data in paired_data.items()}
    names = pd.Index(list(paired_data), name=dim)
    logger.info(f"Combining {len(paired_data)} paired datasets along dimension '{dim}'")

    # Xarray concatenation requires homogeneous Dataset or DataArray inputs.
    # Separate the types so both runtime validation and overload selection agree.
    datasets = [data for data in paired_data.values() if isinstance(data, xr.Dataset)]
    arrays = [data for data in paired_data.values() if isinstance(data, xr.DataArray)]
    if datasets and arrays:
        raise TypeError("All paired items must have the same xarray type (Dataset or DataArray).")
    try:
        if datasets:
            return xr.concat(datasets, dim=names)
        return xr.concat(arrays, dim=names)
    except Exception as e:
        logger.error(f"Failed to combine xarray objects: {e}")
        raise
