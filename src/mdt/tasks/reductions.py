import logging
from typing import Any, List, Optional, Union, cast

import monet_stats
import xarray as xr

from mdt.utils import discover_spatial_dims, update_history

logger = logging.getLogger(__name__)
_REDUCTION_METHODS = {"mean", "max", "min", "std", "sum"}


def spatial_mean(
    obj: Union[xr.DataArray, xr.Dataset],
    lon_dim: str = "lon",
    lat_dim: str = "lat",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the area-weighted spatial mean using monet-stats.

    This function delegates to `monet_stats.weighted_spatial_mean`, which
    implements the Aero Protocol (supporting both NumPy and Dask backends).

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The input data to average spatially.
    lon_dim : str, optional
        The name of the longitude dimension. Default is 'lon'.
    lat_dim : str, optional
        The name of the latitude dimension. Default is 'lat'.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The area-weighted spatial mean of the input object.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([[1, 2], [3, 4]], coords={"lat": [0, 1], "lon": [0, 1]}, dims=("lat", "lon"))
    >>> sm = spatial_mean(da)
    """
    return calculate_reduction(obj, method="mean", dim=[lat_dim, lon_dim], force_weighted=True)


def calculate_reduction(
    obj: Union[xr.DataArray, xr.Dataset],
    method: str = "mean",
    dim: Optional[Union[str, List[str]]] = None,
    force_weighted: bool = False,
    **kwargs: Any,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Generalized reduction task adhering to the Aero Protocol.

    Supports multiple reduction methods and automatically handles area-weighted
    spatial averaging when appropriate.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The input data object.
    method : str, optional
        The reduction method: 'mean', 'max', 'min', 'std', 'sum'. Default is 'mean'.
    dim : str or list, optional
        The dimension(s) to reduce over. If None, reduces over all dimensions.
    force_weighted : bool, optional
        If True, forces area-weighted spatial mean when method='mean'.
    **kwargs : Any
        Additional arguments passed to the xarray reduction method or
        `monet_stats.weighted_spatial_mean`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The reduced data object.

    Raises
    ------
    ValueError
        If an unsupported reduction method is provided.
    """
    logger.info("Calculating reduction: %s over dim: %s", method, dim)
    if not isinstance(method, str) or method not in _REDUCTION_METHODS:
        raise ValueError(f"Unsupported reduction method '{method}'. Supported methods: {sorted(_REDUCTION_METHODS)}")
    if dim is not None:
        requested_dims = [dim] if isinstance(dim, str) else dim
        if not isinstance(requested_dims, list) or not requested_dims or not all(isinstance(item, str) for item in requested_dims):
            raise ValueError("Reduction dim must be a dimension name or a non-empty list of dimension names.")
        missing_dims = [item for item in requested_dims if item not in obj.dims]
        if missing_dims:
            raise ValueError(f"Unknown reduction dimension(s): {', '.join(missing_dims)}. Available: {list(obj.dims)}")
    method_str = method

    # 1. Specialized Case: Area-Weighted Spatial Mean
    is_spatial = False
    dims: List[str] = []
    if dim is not None:
        dims = [dim] if isinstance(dim, str) else dim
        if force_weighted:
            is_spatial = True
        elif any("lat" in d.lower() for d in dims) and any("lon" in d.lower() for d in dims):
            is_spatial = True

    if method == "mean" and is_spatial:
        logger.info("Detected spatial mean; delegating to monet_stats for area-weighting.")
        # Find spatial dimensions within the reduction set
        lat_dim, lon_dim = discover_spatial_dims(obj, dims=dims)

        # Fallback if discovery fails but reduction is forced
        if lat_dim is None:
            lat_dim = dims[0] if dims else "lat"
        if lon_dim is None:
            lon_dim = dims[1] if len(dims) > 1 else (dims[0] if dims else "lon")

        # Step A: Perform the weighted spatial mean
        # Filter kwargs to avoid duplicate arguments
        spatial_kwargs = {k: v for k, v in kwargs.items() if k not in ["lat_dim", "lon_dim"]}
        result = monet_stats.weighted_spatial_mean(obj, lat_dim=lat_dim, lon_dim=lon_dim, **spatial_kwargs)

        # Step B: Dimension Drop Fix - Reduce remaining dimensions if necessary
        remaining_dims = [d for d in dims if d not in [lat_dim, lon_dim]]
        if remaining_dims:
            logger.info("Reducing remaining non-spatial dimensions: %s", remaining_dims)
            reduction_func = getattr(result, method)
            # Filter kwargs for standard xarray reduction
            reduction_kwargs = {k: v for k, v in kwargs.items() if k not in ["lat_dim", "lon_dim"]}
            result = reduction_func(dim=remaining_dims, **reduction_kwargs)

        method_str = f"{method} (area-weighted via monet-stats)"
    else:
        # 2. General Case: Standard Xarray Reductions
        # These are backend-agnostic (work for NumPy and Dask)
        if not hasattr(obj, method):
            raise ValueError(f"Unsupported reduction method: {method}")

        reduction_func = getattr(obj, method)
        result = reduction_func(dim=dim, **kwargs)

    # 3. Provenance Tracking (Aero Protocol Rule 2.3)
    msg = f"Reduced data using method='{method_str}' over dim={dim}."
    result = update_history(result, msg)

    logger.info("Successfully completed reduction.")
    return cast(Union[xr.DataArray, xr.Dataset], result)
