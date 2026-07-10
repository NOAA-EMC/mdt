import logging
import re
from typing import Any, Dict, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region filtering helpers
# ---------------------------------------------------------------------------


def _sanitize_region_name(region: str) -> str:
    """Replace characters that are not alphanumeric, hyphens, or periods with underscores."""
    return re.sub(r"[^a-zA-Z0-9\-.]", "_", region)


def _find_region_variable(data: xr.Dataset) -> str:
    """Find the region label variable in the dataset (added by query_mask)."""
    if not isinstance(data, xr.Dataset):
        raise ValueError("Region filtering requires an xarray Dataset.")
    for var in data.data_vars:
        if data[var].dtype == object or data[var].dtype.kind in ("U", "S"):
            return cast(str, var)
    raise ValueError("No region label variable found in dataset. Ensure a mask is applied during pairing.")


def _filter_by_region(data: xr.Dataset, region_var: str, region_name: str) -> xr.Dataset:
    """Filter dataset to only points matching the given region."""
    mask = data[region_var] == region_name
    return data.where(mask, drop=True)


def _is_empty(data: Union[xr.Dataset, xr.DataArray, pd.DataFrame]) -> bool:
    """Check if filtered dataset has no data points."""
    if isinstance(data, xr.Dataset):
        for var in data.data_vars:
            if data[var].size == 0:
                return True
            return False
    return len(data) == 0


def _format_time_range(data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]) -> str | None:
    """Extract and format a time range from common dataset types."""
    times = None

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        if "time" in data.coords:
            times = pd.to_datetime(data["time"].values, errors="coerce")
        elif "time" in data.dims:
            times = pd.to_datetime(data["time"].values, errors="coerce")
    elif isinstance(data, pd.DataFrame):
        for col in ("time", "valid_time", "date", "datetime"):
            if col in data.columns:
                times = pd.to_datetime(data[col], errors="coerce")
                break

    if times is None:
        return None

    times = pd.Series(times).dropna()
    if times.empty:
        return None

    t0 = times.min()
    t1 = times.max()
    if pd.isna(t0) or pd.isna(t1):
        return None

    fmt = "%Y-%m-%d"
    if t0 == t1:
        return t0.strftime(fmt)
    return f"{t0.strftime(fmt)} to {t1.strftime(fmt)}"


def _get_var_pair(plot_kwargs: Dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve observation/model variable names from common plotting kwargs."""
    col1 = plot_kwargs.get("col1") or plot_kwargs.get("var1")
    col2 = plot_kwargs.get("col2") or plot_kwargs.get("var2")
    if col1 is None and plot_kwargs.get("y") is not None:
        col1 = str(plot_kwargs.get("y"))
    return col1, col2


def _promote_constructor_kwargs(
    plot_type: str, constructor_kwargs: Dict[str, Any], plot_kwargs: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Move plot-type-specific options from plot() kwargs into constructor kwargs."""
    if plot_type.lower() == "spatial_bias_scatter":
        if "cbar_label" in plot_kwargs and "cbar_label" not in constructor_kwargs:
            constructor_kwargs["cbar_label"] = plot_kwargs.pop("cbar_label")
        elif "colorbar_label" in plot_kwargs and "cbar_label" not in constructor_kwargs:
            constructor_kwargs["cbar_label"] = plot_kwargs.pop("colorbar_label")

    if plot_type.lower() == "spatial":
        for key in ("vmin", "vmax", "cmap"):
            if key in constructor_kwargs and key not in plot_kwargs:
                plot_kwargs[key] = constructor_kwargs.pop(key)
    return constructor_kwargs, plot_kwargs


def _extract_pair_values(
    data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]], col1: str, col2: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract paired finite arrays for global metric calculation."""
    a = None
    b = None

    if isinstance(data, xr.Dataset) and col1 in data and col2 in data:
        a = np.asarray(data[col1].values).ravel()
        b = np.asarray(data[col2].values).ravel()
    elif isinstance(data, pd.DataFrame) and col1 in data.columns and col2 in data.columns:
        a = data[col1].to_numpy(dtype=float)
        b = data[col2].to_numpy(dtype=float)

    if a is None or b is None:
        return None, None

    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return None, None
    return a[mask], b[mask]


def _compute_global_stat(a: np.ndarray, b: np.ndarray, stat_name: str) -> float | None:
    """Compute a scalar global comparison metric."""
    if a.size == 0 or b.size == 0:
        return None

    s = stat_name.lower()
    diff = b - a

    if s == "rmse":
        return float(np.sqrt(np.mean(diff**2)))
    if s in ("mb", "bias", "mean_bias"):
        return float(np.mean(diff))
    if s == "mae":
        return float(np.mean(np.abs(diff)))
    if s in ("r", "corr", "correlation"):
        if a.size < 2:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return None


def _build_plot_title(
    name: str,
    plot_type: str,
    data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    plot_kwargs: Dict[str, Any],
    title_main: str | None,
    requested_global_stat: str | None,
) -> str:
    """Build a consistent title: main | time range | global stat."""
    col1, col2 = _get_var_pair(plot_kwargs)

    main = title_main
    if not main:
        if col1 and col2:
            main = f"{col2} vs {col1}"
        else:
            main = name.replace("_", " ").title()

    time_range = _format_time_range(data)

    stat_name = requested_global_stat or plot_kwargs.get("stat")
    if not stat_name and col1 and col2:
        if "rmse" in plot_type.lower():
            stat_name = "rmse"
        elif "bias" in plot_type.lower():
            stat_name = "mb"
        else:
            stat_name = "rmse"

    stat_text = None
    if stat_name and col1 and col2:
        a, b = _extract_pair_values(data, str(col1), str(col2))
        if a is not None and b is not None:
            value = _compute_global_stat(a, b, str(stat_name))
            if value is not None and np.isfinite(value):
                stat_text = f"{str(stat_name).upper()}={value:.3f}"

    pieces = [main]
    if time_range:
        pieces.append(time_range)
    if stat_text:
        pieces.append(stat_text)

    return " | ".join(pieces)


def generate_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
    track: str = "A",
) -> Any:
    """
    Generate visualizations using the Two-Track Rule.

    Track A (Publication): matplotlib + cartopy.
    Track B (Exploration): hvplot / geoviews.

    When a ``regions`` list is present in kwargs, produces one plot per region
    by filtering the input data and substituting ``{region}`` in the savename.

    Parameters
    ----------
    name : str
        The identifier for this plotting task.
    plot_type : str
        The type of plot to generate (e.g., 'spatial', 'scatter', 'timeseries').
    input_data : xarray.Dataset, xarray.DataArray, pandas.DataFrame, or dict
        The data to plot.
    kwargs : dict
        Additional keyword arguments to pass to the plotting function.
    track : str, optional
        Visualization track: 'A' for Static, 'B' for Interactive. Default is 'A'.

    Returns
    -------
    object or list
        The plot object (Matplotlib figure or HoloViews object), or a list of
        plot objects when regions are specified.

    Examples
    --------
    >>> fig = generate_plot("my_plot", "spatial", ds, {"savename": "spatial.png"}, track="A")
    """
    regions = kwargs.pop("regions", None)

    if regions:
        savename_template = kwargs.get("savename", f"{name}.png")
        if "{region}" not in savename_template:
            raise ValueError(f"Plot '{name}': savename must contain '{{region}}' placeholder when regions are specified.")
        region_var = _find_region_variable(input_data)
        results = []
        for region in regions:
            filtered = _filter_by_region(input_data, region_var, region)
            if _is_empty(filtered):
                logger.warning("Plot '%s': region '%s' has no data, skipping.", name, region)
                continue
            region_savename = savename_template.replace("{region}", _sanitize_region_name(region))
            region_kwargs = {**kwargs, "savename": region_savename}
            result = _generate_single_plot(name, plot_type, filtered, region_kwargs, track)
            results.append(result)
        return results
    else:
        return _generate_single_plot(name, plot_type, input_data, kwargs, track)


def _generate_single_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
    track: str = "A",
) -> Any:
    """Generate a single plot (dispatches to static or interactive track)."""
    logger.info("Generating plot '%s' [Track %s] of type: %s", name, track, plot_type)

    if isinstance(input_data, dict):
        typed_values = [v for v in input_data.values() if isinstance(v, (xr.Dataset, xr.DataArray, pd.DataFrame))]
        if len(typed_values) == 1:
            input_data = typed_values[0]

    if track == "A":
        return _generate_static_plot(name, plot_type, input_data, kwargs)
    elif track == "B":
        return _generate_interactive_plot(name, plot_type, input_data, kwargs)
    else:
        raise ValueError(f"Unknown track '{track}'. Use 'A' (Static) or 'B' (Interactive).")


def _find_plot_class(plot_type: str) -> Any:
    """
    Dynamically discover any plot class in monet_plots based on plot_type.

    Supports both case-insensitive, dash/underscore flexible matches, and 
    automatically walks all monet_plots.plots submodules to register new plot
    classes on-the-fly.

    Parameters
    ----------
    plot_type : str
        The name of the plot type (e.g., 'spatial', 'scatter', 'timeseries').

    Returns
    -------
    Any
        The plot class from monet_plots.

    Raises
    ------
    ValueError
        If no matching plot class is found.
    """
    import pkgutil
    import importlib
    import sys
    import monet_plots

    # 1. Dynamically import all submodules under monet_plots.plots
    try:
        import monet_plots.plots as m_plots
        for _, module_name, _ in pkgutil.iter_modules(m_plots.__path__):
            try:
                importlib.import_module(f"monet_plots.plots.{module_name}")
            except Exception:
                pass
    except Exception as e:
        logger.warning("Could not dynamically import monet_plots submodules: %s", e)

    # 2. Search for the class inside all loaded monet_plots.plots submodules
    # Standardize types and attributes to be alphanumeric-only for maximum naming flexibility
    cleaned_type = plot_type.lower().replace("plot", "").replace("_", "").replace("-", "")

    # Priority mapping for common types to exact classes
    priority_map = {
        "spatial": "SpatialImshowPlot",
        "spatial_grid": "SpatialFacetGridPlot",
        "facet_grid": "FacetGridPlot",
    }
    priority_target = priority_map.get(plot_type.lower())

    for mod_name, module in list(sys.modules.items()):
        if mod_name.startswith("monet_plots.plots."):
            for attr in dir(module):
                # Filter to only plotting classes or specific named wrappers
                if attr.endswith("Plot") or attr in ("Meteogram", "UpperAir", "VerticalSlice", "Windrose"):
                    if priority_target and attr == priority_target:
                        cls = getattr(module, attr)
                        if callable(cls):
                            logger.debug("Discovered priority plot class '%s' in module '%s' for type '%s'", attr, mod_name, plot_type)
                            return cls

                    cleaned_attr = attr.lower().replace("plot", "").replace("_", "").replace("-", "")
                    if cleaned_attr == cleaned_type or attr.lower() == plot_type.lower():
                        cls = getattr(module, attr)
                        if callable(cls):
                            logger.debug("Discovered plot class '%s' in module '%s' for type '%s'", attr, mod_name, plot_type)
                            return cls

    # Fallback to substring match across loaded modules if no exact match was found
    for mod_name, module in list(sys.modules.items()):
        if mod_name.startswith("monet_plots.plots."):
            for attr in dir(module):
                if attr.endswith("Plot") or attr in ("Meteogram", "UpperAir", "VerticalSlice", "Windrose"):
                    cleaned_attr = attr.lower().replace("plot", "").replace("_", "").replace("-", "")
                    if cleaned_type in cleaned_attr:
                        cls = getattr(module, attr)
                        if callable(cls):
                            logger.debug("Discovered plot class '%s' in module '%s' via substring match for type '%s'", attr, mod_name, plot_type)
                            return cls

    # Fallback: Search top-level monet_plots module attributes
    for attr in dir(monet_plots):
        if attr.lower() == plot_type.lower() or attr.lower() == f"{plot_type.lower()}plot":
            cls = getattr(monet_plots, attr)
            if callable(cls):
                logger.debug("Discovered fallback plot class '%s' at top-level for type '%s'", attr, plot_type)
                return cls

    raise ValueError(f"Could not find a plot class in monet_plots for type: '{plot_type}'")


def _generate_static_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> Any:
    """Track A: Static plotting with monet-plots (Matplotlib + Cartopy)."""
    import matplotlib

    matplotlib.use("agg")  # Non-interactive backend for thread safety

    savename = kwargs.pop("savename", f"{name}.png")
    title_main = kwargs.pop("title", None)
    requested_global_stat = kwargs.pop("global_stat", None)
    auto_title = kwargs.pop("auto_title", True)
    spatial_mean = kwargs.pop("spatial_mean", False) or kwargs.pop("spatial_average", False)

    try:
        plot_class = _find_plot_class(plot_type)
        if not callable(plot_class):
            raise ValueError(f"Discovered plot type '{plot_type}' is not a valid callable class.")

        data = input_data

        # Handle multi-column overlay: 'columns' plots multiple lines on one axis
        # Supports both list format: ["t2m", "TMP_2maboveground"]
        # and dict format: {col_name: {label: "...", color: "..."}, ...}
        columns_raw = kwargs.pop("columns", None)
        # Also support single 'column' for backward compatibility
        if columns_raw is None and "column" in kwargs:
            columns_raw = [kwargs.pop("column")]

        # Normalize to list of (col_name, label, color) tuples
        columns_spec = []
        if isinstance(columns_raw, dict):
            for col_name, opts in columns_raw.items():
                if isinstance(opts, dict):
                    columns_spec.append((col_name, opts.get("label", col_name), opts.get("color", None)))
                else:
                    columns_spec.append((col_name, col_name, None))
        elif isinstance(columns_raw, list):
            for col in columns_raw:
                columns_spec.append((col, col, None))

        if columns_spec and len(columns_spec) > 1:
            if plot_type == "profile":
                import xarray as xr
                var2_val = kwargs.get("var2", "press")
                first_col, first_label, first_color = columns_spec[0]
                
                plot_data = data
                # Preserve 1D vertical sounding coordinates and variables before taking temporal mean over valid_time
                preserved_vars = {}
                for c in ("press", "temp"):
                    if c in plot_data:
                        val = plot_data[c]
                        if "valid_time" in val.dims:
                            preserved_vars[c] = val.isel(valid_time=0, drop=True)
                        else:
                            preserved_vars[c] = val

                if "valid_time" in plot_data.dims:
                    plot_data = plot_data.mean(dim="valid_time", skipna=True)
                
                # Re-assign preserved variables and coordinates
                for c, val in preserved_vars.items():
                    if c in plot_data.coords:
                        plot_data = plot_data.assign_coords({c: val})
                    else:
                        plot_data[c] = val

                # 1. Find the target/observation pressure coordinate
                target_press_name = None
                for name in ("press", "pressure"):
                    if name in plot_data:
                        target_press_name = name
                        break

                # 2. Find and interpolate the GFS model vertical dimension
                if target_press_name is not None:
                    target_press = plot_data[target_press_name]
                    for col in list(plot_data.variables.keys()):
                        if col in ("temperature", "TMP_2maboveground", "TMP"):
                            # Find matching vertical dimension name
                            vertical_dim = None
                            for dim in ("isobaric_surface", "isobaric", "isobaricInhPa", "plev", "lev", "level", "pressure"):
                                if dim in plot_data[col].dims:
                                    vertical_dim = dim
                                    break
                            
                            if vertical_dim is not None:
                                try:
                                    gfs_vals = plot_data[vertical_dim].values
                                    igra_vals = target_press.values
                                    
                                    # Align Pa vs hPa unit differences
                                    aligned_target_press = target_press
                                    if gfs_vals.max() < 2000 and igra_vals.max() > 10000:
                                        aligned_target_press = aligned_target_press / 100.0
                                        logger.info("Scaled IGRA pressure levels from Pa to hPa for GFS alignment.")
                                    elif gfs_vals.max() > 10000 and igra_vals.max() < 2000:
                                        aligned_target_press = aligned_target_press * 100.0
                                        logger.info("Scaled IGRA pressure levels from hPa to Pa for GFS alignment.")

                                    # Perform the 1D interpolation
                                    plot_data[col] = plot_data[col].interp({vertical_dim: aligned_target_press})
                                    logger.info("Successfully interpolated GFS profile along '%s' to IGRA '%s' levels.", vertical_dim, target_press_name)
                                except Exception as e:
                                    logger.warning("Failed to vertically interpolate GFS profile to IGRA: %s", e)

                plot_obj = plot_class(data=plot_data, var1=plot_data[first_col], var2=plot_data[var2_val], label=first_label)
                if hasattr(plot_obj, "plot"):
                    plot_obj.plot(color=first_color if first_color else "black")
                    
                for col, label, color in columns_spec[1:]:
                    if col in plot_data:
                        # Defense-in-depth: Ensure shapes align before plotting to prevent crashes
                        if plot_data[col].shape != plot_data[var2_val].shape:
                            logger.warning(
                                "Skipping overlay plot for column '%s' because its shape %s "
                                "does not match vertical coordinate '%s' shape %s.",
                                col, plot_data[col].shape, var2_val, plot_data[var2_val].shape
                            )
                            continue
                        plot_obj.ax.plot(plot_data[col], plot_data[var2_val], label=label, color=color if color else "red")
                
                plot_obj.ax.set_ylabel(var2_val)
                
                # Meteorological standard: Plot pressure coordinate on log scale and invert Y-axis (height decreases as pressure decreases)
                if "press" in var2_val.lower() or "isobaric" in var2_val.lower():
                    try:
                        plot_obj.ax.set_yscale("log")
                        plot_obj.ax.invert_yaxis()
                        import matplotlib.ticker as ticker
                        plot_obj.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                        plot_obj.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                        logger.info("Configured logarithmic and reversed vertical pressure coordinate on profile plot Y-axis.")
                    except Exception as e:
                        logger.warning("Could not set log scale or invert y-axis: %s", e)

                plot_obj.ax.legend()
                plot_obj.fig.tight_layout()
                plot_obj.save(savename)
                logger.info("Saved Track A profile plot '%s' (%d lines) to %s", name, len(columns_spec), savename)
                
                if hasattr(plot_obj, "close"):
                    plot_obj.close()
                return plot_obj

            # Multi-line overlay: create one plot with first variable, then overlay others
            constructor_keys = {"x", "plotargs", "fillargs", "title", "ylabel", "label"}
            constructor_kwargs = {k: v for k, v in kwargs.items() if k in constructor_keys}
            plot_kwargs = {k: v for k, v in kwargs.items() if k not in constructor_keys}

            # Create the plot with the first column
            first_col, first_label, first_color = columns_spec[0]
            # Only pass plotargs to plot classes that support it (TimeSeriesPlot)
            init_kwargs = {"y": first_col, "label": first_label}
            if first_color and hasattr(plot_class, "_plot_xarray"):
                # TimeSeriesPlot uses plotargs for line styling
                init_kwargs["plotargs"] = {"color": first_color}
            plot_obj = plot_class(data, **init_kwargs, **constructor_kwargs)
            if hasattr(plot_obj, "plot"):
                plot_obj.plot(**plot_kwargs)

            # Overlay additional columns directly on the same axes
            import xarray as xr

            for col, label, color in columns_spec[1:]:
                if isinstance(data, xr.Dataset) and col in data:
                    var_data = data[col]
                    # Reduce to time dimension (mean over spatial dims)
                    x_dim = constructor_kwargs.get("x", "time")
                    dims_to_reduce = [d for d in var_data.dims if d != x_dim]
                    if dims_to_reduce:
                        mean_data = var_data.mean(dim=dims_to_reduce)
                    else:
                        mean_data = var_data

                    plot_kw = {"label": label}
                    if color:
                        plot_kw["color"] = color

                    # If variable has no time dimension (e.g., single forecast time),
                    # broadcast as a constant line across the time axis
                    if x_dim not in mean_data.dims:
                        if x_dim in data.dims:
                            const_val = float(mean_data.values)
                            plot_obj.ax.axhline(y=const_val, linestyle="--", **plot_kw)
                        else:
                            continue
                    else:
                        # Drop NaN so matplotlib connects valid points
                        mean_data = mean_data.dropna(x_dim)
                        if len(mean_data) > 0:
                            mean_data.plot(ax=plot_obj.ax, **plot_kw)

            plot_obj.ax.legend()
            plot_obj.fig.tight_layout()
            plot_obj.save(savename)
            logger.info("Saved Track A plot '%s' (%d lines) to %s", name, len(columns_spec), savename)

            if hasattr(plot_obj, "close"):
                plot_obj.close()

            return plot_obj

        # Single column (or no column specified)
        if columns_spec and len(columns_spec) == 1:
            kwargs["y"] = columns_spec[0][0]
            if columns_spec[0][1] != columns_spec[0][0]:
                kwargs["label"] = columns_spec[0][1]
        elif "y" not in kwargs:
            kwargs.pop("column", None)  # Remove if still present

        # Separate kwargs into constructor args and plot() args.
        # Include params for all supported plot classes:
        # TimeSeriesPlot: x, y, plotargs, fillargs, title, ylabel, label
        # TaylorDiagramPlot: col1, col2, label1, scale
        # ScatterPlot: x, y, c, colorbar, title
        # SpatialBiasScatterPlot: col1, col2, vmin, vmax, ncolors, fact, cmap
        constructor_keys = {
            "x",
            "y",
            "plotargs",
            "fillargs",
            "title",
            "style",
            "ylabel",
            "label",
            "col1",
            "col2",
            "label1",
            "scale",
            "c",
            "colorbar",
            "vmin",
            "vmax",
            "ncolors",
            "fact",
            "cmap",
            "obs_col",
            "mod_cols",
            "mod_col",
            "metrics",
            "metrics_data",
            "bias_col",
            "error_col",
            "label_col",
            "metric",
            "goal",
            "criteria",
            "modelvar",
            "obsvar",
            "gridobj",
            "discrete",
            "col",
            "row",
            "col_wrap",
            "size",
            "aspect",
            "var1",
            "var2",
            "time_col",
            "second_dim",
        }
        constructor_kwargs = {k: v for k, v in kwargs.items() if k in constructor_keys}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in constructor_keys}
        constructor_kwargs, plot_kwargs = _promote_constructor_kwargs(plot_type, constructor_kwargs, plot_kwargs)

        # For spatial plots, reduce time dimension to mean (one value per station)
        import xarray as xr

        if isinstance(data, xr.Dataset) and "time" in data.dims and len(data.dims) > 1 and plot_type in ("spatial_bias_scatter", "spatial"):
            data = data.mean(dim="time", skipna=True)

        # For timeseries plots of point datasets (where 'time' is the flat station-indexing dimension),
        # optionally group by 'time' and take the mean to compute the spatial average timeseries across all sites.
        if isinstance(data, xr.Dataset) and "time" in data.dims and len(data.dims) == 1 and plot_type == "timeseries" and spatial_mean:
            data = data.groupby("time").mean(skipna=True)
            logger.info("Computed spatial average timeseries across all sites for '%s'.", name)

        # SpatialImshow expects monotonic 1D map axes. Some regridded outputs
        # carry both 1D (lat/lon) and 2D (latitude/longitude) coordinates;
        # prefer the 1D axes for imshow-style rendering.
        if (
            plot_type == "spatial"
            and isinstance(data, xr.DataArray)
            and "lat" in data.coords
            and "lon" in data.coords
            and "latitude" in data.coords
            and "longitude" in data.coords
            and data.coords["latitude"].ndim == 2
            and data.coords["longitude"].ndim == 2
            and data.coords["lat"].ndim == 1
            and data.coords["lon"].ndim == 1
        ):
            data = data.drop_vars(["latitude", "longitude"])

        if plot_type == "spatial" and isinstance(data, xr.DataArray) and data.ndim > 2:
            data = data.squeeze(drop=True)

        # If modelvar is specified, subset the dataset to only that variable
        if isinstance(data, xr.Dataset) and "modelvar" in constructor_kwargs:
            modelvar = constructor_kwargs["modelvar"]
            if modelvar in data:
                data = data[[modelvar]]

        plot_obj = plot_class(data, **constructor_kwargs)

        if hasattr(plot_obj, "plot"):
            plot_obj.plot(**plot_kwargs)

        if auto_title and hasattr(plot_obj, "ax") and plot_obj.ax is not None:
            merged_plot_kwargs = {**constructor_kwargs, **plot_kwargs}
            built_title = _build_plot_title(
                name,
                plot_type,
                data,
                merged_plot_kwargs,
                title_main,
                requested_global_stat,
            )
            plot_obj.ax.set_title(built_title)

        plot_obj.save(savename)
        logger.info("Saved Track A plot '%s' to %s via monet-plots", name, savename)

        if hasattr(plot_obj, "close"):
            plot_obj.close()

        return plot_obj

    except ValueError as e:
        raise NotImplementedError(f"Static plot type '{plot_type}' not supported: {e}") from e


def _generate_interactive_plot(
    name: str,
    plot_type: str,
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> Any:
    """Track B: Interactive plotting with monet-plots (HvPlot/GeoViews)."""
    # Pop savename if it exists, though Track B usually doesn't save to file immediately
    kwargs.pop("savename", None)

    try:
        plot_class = _find_plot_class(plot_type)
        if not callable(plot_class):
            raise ValueError(f"Discovered plot type '{plot_type}' is not a valid callable class.")

        # MDT Architecture Rule: Data remains in its native format (Xarray) during orchestration.
        # Plot classes in monet-plots are responsible for any internal data conversions.
        data = input_data

        plot_obj = plot_class(data, **kwargs)

        if not hasattr(plot_obj, "hvplot"):
            raise AttributeError(f"Plot class '{plot_class.__name__}' does not support interactive hvplot.")

        logger.info("Generated Track B interactive plot '%s' via monet-plots", name)
        return plot_obj.hvplot(**kwargs)

    except (ValueError, AttributeError) as e:
        raise NotImplementedError(f"Interactive plot type '{plot_type}' not supported: {e}") from e
