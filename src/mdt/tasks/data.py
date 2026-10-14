import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)
_ICECHUNK_ENV_LOCK = threading.RLock()


@contextmanager
def _icechunk_retry_environment(max_attempts: int) -> Iterator[None]:
    """Set Icechunk's AWS retry setting temporarily and restore the process environment."""
    with _ICECHUNK_ENV_LOCK:
        old_value = os.environ.get("AWS_MAX_ATTEMPTS")
        os.environ["AWS_MAX_ATTEMPTS"] = str(max_attempts)
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop("AWS_MAX_ATTEMPTS", None)
            else:
                os.environ["AWS_MAX_ATTEMPTS"] = old_value


def _is_unsupported_keyword_type_error(error: TypeError) -> bool:
    message = str(error).lower()
    return "unexpected keyword argument" in message or "takes no keyword arguments" in message


def _is_as_xarray_rejection(error: BaseException) -> bool:
    """True when a reader refused the ``as_xarray`` option specifically."""
    if getattr(error, "option", None) == "as_xarray":
        return True
    return isinstance(error, TypeError) and "as_xarray" in str(error)


def _reader_accepts_as_xarray(dataset_type: str) -> bool:
    """Return True when the monetio reader for ``dataset_type`` declares ``as_xarray``.

    ``as_xarray`` is only meaningful for tabular (point) readers backed by
    ``PandasDriver``, where it selects an xarray return instead of a DataFrame.
    Gridded readers use ``XarrayDriver``, which has no such option: forwarding it
    reaches ``xr.open_dataset`` and raises ``OptionUnsupportedError``. Instead of
    assuming by name, inspect the resolved reader's ``open_dataset`` signature and
    only inject the option when the reader actually accepts it. When the reader
    cannot be resolved, preserve the previous behavior (assume it is accepted).
    """
    import inspect

    try:
        from monetio._lazy import import_reader
        from monetio._manifest import resolve
        from monetio.readers.base import READER_REGISTRY

        spec = resolve(dataset_type)
        import_reader(dataset_type)
        reader_cls = READER_REGISTRY.get(spec.name) or READER_REGISTRY.get(dataset_type)
    except Exception:
        logger.debug("Could not resolve reader for '%s'; assuming as_xarray is accepted.", dataset_type)
        return True

    if reader_cls is None:
        return True

    try:
        parameters = inspect.signature(reader_cls.open_dataset).parameters
    except (TypeError, ValueError):
        return True

    return "as_xarray" in parameters


# Options that only gridded readers (XarrayDriver) understand; tabular readers
# reject every one of them (see monetio PointReader._reject_virtualizarr).
_VIRTUALIZARR_OPTIONS = frozenset(
    {
        "use_virtualizarr",
        "virtualizarr_file",
        "virtualizarr_parser",
        "virtualizarr_backend",
        "icechunk_repo",
        "use_icechunk",
        "icechunk_url",
    }
)


def _is_virtualizarr_rejection(error: BaseException) -> bool:
    """True when a reader refused a VirtualiZarr/Icechunk ingestion option."""
    if getattr(error, "option", None) in _VIRTUALIZARR_OPTIONS:
        return True
    message = str(error).lower()
    return "virtualizarr" in message or "icechunk" in message


def _reader_supports_virtualizarr(dataset_type: str) -> bool:
    """Return True when the monetio reader for ``dataset_type`` supports VirtualiZarr ingestion.

    VirtualiZarr/Icechunk ingestion writes virtual references for chunked gridded
    formats (NetCDF/HDF5/GRIB). There is nothing to virtualize in a flat tabular
    file, so point (tabular) readers backed by ``PandasDriver`` reject every
    virtualization option. Only gridded readers (``GriddedReader``/``XarrayDriver``)
    accept them. Inspect the resolved reader's base class rather than assuming by
    name. When the reader cannot be resolved, preserve the previous behavior
    (assume ingestion is supported).
    """
    try:
        from monetio._lazy import import_reader
        from monetio._manifest import resolve
        from monetio.readers.base import READER_REGISTRY, GriddedReader

        spec = resolve(dataset_type)
        import_reader(dataset_type)
        reader_cls = READER_REGISTRY.get(spec.name) or READER_REGISTRY.get(dataset_type)
    except Exception:
        logger.debug("Could not resolve reader for '%s'; assuming VirtualiZarr is supported.", dataset_type)
        return True

    # Only a real reader class reveals the family. Anything else (None, or a
    # MagicMock from a test that stubs the monetio module) leaves the previous
    # behavior intact by assuming ingestion is supported.
    if not isinstance(reader_cls, type):
        return True

    return issubclass(reader_cls, GriddedReader)


def load_data(
    name: str,
    dataset_type: str,
    kwargs: Dict[str, Any],
) -> xr.Dataset | xr.DataArray:
    """
    Unified entry point for loading data via monetio.load.

    Supports standard loading and VirtualiZarr-based ingestion (kerchunk/icechunk).
    If VirtualiZarr parameters cause a TypeError (e.g., old reader version),
    it automatically falls back to standard loading.

    Parameters
    ----------
    name : str
        A user-defined name for this dataset.
    dataset_type : str
        The dataset type/reader name recognized by monetio (e.g., 'cmaq', 'aeronet').
    kwargs : dict
        Keyword arguments passed to monetio.load.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The loaded data object.

    Notes
    -----
    Aero Protocol: Ensures that data is loaded lazily where supported.
    """
    logger.info("Loading dataset '%s' of type '%s'", name, dataset_type)

    import monetio

    kwargs = dict(kwargs)
    # ``as_xarray`` is a tabular/point-reader option (PandasDriver returns an
    # xarray object instead of a DataFrame). Gridded readers use XarrayDriver,
    # which never accepts it and would forward it into xr.open_dataset. Only
    # request it for readers whose open_dataset actually declares the parameter,
    # so the rest of the workflow still never has to branch on a pandas view.
    if _reader_accepts_as_xarray(dataset_type):
        kwargs["as_xarray"] = True

    # Extract optional isel / sel dictionary for general dimension subsetting
    isel_dict = kwargs.pop("isel", None)
    sel_dict = kwargs.pop("sel", None)
    rename_dict = kwargs.pop("rename", None)
    scale_dict = kwargs.pop("scale", None)
    offset_dict = kwargs.pop("offset", None)

    def _apply_subsets(loaded_obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """Apply optional index-based (isel) or coordinate-based (sel) dimension slicing and generic post-processing."""
        if dataset_type == "aeronet" and isinstance(loaded_obj, pd.DataFrame):
            logger.info("Converting AERONET DataFrame '%s' to xarray Dataset.", name)
            if loaded_obj.index.has_duplicates:
                loaded_obj = loaded_obj.loc[~loaded_obj.index.duplicated(keep="first")]
            loaded_obj = loaded_obj.to_xarray()

        if not isinstance(loaded_obj, xr.Dataset):
            return loaded_obj

        # Standardize time coordinate name: map "valid_time" to "time" if "time" is not present
        if "valid_time" in loaded_obj.coords or "valid_time" in loaded_obj.dims:
            if "time" not in loaded_obj.coords and "time" not in loaded_obj.dims:
                loaded_obj = loaded_obj.rename({"valid_time": "time"})
                logger.info("Automatically mapped coordinate/dimension 'valid_time' to 'time' for '%s'.", name)
        # Apply generic variable/coordinate renaming
        if rename_dict is not None and isinstance(rename_dict, dict):
            # Support both list and dataset variables/coordinates
            valid_rename = {k: v for k, v in rename_dict.items() if k in loaded_obj.variables or k in loaded_obj.coords}
            if valid_rename:
                loaded_obj = loaded_obj.rename(valid_rename)
                logger.info("Applied generic renaming on '%s': %s", name, valid_rename)

        # Apply generic variable scaling (multiplication)
        if scale_dict is not None and isinstance(scale_dict, dict):
            for var, factor in scale_dict.items():
                if var in loaded_obj.variables:
                    loaded_obj[var] = loaded_obj[var] * factor
                    logger.info("Applied generic scaling to variable '%s' of '%s' by factor %s", var, name, factor)

        # Apply generic variable offsets (addition/subtraction)
        if offset_dict is not None and isinstance(offset_dict, dict):
            for var, shift in offset_dict.items():
                if var in loaded_obj.variables:
                    loaded_obj[var] = loaded_obj[var] + shift
                    logger.info("Applied generic offset to variable '%s' of '%s' by shift %s", var, name, shift)

        # For GFS and GEFS datasets, GRIB2 loading of multiple lead times creates a 'time' dimension of size > 1,
        # but the time index/coordinate is missing or scalar. Reconstruct it before coordinate-based sel.
        if "time" in loaded_obj.sizes and loaded_obj.sizes["time"] > 1:
            if "time" not in loaded_obj.coords or loaded_obj.coords["time"].ndim == 0:
                try:
                    dates_val = kwargs.get("dates")
                    lead_time_val = kwargs.get("lead_time", [0])

                    if dates_val is None:
                        raise ValueError("'dates' is required to reconstruct a missing time coordinate")

                    # Normalize dates to a list of strings/timestamps
                    if isinstance(dates_val, str):
                        dates_list = [dates_val]
                    elif isinstance(dates_val, (list, tuple)):
                        dates_list = list(dates_val)
                    else:
                        dates_list = [pd.to_datetime(dates_val).strftime("%Y-%m-%d %H:%M:%S")]

                    # Normalize lead_time to a list of integers
                    if isinstance(lead_time_val, (int, float)):
                        lead_times = [int(lead_time_val)]
                    else:
                        lead_times = [int(lt) for lt in lead_time_val]

                    valid_times = []
                    for d in dates_list:
                        base_dt = pd.to_datetime(d)
                        for lt in lead_times:
                            valid_times.append(base_dt + pd.to_timedelta(lt, unit="h"))

                    if len(valid_times) == loaded_obj.sizes["time"]:
                        loaded_obj = loaded_obj.assign_coords({"time": ("time", valid_times)})
                        logger.info("Dynamically assigned matching 'time' coordinate index for '%s' (%d times).", name, len(valid_times))
                except Exception as e:
                    logger.warning("Failed to dynamically reconstruct time index for '%s': %s", name, e)

        if isel_dict is not None:
            # Safely filter dictionary keys to those actually present in the dataset dimensions
            valid_isel = {k: v for k, v in isel_dict.items() if k in loaded_obj.dims}
            if valid_isel:
                loaded_obj = loaded_obj.isel(valid_isel)
                logger.info("Applied isel subsetting on '%s': %s", name, valid_isel)

        if sel_dict is not None:
            # Safely filter dictionary keys to those actually present in the dataset coordinates/dimensions
            valid_sel = {k: v for k, v in sel_dict.items() if k in loaded_obj.coords or k in loaded_obj.dims}
            if valid_sel:
                loaded_obj = loaded_obj.sel(valid_sel)
                logger.info("Applied sel subsetting on '%s': %s", name, valid_sel)

        # Dynamically calculate wind speed if U and V wind components are present
        import numpy as np

        if "UGRD" in loaded_obj.variables and "VGRD" in loaded_obj.variables:
            if "WSPD_10maboveground" not in loaded_obj.variables:
                loaded_obj["WSPD_10maboveground"] = np.sqrt(loaded_obj["UGRD"] ** 2 + loaded_obj["VGRD"] ** 2)
                loaded_obj["WSPD_10maboveground"].attrs["long_name"] = "Wind Speed"
                loaded_obj["WSPD_10maboveground"].attrs["units"] = "m s-1"
                logger.info("Dynamically calculated 'WSPD_10maboveground' from UGRD and VGRD for '%s'", name)
        elif "u_wind" in loaded_obj.variables and "v_wind" in loaded_obj.variables:
            if "WSPD_10maboveground" not in loaded_obj.variables:
                loaded_obj["WSPD_10maboveground"] = np.sqrt(loaded_obj["u_wind"] ** 2 + loaded_obj["v_wind"] ** 2)
                loaded_obj["WSPD_10maboveground"].attrs["long_name"] = "Wind Speed"
                loaded_obj["WSPD_10maboveground"].attrs["units"] = "m s-1"
                logger.info("Dynamically calculated 'WSPD_10maboveground' from u_wind and v_wind for '%s'", name)

        if isinstance(loaded_obj, xr.Dataset):
            if len(loaded_obj.variables) == 0:
                raise ValueError(
                    f"Loaded dataset '{name}' is empty (contains 0 variables/dimensions). "
                    f"This can happen if the requested date range has no matching files on the server, "
                    f"or if filters excluded all variables."
                )
        elif isinstance(loaded_obj, pd.DataFrame):
            if loaded_obj.empty:
                raise ValueError(
                    f"Loaded dataset '{name}' is empty (DataFrame contains 0 rows). "
                    f"This can happen if the requested date range has no matching files on the server, "
                    f"or if filters excluded all data."
                )

        return loaded_obj

    def _prefer_xarray_result(
        loaded_obj: Any,
    ) -> xr.Dataset | xr.DataArray:
        """Enforce the xarray reader contract at the MonetIO boundary."""
        if isinstance(loaded_obj, (xr.Dataset, xr.DataArray)):
            return loaded_obj
        raise TypeError(
            f"Dataset '{name}' reader '{dataset_type}' returned {type(loaded_obj).__name__} "
            "with as_xarray=True; MDT requires MonetIO to provide xarray objects."
        )

    # Compatibility shim: ICAP config often uses singular `date`, while
    # monetio reader interface expects `dates`.
    if dataset_type == "icap_mme" and "dates" not in kwargs and "date" in kwargs:
        kwargs["dates"] = kwargs.pop("date")

    # Support date-range syntax in YAML without requiring users to enumerate every day.
    # If both start_date and end_date are provided, expand to a daily inclusive list.
    if "start_date" in kwargs and "end_date" in kwargs and "dates" not in kwargs:
        date_freq = kwargs.pop("date_freq", "D")
        start_date = kwargs.pop("start_date")
        end_date = kwargs.pop("end_date")
        date_index = pd.date_range(start=start_date, end=end_date, freq=date_freq)
        if date_index.empty:
            raise ValueError(f"Invalid date range for dataset '{name}': {start_date} -> {end_date}")
        kwargs["dates"] = [d.strftime("%Y-%m-%d") for d in date_index]
        logger.info(
            "Expanded date range for '%s' from %s to %s (freq=%s, n=%d).",
            name,
            start_date,
            end_date,
            date_freq,
            len(kwargs["dates"]),
        )

    def _strip_virtualizarr_kwargs(input_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        vz_keys = [
            "use_virtualizarr",
            "virtualizarr_backend",
            "store_path",
            "icechunk_url",
            "icechunk_repo",
            "use_icechunk",
            "max_scan_attempts",
            "network_timeout",
            "max_concurrent_requests",
            "existing_zarr",
        ]
        return {k: v for k, v in input_kwargs.items() if k not in vz_keys}

    def _normalize_virtualizarr_kwargs(input_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate MDT kwargs to MonetIO's preferred VirtualiZarr interface."""
        load_kwargs = dict(input_kwargs)
        backend = load_kwargs.get("virtualizarr_backend")

        # MonetIO deprecates explicit backend arg for virtual paths.
        load_kwargs.pop("virtualizarr_backend", None)
        if backend == "icechunk" or load_kwargs.get("use_icechunk", False):
            load_kwargs["use_icechunk"] = True
            load_kwargs["use_virtualizarr"] = True
        elif backend == "kerchunk_json" and "store_path" in load_kwargs:
            load_kwargs["virtualizarr_file"] = load_kwargs["store_path"]

        # MonetIO prefers icechunk_url; keep legacy key support in config.
        if "icechunk_url" not in load_kwargs and "icechunk_repo" in load_kwargs:
            load_kwargs["icechunk_url"] = load_kwargs["icechunk_repo"]
        load_kwargs.pop("icechunk_repo", None)
        return load_kwargs

    # Auto-detect if standard Zarr or Icechunk store is already found on disk
    is_icechunk_check = kwargs.get("use_icechunk", False)
    icechunk_url_check = kwargs.get("icechunk_url") or kwargs.get("icechunk_repo", "")
    store_path_check = kwargs.get("store_path", f"./zarr_stores/{name}/") if not is_icechunk_check else ""

    if not kwargs.get("existing_zarr", False):
        auto_use_existing = False
        if is_icechunk_check and icechunk_url_check and not icechunk_url_check.startswith("s3://"):
            icechunk_path = Path(icechunk_url_check)
            if icechunk_path.is_dir():
                # Verify if it contains icechunk files
                if (icechunk_path / "repo").exists():
                    auto_use_existing = True
        elif not is_icechunk_check and store_path_check:
            store_path = Path(store_path_check)
            if store_path.is_dir():
                # Verify if it contains any zarr metadata files
                if (store_path / "zarr.json").exists() or (store_path / ".zgroup").exists():
                    auto_use_existing = True

        if auto_use_existing:
            # Gather requested dates to check them against the existing store
            requested_dates_str = []
            req_dates = kwargs.get("dates") or kwargs.get("date")
            if req_dates:
                if isinstance(req_dates, str):
                    requested_dates_str = [req_dates]
                elif isinstance(req_dates, (list, tuple)):
                    requested_dates_str = [str(d) for d in req_dates]
                elif isinstance(req_dates, pd.DatetimeIndex):
                    requested_dates_str = [d.strftime("%Y-%m-%d") for d in req_dates]

            try:
                requested_dates_set = {pd.to_datetime(d).date() for d in requested_dates_str}
            except Exception:
                requested_dates_set = set()

            ds_check = None
            if is_icechunk_check:
                try:
                    import icechunk

                    storage = icechunk.local_filesystem_storage(icechunk_url_check)
                    concurrency = kwargs.get("max_concurrent_requests", 4)
                    config = icechunk.RepositoryConfig(max_concurrent_requests=int(concurrency))
                    net_timeout = int(kwargs.get("network_timeout", 60))
                    s3_conf = icechunk.s3_store(
                        anonymous=True,
                        region="us-east-1",
                        network_stream_timeout_seconds=net_timeout,
                    )
                    prefixes = [
                        "s3://noaa-gefs-pds/",
                        "s3://noaa-gfs-bdp-pds/",
                        "s3://noaa-ufs-pds/",
                    ]
                    for prefix in prefixes:
                        container = icechunk.VirtualChunkContainer(
                            url_prefix=prefix,
                            store=s3_conf,
                        )
                        config.set_virtual_chunk_container(container)
                    authorize = {prefix: None for prefix in prefixes}
                    repo = icechunk.Repository.open(
                        storage,
                        config=config,
                        authorize_virtual_chunk_access=authorize,
                    )
                    session = repo.readonly_session(branch="main")
                    ds_check = xr.open_zarr(session.store, consolidated=False)
                except Exception as e:
                    logger.info("Failed to open existing local icechunk store during pre-check: %s. Forcing retrieval.", e)
                    auto_use_existing = False
            else:
                try:
                    ds_check = xr.open_zarr(store_path_check)
                except Exception as e:
                    logger.info("Failed to open existing standard Zarr store during pre-check: %s. Forcing retrieval.", e)
                    auto_use_existing = False

            if auto_use_existing and ds_check is not None:
                try:
                    time_coord = None
                    # First try exact matches in order of preference
                    for candidate in ["time", "valid_time", "dates", "date", "valid_date", "reference_time"]:
                        if candidate in ds_check.coords or candidate in ds_check.dims:
                            time_coord = candidate
                            break

                    # If no exact match, look for any coordinate containing "time" or "date"
                    if time_coord is None:
                        for coord in ds_check.coords:
                            if "time" in str(coord).lower() or "date" in str(coord).lower():
                                time_coord = coord
                                break

                    if time_coord is None:
                        logger.info("No time/date coordinate found in existing store during pre-check. Forcing retrieval.")
                        auto_use_existing = False
                    else:
                        store_times = ds_check[time_coord].values
                        if not isinstance(store_times, (list, tuple, np.ndarray, pd.DatetimeIndex)):
                            store_times = [store_times]

                        store_dates_set = set()
                        for t in store_times:
                            try:
                                store_dates_set.add(pd.to_datetime(t).date())
                            except Exception as e:
                                logger.debug("Skipping unparseable time coordinate value %r in existing store: %s", t, e)

                        if requested_dates_set:
                            missing_dates = requested_dates_set - store_dates_set
                            if missing_dates:
                                logger.info(
                                    "Existing store is missing requested dates: %s. Forcing retrieval.",
                                    sorted(list(missing_dates)),
                                )
                                auto_use_existing = False
                            else:
                                logger.info("All requested dates are present in the existing store.")
                        else:
                            if not store_dates_set:
                                logger.info("Existing store appears to be empty. Forcing retrieval.")
                                auto_use_existing = False
                except Exception as e:
                    logger.info("Error checking dates in existing store: %s. Forcing retrieval.", e)
                    auto_use_existing = False
                finally:
                    try:
                        ds_check.close()
                    except Exception as e:
                        logger.debug("Ignored error while closing the existing-store pre-check dataset: %s", e)

        if auto_use_existing:
            kwargs["existing_zarr"] = True
            logger.info(
                "Auto-detected existing %s store at '%s' with all requested dates. Bypassing monetio reloading.",
                "icechunk" if is_icechunk_check else "zarr",
                icechunk_url_check if is_icechunk_check else store_path_check,
            )

    # 1. Attempt Direct Zarr/Icechunk Loading if existing store is pointed to
    existing_zarr = kwargs.get("existing_zarr", False)
    if existing_zarr:
        backend = "icechunk" if kwargs.get("use_icechunk", False) else kwargs.get("virtualizarr_backend", "zarr")
        store_path = kwargs.get("store_path", f"./zarr_stores/{name}/")
        icechunk_url = kwargs.get("icechunk_url") or kwargs.get("icechunk_repo", "")
        zarr_kwargs = kwargs.get("zarr_kwargs", {})

        logger.info(
            "Loading existing %s store for '%s' from %s",
            backend,
            name,
            icechunk_url if backend == "icechunk" else store_path,
        )

        if backend == "icechunk":
            try:
                import icechunk

                if icechunk_url.startswith("s3://"):
                    parts = icechunk_url[5:].split("/", 1)
                    bucket = parts[0]
                    # Distinct name: `prefix` is already bound as str by the virtual-chunk
                    # container loops elsewhere in this function.
                    s3_prefix = parts[1] if len(parts) > 1 else None
                    storage = icechunk.s3_storage(bucket=bucket, prefix=s3_prefix, region="us-east-1")
                else:
                    storage = icechunk.local_filesystem_storage(icechunk_url)

                # Build high-robustness RepositoryConfig registering S3 virtual containers.
                # Mapping them to the proper AWS region and anonymous access prevents handshake timeouts
                # and connection drops. We also set AWS_MAX_ATTEMPTS for object_store retries,
                # and apply a strict max_concurrent_requests limit on RepositoryConfig.
                max_attempts = kwargs.get("max_scan_attempts", 5)
                with _icechunk_retry_environment(max_attempts):
                    concurrency = kwargs.get("max_concurrent_requests", 16)
                    config = icechunk.RepositoryConfig(max_concurrent_requests=int(concurrency))

                    net_timeout = int(kwargs.get("network_timeout", 60))
                    s3_conf = icechunk.s3_store(
                        anonymous=True,
                        region="us-east-1",
                        network_stream_timeout_seconds=net_timeout,
                    )

                    prefixes = [
                        "s3://noaa-gefs-pds/",
                        "s3://noaa-gfs-bdp-pds/",
                        "s3://noaa-ufs-pds/",
                    ]

                    for prefix in prefixes:
                        container = icechunk.VirtualChunkContainer(
                            url_prefix=prefix,
                            store=s3_conf,
                        )
                        config.set_virtual_chunk_container(container)

                    authorize = {prefix: None for prefix in prefixes}
                    repo = icechunk.Repository.open(
                        storage,
                        config=config,
                        authorize_virtual_chunk_access=authorize,
                    )
                    session = repo.readonly_session(branch="main")
                    ds = xr.open_zarr(session.store, consolidated=False, **zarr_kwargs)
                msg = f"Loaded existing Icechunk store '{name}' from {icechunk_url}."

                ds = update_history(ds, msg)
                return _apply_subsets(ds)
            except Exception as e:
                logger.error("Failed to load existing Icechunk store '%s': %s", name, e)
                raise
        else:
            # Standard Zarr
            try:
                ds = xr.open_zarr(store_path, **zarr_kwargs)
                msg = f"Loaded existing Zarr store '{name}' from {store_path}."
                ds = update_history(ds, msg)
                return _apply_subsets(ds)
            except Exception as e:
                logger.error("Failed to load existing Zarr store '%s': %s", name, e)
                raise

    # 2. Attempt VirtualiZarr Loading if enabled
    use_virtualizarr = kwargs.get("use_virtualizarr", False) or kwargs.get("use_icechunk", False)
    if use_virtualizarr and not _reader_supports_virtualizarr(dataset_type):
        # VirtualiZarr/Icechunk ingestion is gridded-only; a tabular reader (for
        # example AERONET) rejects the options. Drop them so the dataset loads
        # directly instead of failing the workflow. A configured save: step can
        # still persist the loaded data to Zarr/Icechunk afterward.
        logger.warning(
            "Dataset '%s' (type '%s') uses a tabular/point reader that does not support "
            "VirtualiZarr/Icechunk ingestion; ignoring the zarr_store ingestion options and loading directly.",
            name,
            dataset_type,
        )
        kwargs = _strip_virtualizarr_kwargs(kwargs)
        use_virtualizarr = False
    if use_virtualizarr:
        backend = "icechunk" if kwargs.get("use_icechunk", False) else kwargs.get("virtualizarr_backend", "kerchunk_json")
        store_path = kwargs.get("store_path", f"./zarr_stores/{name}/")
        icechunk_url = kwargs.get("icechunk_url") or kwargs.get("icechunk_repo", "")

        logger.info(
            "Attempting VirtualiZarr load for '%s' (backend=%s, store=%s, icechunk_url=%s)",
            name,
            backend,
            store_path,
            icechunk_url,
        )

        try:
            # Call monetio.load with all VirtualiZarr parameters
            load_kwargs = _normalize_virtualizarr_kwargs(kwargs)
            ds = monetio.load(dataset_type, **load_kwargs)
            ds = _prefer_xarray_result(ds)
            logger.info("Successfully loaded '%s' via VirtualiZarr pathway.", name)

            # Add MDT provenance tracking
            msg = f"Loaded dataset '{name}' via VirtualiZarr (backend={backend})."
            ds = update_history(ds, msg)
            return _apply_subsets(ds)

        except TypeError as e:
            # OptionUnsupportedError subclasses TypeError but reports "does not
            # support option 'X'", so check both the generic keyword rejection and
            # an explicit VirtualiZarr refusal before falling back.
            if not (_is_unsupported_keyword_type_error(e) or _is_virtualizarr_rejection(e)):
                raise
            # Handle cases where reader doesn't support VirtualiZarr kwargs yet
            logger.warning(
                "VirtualiZarr load failed for '%s' (reader does not support the option): %s. Retrying with standard monetio.load.",
                name,
                e,
            )
            fallback_kwargs = _strip_virtualizarr_kwargs(kwargs)
            logger.info("Retrying standard monetio.load with kwargs: %s", fallback_kwargs)
            ds = monetio.load(dataset_type, **fallback_kwargs)
            ds = _prefer_xarray_result(ds)
            msg = f"Loaded dataset '{name}' (VirtualiZarr fallback used)."
            ds = update_history(ds, msg)
            return _apply_subsets(ds)

    # 3. Standard Loading Pathway
    try:
        try:
            ds = monetio.load(dataset_type, **kwargs)
        except Exception as e:
            # Defensive: if the reader still refuses as_xarray (for example a
            # reader that forwards **kwargs we could not introspect), retry once
            # without it rather than failing the whole workflow.
            if not kwargs.get("as_xarray") or not _is_as_xarray_rejection(e):
                raise
            logger.warning(
                "Reader '%s' rejected as_xarray for '%s'; retrying without it.",
                dataset_type,
                name,
            )
            ds = monetio.load(dataset_type, **{k: v for k, v in kwargs.items() if k != "as_xarray"})
        ds = _prefer_xarray_result(ds)

        # Provenance Tracking
        msg = f"Loaded dataset '{name}' using type '{dataset_type}' with params {kwargs}."
        ds = update_history(ds, msg)

        logger.info("Successfully loaded dataset '%s'", name)
        return _apply_subsets(ds)

    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", name, e)
        raise


def save_data(
    name: str,
    data: xr.Dataset | xr.DataArray,
    backend: str,
    url: str,
    kwargs: Dict[str, Any] | None = None,
) -> None:
    """
    Saves an xarray Dataset or DataArray to Icechunk or a local Zarr store.

    Parameters
    ----------
    name : str
        The task name/identifier.
    data : xr.Dataset or xr.DataArray
        The xarray object to save.
    backend : str
        The storage backend to use: 'icechunk' or 'zarr'.
    url : str
        The target store URL or local path.
    kwargs : dict
        Additional arguments passed to to_zarr().
    """
    logger.info("Saving dataset '%s' to %s store at '%s'", name, backend, url)

    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError(f"Only xarray Dataset or DataArray objects can be saved, got {type(data).__name__}")

    # Clear variable and coordinate encodings to prevent write failures
    # from decode-only backend codecs (such as grib2io/Grib2SerializerCodec) when writing to Zarr.
    data = data.copy(deep=False)
    for var in list(data.variables):
        data[var].encoding = {}
    data.encoding = {}

    import pathlib
    import shutil

    kwargs = kwargs or {}

    if backend == "icechunk":
        import icechunk

        # Ensure parent directories exist
        pathlib.Path(url).parent.mkdir(parents=True, exist_ok=True)

        if url.startswith("s3://"):
            parts = url[5:].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else None
            storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-east-1")
        else:
            storage = icechunk.local_filesystem_storage(url)

        # Open or create repo
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session(branch="main")

        try:
            data.to_zarr(session.store, mode="w", consolidated=False, **kwargs)
            session.commit(f"MDT Saved dataset: {name}")
            logger.info("Successfully committed dataset '%s' to Icechunk at '%s'", name, url)
        except Exception as e:
            logger.error("Failed to save to Icechunk: %s", e)
            raise
    elif backend == "zarr":
        # Ensure target directory is clean (idempotency rule)
        if pathlib.Path(url).exists():
            shutil.rmtree(url)
        pathlib.Path(url).parent.mkdir(parents=True, exist_ok=True)

        try:
            data.to_zarr(url, mode="w", **kwargs)
            logger.info("Successfully saved dataset '%s' to Zarr at '%s'", name, url)
        except Exception as e:
            logger.error("Failed to save to Zarr: %s", e)
            raise
    else:
        raise ValueError(f"Unsupported save backend '{backend}'. Supported: 'icechunk', 'zarr'.")
