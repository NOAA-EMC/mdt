import logging
from typing import Any, Dict, Union, cast

import pandas as pd
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def load_data(
    name: str,
    dataset_type: str,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, pd.DataFrame]:
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
    xarray.Dataset or pandas.DataFrame
        The loaded data object.

    Notes
    -----
    Aero Protocol: Ensures that data is loaded lazily where supported.
    """
    logger.info("Loading dataset '%s' of type '%s'", name, dataset_type)

    import monetio

    kwargs = dict(kwargs)

    # Force as_xarray=True for AERONET dataset type
    if dataset_type == "aeronet":
        kwargs["as_xarray"] = True

    # Extract optional isel / sel dictionary for general dimension subsetting
    isel_dict = kwargs.pop("isel", None)
    sel_dict = kwargs.pop("sel", None)
    rename_dict = kwargs.pop("rename", None)
    scale_dict = kwargs.pop("scale", None)
    offset_dict = kwargs.pop("offset", None)

    def _apply_subsets(loaded_obj: Union[xr.Dataset, pd.DataFrame]) -> Union[xr.Dataset, pd.DataFrame]:
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
        # but the time index/coordinate is missing or scalar. We dynamically reconstruct and assign the proper
        # datetime64[ns] 'time' index based on the cycle dates and lead times.
        if "time" in loaded_obj.dims and loaded_obj.dims["time"] > 1:
            if "time" not in loaded_obj.coords or loaded_obj.coords["time"].ndim == 0:
                try:
                    dates_val = kwargs.get("dates")
                    lead_time_val = kwargs.get("lead_time", [0])

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

                    if len(valid_times) == loaded_obj.dims["time"]:
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
                loaded_obj["WSPD_10maboveground"] = np.sqrt(loaded_obj["UGRD"]**2 + loaded_obj["VGRD"]**2)
                loaded_obj["WSPD_10maboveground"].attrs["long_name"] = "Wind Speed"
                loaded_obj["WSPD_10maboveground"].attrs["units"] = "m s-1"
                logger.info("Dynamically calculated 'WSPD_10maboveground' from UGRD and VGRD for '%s'", name)
        elif "u_wind" in loaded_obj.variables and "v_wind" in loaded_obj.variables:
            if "WSPD_10maboveground" not in loaded_obj.variables:
                loaded_obj["WSPD_10maboveground"] = np.sqrt(loaded_obj["u_wind"]**2 + loaded_obj["v_wind"]**2)
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
        loaded_obj: Union[xr.Dataset, pd.DataFrame],
        load_kwargs: Dict[str, Any],
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """Prefer xarray output for point-reader pathways when possible."""
        if isinstance(loaded_obj, xr.Dataset):
            return loaded_obj

        if load_kwargs.get("as_xarray") is False:
            return loaded_obj

        retry_kwargs = dict(load_kwargs)
        retry_kwargs["as_xarray"] = True
        logger.info(
            "Dataset '%s' returned DataFrame; retrying monetio.load with as_xarray=True.",
            name,
        )
        retry_obj = monetio.load(dataset_type, **retry_kwargs)
        if isinstance(retry_obj, xr.Dataset):
            return retry_obj

        logger.warning(
            "Dataset '%s' still returned non-xarray output after as_xarray=True retry.",
            name,
        )
        return loaded_obj

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

    import os
    import numpy as np

    # Auto-detect if standard Zarr or Icechunk store is already found on disk
    is_icechunk_check = kwargs.get("use_icechunk", False)
    icechunk_url_check = kwargs.get("icechunk_url") or kwargs.get("icechunk_repo", "")
    store_path_check = kwargs.get("store_path", f"./zarr_stores/{name}/") if not is_icechunk_check else ""

    if not kwargs.get("existing_zarr", False):
        auto_use_existing = False
        if is_icechunk_check and icechunk_url_check and not icechunk_url_check.startswith("s3://"):
            if os.path.exists(icechunk_url_check) and os.path.isdir(icechunk_url_check):
                # Verify if it contains icechunk files
                if os.path.exists(os.path.join(icechunk_url_check, "repo")):
                    auto_use_existing = True
        elif not is_icechunk_check and store_path_check:
            if os.path.exists(store_path_check) and os.path.isdir(store_path_check):
                # Verify if it contains any zarr metadata files
                if os.path.exists(os.path.join(store_path_check, "zarr.json")) or os.path.exists(os.path.join(store_path_check, ".zgroup")):
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
                            except Exception:
                                pass

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
                    except Exception:
                        pass

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
                    prefix = parts[1] if len(parts) > 1 else None
                    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-east-1")
                else:
                    storage = icechunk.local_filesystem_storage(icechunk_url)

                # Build high-robustness RepositoryConfig registering S3 virtual containers.
                # Mapping them to the proper AWS region and anonymous access prevents handshake timeouts
                # and connection drops. We also set AWS_MAX_ATTEMPTS for object_store retries,
                # and apply a strict max_concurrent_requests limit on RepositoryConfig.
                import os
                max_attempts = kwargs.get("max_scan_attempts", 5)
                os.environ["AWS_MAX_ATTEMPTS"] = str(max_attempts)

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
                return cast(Union[xr.Dataset, pd.DataFrame], _apply_subsets(ds))
            except Exception as e:
                logger.error("Failed to load existing Icechunk store '%s': %s", name, e)
                raise
        else:
            # Standard Zarr
            try:
                ds = xr.open_zarr(store_path, **zarr_kwargs)
                msg = f"Loaded existing Zarr store '{name}' from {store_path}."
                ds = update_history(ds, msg)
                return cast(Union[xr.Dataset, pd.DataFrame], _apply_subsets(ds))
            except Exception as e:
                logger.error("Failed to load existing Zarr store '%s': %s", name, e)
                raise

    # 2. Attempt VirtualiZarr Loading if enabled
    use_virtualizarr = kwargs.get("use_virtualizarr", False) or kwargs.get("use_icechunk", False)
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
            ds = _prefer_xarray_result(cast(Union[xr.Dataset, pd.DataFrame], ds), kwargs)
            logger.info("Successfully loaded '%s' via VirtualiZarr pathway.", name)

            # Add MDT provenance tracking
            msg = f"Loaded dataset '{name}' via VirtualiZarr (backend={backend})."
            ds = update_history(ds, msg)
            return cast(Union[xr.Dataset, pd.DataFrame], _apply_subsets(ds))

        except TypeError as e:
            # Handle cases where reader doesn't support VirtualiZarr kwargs yet
            logger.warning(
                "VirtualiZarr load failed for '%s' due to unexpected arguments: %s. Retrying with standard monetio.load.",
                name,
                e,
            )
            fallback_kwargs = _strip_virtualizarr_kwargs(kwargs)
            logger.info("Retrying standard monetio.load with kwargs: %s", fallback_kwargs)
            ds = monetio.load(dataset_type, **fallback_kwargs)
            ds = _prefer_xarray_result(cast(Union[xr.Dataset, pd.DataFrame], ds), fallback_kwargs)
            msg = f"Loaded dataset '{name}' (VirtualiZarr fallback used)."
            ds = update_history(ds, msg)
            return cast(Union[xr.Dataset, pd.DataFrame], _apply_subsets(ds))

    # 3. Standard Loading Pathway
    try:
        ds = monetio.load(dataset_type, **kwargs)
        ds = _prefer_xarray_result(cast(Union[xr.Dataset, pd.DataFrame], ds), kwargs)

        # Provenance Tracking
        msg = f"Loaded dataset '{name}' using type '{dataset_type}' with params {kwargs}."
        ds = update_history(ds, msg)

        logger.info("Successfully loaded dataset '%s'", name)
        return cast(Union[xr.Dataset, pd.DataFrame], _apply_subsets(ds))

    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", name, e)
        raise


def save_data(
    name: str,
    data: xr.Dataset,
    backend: str,
    url: str,
    kwargs: Dict[str, Any] = {},
) -> None:
    """
    Saves an xarray Dataset to either an Icechunk repository or a local Zarr store.

    Parameters
    ----------
    name : str
        The task name/identifier.
    data : xr.Dataset
        The dataset to save.
    backend : str
        The storage backend to use: 'icechunk' or 'zarr'.
    url : str
        The target store URL or local path.
    kwargs : dict
        Additional arguments passed to to_zarr().
    """
    logger.info("Saving dataset '%s' to %s store at '%s'", name, backend, url)

    if not isinstance(data, xr.Dataset):
        raise TypeError(f"Only xarray.Dataset objects can be saved, got {type(data).__name__}")

    # Clear variable and coordinate encodings to prevent write failures
    # from decode-only backend codecs (such as grib2io/Grib2SerializerCodec) when writing to Zarr.
    data = data.copy(deep=False)
    for var in list(data.variables):
        data[var].encoding = {}
    data.encoding = {}

    import pathlib
    import shutil

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
