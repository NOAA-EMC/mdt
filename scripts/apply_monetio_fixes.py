#!/usr/bin/env python
"""Apply monetio reader fixes to a local monetio source checkout.

Usage:
    cd /Users/barry/Documents/monetio-1
    python /Users/barry/Documents/mdt/scripts/apply_monetio_fixes.py

Fixes applied:
1. drivers.py (XarrayDriver): Download S3 files locally for grib2io engine
2. drivers.py (PandasDriver): Skip missing S3 files + parallel downloads
3. gfs.py: Default grib2io filters for 2m surface vars, TMP->TMP_2maboveground
4. ish.py: Handle single Timestamp in read_ish_history
5. ish_lite.py: Date coercion, date filtering, temp->t2m rename
"""

import os
import sys
from collections.abc import Sequence

# Determine the monetio source root
if os.path.exists("monetio/readers/drivers.py"):
    ROOT = "monetio/readers"
elif os.path.exists("readers/drivers.py"):
    ROOT = "readers"
else:
    print("ERROR: Run this script from the monetio repo root (where monetio/readers/ exists)")
    sys.exit(1)


def patch_file(relpath: str, patches: Sequence[tuple[str, str, str]]) -> None:
    """Apply string replacements to a file."""
    filepath = os.path.join(ROOT, relpath)
    with open(filepath) as f:
        content = f.read()

    for old, new, desc in patches:
        if old in content:
            content = content.replace(old, new, 1)
            print(f"  ✓ {desc}")
        else:
            print(f"  ⚠ SKIP (already applied or not found): {desc}")

    with open(filepath, "w") as f:
        f.write(content)


# =============================================================================
# 1. drivers.py — XarrayDriver S3+grib2io fix + PandasDriver missing files
# =============================================================================
print("\n=== Patching drivers.py ===")

DRIVERS_PATCHES = [
    # 1a. Add 'import os' at top if not present
    (
        "from collections.abc import Callable",
        "import os\nfrom collections.abc import Callable",
        "Added 'import os' at module level",
    ),
    # 1b. Remove local 'import os' inside open() that shadows module-level
    # (This is inside the VirtualiZarr block)
    (
        """            import os\n\n            # --- Kerchunk cache: load existing refs if available ---""",
        """            # os imported at module level\n\n            # --- Kerchunk cache: load existing refs if available ---""",
        "Removed local 'import os' that shadowed module-level import",
    ),
    # 1c. XarrayDriver: download S3 files locally for grib2io engine
    (
        """                    # Logic for standard engine/remote access
                    if filename.startswith("s3://") or filename.startswith("http"):
                        fs = FileUtility.get_fs(filename)
                        file_obj = fs.open(filename)
                    else:
                        file_obj = filename""",
        """                    # Logic for standard engine/remote access
                    if filename.startswith("s3://") or filename.startswith("http"):
                        engine = xr_kwargs.get("engine", None)
                        if engine == "grib2io":
                            # grib2io requires a local file path — download from S3
                            import tempfile
                            import hashlib
                            _cache_dir = os.path.join(tempfile.gettempdir(), "monetio_grib_cache")
                            os.makedirs(_cache_dir, exist_ok=True)
                            _hash = hashlib.md5(filename.encode()).hexdigest()
                            _ext = os.path.splitext(filename)[-1] or ".grib2"
                            _local_path = os.path.join(_cache_dir, _hash + _ext)
                            if not os.path.exists(_local_path):
                                fs = FileUtility.get_fs(filename)
                                fs.get(filename, _local_path)
                            file_obj = _local_path
                        else:
                            fs = FileUtility.get_fs(filename)
                            file_obj = fs.open(filename)
                    else:
                        file_obj = filename""",
        "XarrayDriver: download S3 files locally for grib2io engine",
    ),
    # 1d. PandasDriver: skip missing files + parallel downloads
    (
        """        data_frames = []
        # Reuse our filesystem logic
        try:
            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)

            for f in file_list:
                if f.startswith("s3://"):
                    # Pandas can read S3 URLs directly if s3fs is installed!
                    if "storage_options" not in kwargs:
                        kwargs["storage_options"] = {"anon": True}  # Default to public
                    df = reader_func(f, **kwargs)
                else:
                    df = reader_func(f, **kwargs)

                if preprocess:
                    df = preprocess(df)
                data_frames.append(df)

            if not data_frames:
                return pd.DataFrame()

            return pd.concat(data_frames, ignore_index=True)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise OSError(f"PandasDriver failed to open files. Error: {e}") from e""",
        """        data_frames = []
        # Reuse our filesystem logic
        try:
            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)
            _skipped = 0

            # Use concurrent downloads for S3 files when there are many
            if len(file_list) > 5 and file_list[0].startswith("s3://"):
                import concurrent.futures
                import warnings

                if "storage_options" not in kwargs:
                    kwargs["storage_options"] = {"anon": True}

                def _read_one(f):
                    df = reader_func(f, **kwargs)
                    if preprocess:
                        df = preprocess(df)
                    return df

                max_workers = min(32, len(file_list))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_read_one, f): f for f in file_list}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            data_frames.append(future.result())
                        except FileNotFoundError:
                            _skipped += 1
                        except Exception:
                            _skipped += 1

                if _skipped > 0:
                    warnings.warn(
                        f"PandasDriver: Skipped {_skipped} missing/failed files "
                        f"out of {len(file_list)} total.",
                        stacklevel=2,
                    )
            else:
                for f in file_list:
                    try:
                        if f.startswith("s3://"):
                            if "storage_options" not in kwargs:
                                kwargs["storage_options"] = {"anon": True}
                            df = reader_func(f, **kwargs)
                        else:
                            df = reader_func(f, **kwargs)

                        if preprocess:
                            df = preprocess(df)
                        data_frames.append(df)
                    except FileNotFoundError:
                        _skipped += 1
                        continue

                if _skipped > 0:
                    import warnings
                    warnings.warn(
                        f"PandasDriver: Skipped {_skipped} missing files "
                        f"out of {len(file_list)} total.",
                        stacklevel=2,
                    )

            if not data_frames:
                return pd.DataFrame()

            return pd.concat(data_frames, ignore_index=True)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise OSError(f"PandasDriver failed to open files. Error: {e}") from e""",
        "PandasDriver: parallel downloads + skip missing S3 files",
    ),
]

patch_file("drivers.py", DRIVERS_PATCHES)


# =============================================================================
# 2. gfs.py — Default filters + TMP rename
# =============================================================================
print("\n=== Patching gfs.py ===")

GFS_PATCHES = [
    # 2a. Add default grib2io filters
    (
        """        if "engine" not in kwargs:
            kwargs["engine"] = "grib2io"

        # grib2io engine generally requires local files or file-like objects.
        # XarrayDriver handles S3 URLs by opening them via fsspec.
        ds = super().open_dataset(files, **kwargs)""",
        """        if "engine" not in kwargs:
            kwargs["engine"] = "grib2io"

        # grib2io requires a filter for typeOfFirstFixedSurface when the GRIB
        # file contains multiple level types. Default to 2m above ground surface
        # variables (temperature, dewpoint, humidity, wind at 2m).
        if kwargs.get("engine") == "grib2io" and "filters" not in kwargs:
            kwargs["filters"] = {
                "typeOfFirstFixedSurface": 103,
                "scaledValueOfFirstFixedSurface": 2,
                "typeOfSecondFixedSurface": 255,
            }

        # grib2io engine generally requires local files or file-like objects.
        # XarrayDriver handles S3 URLs by opening them via fsspec.
        ds = super().open_dataset(files, **kwargs)""",
        "Added default grib2io filters for 2m surface variables",
    ),
    # 2b. Rename TMP to TMP_2maboveground
    (
        '"TMP": "temperature",',
        '"TMP": "TMP_2maboveground",  # Include level info for model-obs pairing',
        "Renamed TMP -> TMP_2maboveground for pairing compatibility",
    ),
]

patch_file("gfs.py", GFS_PATCHES)


# =============================================================================
# 3. ish.py — Handle single Timestamp in read_ish_history
# =============================================================================
print("\n=== Patching ish.py ===")

ISH_PATCHES = [
    (
        """        if dates is not None:
            index1 = (self.history.end >= dates.min()) & (self.history.begin <= dates.max())""",
        """        if dates is not None:
            # Ensure dates is a DatetimeIndex (not a single Timestamp)
            if not hasattr(dates, '__len__'):
                dates = pd.DatetimeIndex([dates])
            elif not isinstance(dates, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(dates)
            index1 = (self.history.end >= dates.min()) & (self.history.begin <= dates.max())""",
        "Fixed read_ish_history to handle single Timestamp dates",
    ),
]

patch_file("ish.py", ISH_PATCHES)


# =============================================================================
# 4. ish_lite.py — Date coercion, filtering, temp->t2m
# =============================================================================
print("\n=== Patching ish_lite.py ===")

ISH_LITE_PATCHES = [
    # 4a. Coerce single dates to DatetimeIndex
    (
        """        if files is None and dates is not None:
            dates = pd.to_datetime(dates)""",
        """        if files is None and dates is not None:
            dates = pd.to_datetime(dates)
            # Ensure dates is always a DatetimeIndex, not a single Timestamp
            if not hasattr(dates, '__len__'):
                dates = pd.DatetimeIndex([dates])""",
        "Fixed date coercion to always produce DatetimeIndex",
    ),
    # 4b. Fix date filtering for single-date queries
    (
        """            df = df.loc[(df.time >= dates.min()) & (df.time < dates.max())]""",
        """            # For single-date queries, include the full day (min == max otherwise)
            _end = dates.max() + pd.Timedelta(days=1) if len(dates) == 1 else dates.max()
            df = df.loc[(df.time >= dates.min()) & (df.time < _end)]""",
        "Fixed date filtering to include full day for single-date queries",
    ),
    # 4c. Rename temp -> t2m for pairing
    (
        """        df = self.harmonize(df)

        if as_xarray:""",
        """        df = self.harmonize(df)

        # Rename temperature to standard name for model-obs pairing
        if "temp" in df.columns:
            df = df.rename(columns={"temp": "t2m"})

        if as_xarray:""",
        "Added temp -> t2m rename for model-obs pairing compatibility",
    ),
]

patch_file("ish_lite.py", ISH_LITE_PATCHES)


print("\n=== Done! All patches applied. ===")
print("\nTo install the patched version:")
print("  cd /Users/barry/Documents/monetio-1")
print("  pip install -e .")
