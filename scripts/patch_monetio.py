"""Patch monetio readers to fix S3 + grib2io compatibility and missing file handling.

Fixes:
1. XarrayDriver: When engine='grib2io' and file is on S3, download to local temp file
   first since grib2io requires a local file path (can't handle S3File objects).
2. PandasDriver: Skip files that don't exist on S3 instead of failing the entire load.

Run this script once to apply patches to the installed monetio package.
"""

import os
import shutil

DRIVERS_PATH = "/opt/homebrew/Caskroom/miniforge/base/envs/mdt/lib/python3.14/site-packages/monetio/readers/drivers.py"


def patch_xarray_driver() -> bool:
    """Patch XarrayDriver.open to download S3 files locally when using grib2io engine."""
    with open(DRIVERS_PATH) as f:
        content = f.read()

    # The old code that opens S3 files as file objects:
    old_code = """\
                    # Logic for standard engine/remote access
                    if filename.startswith("s3://") or filename.startswith("http"):
                        fs = FileUtility.get_fs(filename)
                        file_obj = fs.open(filename)
                    else:
                        file_obj = filename

                    if "engine" not in xr_kwargs:
                        try:
                            ds = xr.open_dataset(file_obj, engine="h5netcdf", **xr_kwargs)
                        except Exception:
                            ds = xr.open_dataset(file_obj, **xr_kwargs)
                    else:
                        ds = xr.open_dataset(file_obj, **xr_kwargs)"""

    # The new code that downloads S3 files locally when grib2io is the engine:
    new_code = """\
                    # Logic for standard engine/remote access
                    if filename.startswith("s3://") or filename.startswith("http"):
                        engine = xr_kwargs.get("engine", None)
                        if engine == "grib2io":
                            # grib2io requires a local file path — download from S3 first
                            import tempfile
                            import hashlib
                            _cache_dir = os.path.join(tempfile.gettempdir(), "monetio_grib_cache")
                            os.makedirs(_cache_dir, exist_ok=True)
                            _hash = hashlib.md5(filename.encode()).hexdigest()
                            _ext = os.path.splitext(filename)[-1] or ".grib2"
                            _local_path = os.path.join(_cache_dir, f"{_hash}{_ext}")
                            if not os.path.exists(_local_path):
                                fs = FileUtility.get_fs(filename)
                                fs.get(filename, _local_path)
                            file_obj = _local_path
                        else:
                            fs = FileUtility.get_fs(filename)
                            file_obj = fs.open(filename)
                    else:
                        file_obj = filename

                    if "engine" not in xr_kwargs:
                        try:
                            ds = xr.open_dataset(file_obj, engine="h5netcdf", **xr_kwargs)
                        except Exception:
                            ds = xr.open_dataset(file_obj, **xr_kwargs)
                    else:
                        ds = xr.open_dataset(file_obj, **xr_kwargs)"""

    if old_code not in content:
        print("WARNING: XarrayDriver patch target not found — may already be patched")
        return False

    content = content.replace(old_code, new_code)

    # Also need to add 'import os' if not already present at the top
    if "import os" not in content:
        content = "import os\n" + content

    with open(DRIVERS_PATH, "w") as f:
        f.write(content)

    print("✓ Patched XarrayDriver: S3 files downloaded locally for grib2io engine")
    return True


def patch_pandas_driver() -> bool:
    """Patch PandasDriver.open to skip missing S3 files instead of failing."""
    with open(DRIVERS_PATH) as f:
        content = f.read()

    # The old code that fails on any exception:
    old_code = """\
        data_frames = []
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
            raise OSError(f"PandasDriver failed to open files. Error: {e}") from e"""

    # The new code that skips missing files gracefully:
    new_code = """\
        data_frames = []
        # Reuse our filesystem logic
        try:
            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)
            _skipped = 0

            for f in file_list:
                try:
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
                except FileNotFoundError:
                    # Skip missing files (common with ISH station data on S3)
                    _skipped += 1
                    continue

            if _skipped > 0:
                import warnings
                warnings.warn(
                    f"PandasDriver: Skipped {_skipped} missing files out of {len(file_list)} total.",
                    stacklevel=2,
                )

            if not data_frames:
                return pd.DataFrame()

            return pd.concat(data_frames, ignore_index=True)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise OSError(f"PandasDriver failed to open files. Error: {e}") from e"""

    if old_code not in content:
        print("WARNING: PandasDriver patch target not found — may already be patched")
        return False

    content = content.replace(old_code, new_code)

    with open(DRIVERS_PATH, "w") as f:
        f.write(content)

    print("✓ Patched PandasDriver: Missing S3 files are now skipped gracefully")
    return True


def add_os_import() -> None:
    """Ensure 'import os' is at the top of the file."""
    with open(DRIVERS_PATH) as f:
        content = f.read()

    if "import os\n" not in content:
        # Add after the existing imports
        content = content.replace(
            "from collections.abc import Callable",
            "import os\nfrom collections.abc import Callable",
        )
        with open(DRIVERS_PATH, "w") as f:
            f.write(content)
        print("✓ Added 'import os' to drivers.py")


if __name__ == "__main__":
    # Backup the original file
    backup_path = DRIVERS_PATH + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(DRIVERS_PATH, backup_path)
        print(f"✓ Backed up original to {backup_path}")

    patch_xarray_driver()
    patch_pandas_driver()
    add_os_import()
    print("\nDone! monetio readers patched successfully.")
