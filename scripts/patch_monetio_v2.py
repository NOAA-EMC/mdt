"""Patch monetio readers to fix S3 + grib2io compatibility and missing file handling.

Fixes:
1. XarrayDriver: When engine='grib2io' and file is on S3, download to local temp file
   first since grib2io requires a local file path (can't handle S3File objects).
2. PandasDriver: Skip files that don't exist on S3 instead of failing the entire load.
"""

import os
import shutil

DRIVERS_PATH = "/opt/homebrew/Caskroom/miniforge/base/envs/mdt/lib/python3.14/site-packages/monetio/readers/drivers.py"


def apply_patches() -> None:
    with open(DRIVERS_PATH) as f:
        lines = f.readlines()

    # === PATCH 1: Add 'import os' at the top ===
    if not any(line.strip() == "import os" for line in lines[:10]):
        lines.insert(0, "import os\n")
        print("✓ Added 'import os' at top of file")

    # === PATCH 2: Remove local 'import os' inside open() method ===
    # This shadows the module-level import and causes UnboundLocalError
    for i, line in enumerate(lines):
        if i > 200 and i < 300 and line.strip() == "import os":
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}pass  # import os removed (using module-level)\n"
            print(f"✓ Removed local 'import os' at line {i + 1}")
            break

    # === PATCH 3: Fix XarrayDriver S3 + grib2io handling ===
    # Find the section: "if filename.startswith("s3://") or filename.startswith("http"):"
    # followed by "fs = FileUtility.get_fs(filename)" and "file_obj = fs.open(filename)"
    # Replace with logic that downloads locally for grib2io
    for i, line in enumerate(lines):
        if "file_obj = fs.open(filename)" in line and i > 300:
            # Found the target line. Replace the block from the 'if filename.startswith' line
            # Find the start of the if block (should be 2 lines before)
            start_idx = i - 2
            if "filename.startswith" in lines[start_idx]:
                # Get the indentation
                indent = lines[start_idx][: len(lines[start_idx]) - len(lines[start_idx].lstrip())]
                inner = indent + "    "
                inner2 = inner + "    "

                new_block = [
                    f"{indent}# Logic for standard engine/remote access\n",
                    f'{indent}if filename.startswith("s3://") or filename.startswith("http"):\n',
                    f'{inner}engine = xr_kwargs.get("engine", None)\n',
                    f'{inner}if engine == "grib2io":\n',
                    f"{inner2}# grib2io requires a local file path — download from S3\n",
                    f"{inner2}import tempfile\n",
                    f"{inner2}import hashlib\n",
                    f'{inner2}_cache_dir = os.path.join(tempfile.gettempdir(), "monetio_grib_cache")\n',
                    f"{inner2}os.makedirs(_cache_dir, exist_ok=True)\n",
                    f"{inner2}_hash = hashlib.md5(filename.encode()).hexdigest()\n",
                    f'{inner2}_ext = os.path.splitext(filename)[-1] or ".grib2"\n',
                    f"{inner2}_local_path = os.path.join(_cache_dir, _hash + _ext)\n",
                    f"{inner2}if not os.path.exists(_local_path):\n",
                    f"{inner2}    fs = FileUtility.get_fs(filename)\n",
                    f"{inner2}    fs.get(filename, _local_path)\n",
                    f"{inner2}file_obj = _local_path\n",
                    f"{inner}else:\n",
                    f"{inner2}fs = FileUtility.get_fs(filename)\n",
                    f"{inner2}file_obj = fs.open(filename)\n",
                    f"{indent}else:\n",
                    f"{inner}file_obj = filename\n",
                ]

                # Replace lines[start_idx] through lines[i+2] (the else: file_obj = filename block)
                # Find the end of the else block
                end_idx = i + 1
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "file_obj = filename" in lines[j]:
                        end_idx = j + 1
                        break

                # Also need to remove the comment line before the if
                if "# Logic for standard engine" in lines[start_idx - 1]:
                    start_idx -= 1

                lines[start_idx:end_idx] = new_block
                print(f"✓ Patched XarrayDriver S3+grib2io handling (lines {start_idx + 1}-{end_idx})")
                break
    else:
        print("WARNING: Could not find XarrayDriver S3 handling code to patch")

    # === PATCH 4: Fix PandasDriver to skip missing files ===
    # Find "for f in file_list:" in PandasDriver.open (the non-lazy path)
    # and wrap the inner body in a try/except FileNotFoundError
    patched_pandas = False
    for i, line in enumerate(lines):
        if "for f in file_list:" in line and i > 400:
            # Check this is the right one (non-lazy path, after "data_frames = []")
            context = "".join(lines[max(0, i - 10) : i])
            if "data_frames" in context and "lazy" not in lines[i - 1]:
                indent = line[: len(line) - len(line.lstrip())]
                inner = indent + "    "
                inner2 = inner + "    "

                # Find the end of the for loop body (next line at same or lower indent, or data_frames.append)
                # Look for the pattern: the for loop body ends at "data_frames.append(df)"
                append_idx = None
                for j in range(i + 1, min(i + 20, len(lines))):
                    if "data_frames.append" in lines[j]:
                        append_idx = j
                        break

                if append_idx:
                    # Replace the for loop with a try/except version
                    # Get the original body lines
                    body_lines = lines[i + 1 : append_idx + 1]

                    new_for_block = [line]  # keep the "for f in file_list:" line
                    new_for_block.append(f"{inner}try:\n")
                    # Re-indent body lines by one extra level
                    for bl in body_lines:
                        if bl.strip():
                            new_for_block.append(f"    {bl}")
                        else:
                            new_for_block.append(bl)
                    new_for_block.append(f"{inner}except FileNotFoundError:\n")
                    new_for_block.append(f"{inner2}continue\n")

                    lines[i : append_idx + 1] = new_for_block
                    patched_pandas = True
                    print("✓ Patched PandasDriver to skip missing files")
                break

    if not patched_pandas:
        print("WARNING: Could not find PandasDriver for-loop to patch")

    # Write the patched file
    with open(DRIVERS_PATH, "w") as f:
        f.writelines(lines)

    print("\nDone! Patches applied.")


if __name__ == "__main__":
    # Backup
    backup_path = DRIVERS_PATH + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(DRIVERS_PATH, backup_path)
        print(f"✓ Backed up original to {backup_path}")
    else:
        # Restore from backup first to ensure clean state
        shutil.copy2(backup_path, DRIVERS_PATH)
        print("✓ Restored from backup for clean patching")

    apply_patches()
