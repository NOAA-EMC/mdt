"""Patch monet-plots TimeSeriesStatsPlot to fix missing self.col1/self.col2 assignments.

Fixes:
    TimeSeriesStatsPlot.__init__ does not assign self.col1 and self.col2 from
    constructor parameters, causing AttributeError when _plot_xarray or
    _plot_dataframe accesses them.

This script:
1. Locates the installed monet_plots timeseries.py source file.
2. Checks if the patch has already been applied (idempotency).
3. Inserts `self.col1 = col1` and `self.col2 = [col2] if isinstance(col2, str) else col2`
   into TimeSeriesStatsPlot.__init__, preserving the existing super().__init__() call
   and data normalization.

Run this script once:
    python scripts/patch_timeseries_stats.py
"""

import inspect
import re
import sys


def find_source_file() -> str:
    """Locate the timeseries.py source file from the installed monet_plots package."""
    try:
        from monet_plots.plots import timeseries
    except ImportError:
        print("ERROR: monet_plots package is not installed.")
        sys.exit(1)

    source_path = inspect.getfile(timeseries)
    # Ensure we get the .py file, not .pyc
    if source_path.endswith(".pyc"):
        source_path = source_path[:-1]
    return source_path


def is_already_patched(content: str) -> bool:
    """Check if self.col1 and self.col2 assignments already exist in __init__."""
    # Look for both assignments within the TimeSeriesStatsPlot class
    in_class = False
    in_init = False
    has_col1 = False
    has_col2 = False

    for line in content.splitlines():
        if "class TimeSeriesStatsPlot" in line:
            in_class = True
            continue
        if in_class and re.match(r"^class\s", line):
            # Hit another class definition, stop
            break
        if in_class and re.match(r"\s+def __init__\(", line):
            in_init = True
            continue
        if in_init and re.match(r"\s+def \w+\(", line):
            # Hit another method, stop
            break
        if in_init:
            if "self.col1" in line and "=" in line:
                has_col1 = True
            if "self.col2" in line and "=" in line:
                has_col2 = True

    return has_col1 and has_col2


def apply_patch(content: str) -> str:
    """Insert self.col1 and self.col2 assignments into TimeSeriesStatsPlot.__init__.

    Places the assignments after `self.df = normalize_data(df)` to preserve
    the existing super().__init__() call and data normalization.
    """
    lines = content.splitlines(keepends=True)
    in_class = False
    in_init = False
    insert_idx = None

    for i, line in enumerate(lines):
        if "class TimeSeriesStatsPlot" in line:
            in_class = True
            continue
        if in_class and re.match(r"^class\s", line):
            break
        if in_class and re.match(r"\s+def __init__\(", line):
            in_init = True
            continue
        if in_init and re.match(r"\s+def \w+\(", line):
            break
        # Insert after `self.df = normalize_data(df)` line
        if in_init and "self.df = normalize_data" in line:
            insert_idx = i + 1
            # Detect indentation from the current line
            indent_match = re.match(r"(\s*)", line)
            indent = indent_match.group(1) if indent_match is not None else ""
            break

    if insert_idx is None:
        # Fallback: insert after super().__init__() call
        in_class = False
        in_init = False
        for i, line in enumerate(lines):
            if "class TimeSeriesStatsPlot" in line:
                in_class = True
                continue
            if in_class and re.match(r"^class\s", line):
                break
            if in_class and re.match(r"\s+def __init__\(", line):
                in_init = True
                continue
            if in_init and re.match(r"\s+def \w+\(", line):
                break
            if in_init and "super().__init__" in line:
                insert_idx = i + 1
                indent_match = re.match(r"(\s*)", line)
                indent = indent_match.group(1) if indent_match is not None else ""
                break

    if insert_idx is None:
        print("ERROR: Could not find insertion point in TimeSeriesStatsPlot.__init__")
        sys.exit(1)

    # Insert the two assignment lines
    patch_lines = [
        f"{indent}self.col1 = col1\n",
        f"{indent}self.col2 = [col2] if isinstance(col2, str) else col2\n",
    ]

    lines[insert_idx:insert_idx] = patch_lines
    return "".join(lines)


def main() -> None:
    source_path = find_source_file()
    print(f"Found source: {source_path}")

    with open(source_path) as f:
        content = f.read()

    if is_already_patched(content):
        print("Already patched — self.col1 and self.col2 assignments exist.")
        return

    patched_content = apply_patch(content)

    with open(source_path, "w") as f:
        f.write(patched_content)

    print("✓ Patched TimeSeriesStatsPlot.__init__: added self.col1 and self.col2 assignments")


if __name__ == "__main__":
    main()
