"""Patch monet-plots ScatterPlot to fix missing self.y assignment bug.

Fixes:
  ScatterPlot.__init__ does not assign self.y from the constructor parameter,
  causing AttributeError when plot() accesses self.y. This patch inserts:
      self.y = [y] if isinstance(y, str) else y
  after the super().__init__() call in ScatterPlot.__init__.

Run this script once to apply the patch to the installed monet-plots package.
"""

import inspect
import re
import sys


def get_scatter_source_path() -> str:
    """Locate the ScatterPlot source file from the installed monet-plots package."""
    from monet_plots.plots.scatter import ScatterPlot

    return inspect.getfile(ScatterPlot)


def is_already_patched(content: str) -> bool:
    """Check if self.y assignment already exists in __init__."""
    # Look for self.y assignment that normalizes y to a list
    # Matches patterns like:
    #   self.y = [y] if isinstance(y, str) else y
    #   self.y = [y] if isinstance(y, str) else (y if y is not None else [])
    pattern = r"self\.y\s*=\s*\[y\]\s*if\s+isinstance\(y,\s*str\)"
    return bool(re.search(pattern, content))


def apply_patch() -> None:
    """Apply the self.y assignment patch to ScatterPlot.__init__."""
    source_path = get_scatter_source_path()
    print(f"ScatterPlot source: {source_path}")

    with open(source_path) as f:
        content = f.read()

    # Idempotency check
    if is_already_patched(content):
        print("Already patched — self.y assignment exists in ScatterPlot.__init__")
        return

    # Find the __init__ method and insert self.y assignment after super().__init__() call
    lines = content.splitlines(keepends=True)
    insert_idx = None

    in_init = False
    for i, line in enumerate(lines):
        # Detect start of __init__
        if "def __init__(" in line and not in_init:
            in_init = True
            continue

        if in_init:
            # Look for the super().__init__() call — insert after it
            if "super().__init__(" in line:
                insert_idx = i + 1
                break

    if insert_idx is None:
        # Fallback: insert after the def __init__ signature block
        # Find the first line of the method body (after docstring)
        in_init = False
        in_docstring = False
        for i, line in enumerate(lines):
            if "def __init__(" in line:
                in_init = True
                continue
            if in_init and not in_docstring:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    # Check if single-line docstring
                    if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        in_docstring = False
                    continue
                elif stripped and not stripped.startswith(")"):
                    # First real statement in __init__ body
                    insert_idx = i
                    break
            elif in_init and in_docstring:
                stripped = line.strip()
                if '"""' in stripped or "'''" in stripped:
                    in_docstring = False
                continue

    if insert_idx is None:
        print("ERROR: Could not find insertion point in ScatterPlot.__init__")
        sys.exit(1)

    # Determine indentation from the line at insert_idx
    ref_line = lines[insert_idx]
    indent = ref_line[: len(ref_line) - len(ref_line.lstrip())]

    # Insert the self.y assignment
    patch_line = f"{indent}self.y = [y] if isinstance(y, str) else y\n"
    lines.insert(insert_idx, patch_line)

    with open(source_path, "w") as f:
        f.writelines(lines)

    print(f"✓ Patched ScatterPlot.__init__: added self.y assignment at line {insert_idx + 1}")


if __name__ == "__main__":
    apply_patch()
