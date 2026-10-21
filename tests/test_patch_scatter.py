"""Unit tests for scripts/patch_scatter.py.

Tests verify that:
- The patch correctly inserts `self.y = [y] if isinstance(y, str) else y`
  into ScatterPlot.__init__ after the super().__init__() call.
- Running the patch twice does not duplicate the assignment (idempotency).

Requirements: 6.1, 6.3
"""

import textwrap

import pytest

from scripts.patch_scatter import apply_patch, is_already_patched

# ---------------------------------------------------------------------------
# Fixtures: simulated ScatterPlot source files
# ---------------------------------------------------------------------------

UNPATCHED_SOURCE = textwrap.dedent("""\
    class ScatterPlot(BasePlot):
        def __init__(self, x, y, data=None, **kwargs):
            \"\"\"Initialize ScatterPlot.\"\"\"
            super().__init__(data=data, **kwargs)
            self.x = x
            # missing self.y assignment — this is the bug

        def plot(self):
            return self.y
""")

PATCHED_SOURCE = textwrap.dedent("""\
    class ScatterPlot(BasePlot):
        def __init__(self, x, y, data=None, **kwargs):
            \"\"\"Initialize ScatterPlot.\"\"\"
            super().__init__(data=data, **kwargs)
            self.y = [y] if isinstance(y, str) else y
            self.x = x
            # missing self.y assignment — this is the bug

        def plot(self):
            return self.y
""")


# ---------------------------------------------------------------------------
# Tests for is_already_patched
# ---------------------------------------------------------------------------


class TestIsAlreadyPatched:
    """Tests for the idempotency detection function."""

    def test_unpatched_source_returns_false(self):
        """Unpatched source (no self.y assignment) is detected as not patched."""
        assert is_already_patched(UNPATCHED_SOURCE) is False

    def test_patched_source_returns_true(self):
        """Source containing the self.y normalizing assignment is detected as patched."""
        assert is_already_patched(PATCHED_SOURCE) is True

    def test_different_self_y_assignment_not_detected(self):
        """A plain `self.y = y` (without isinstance normalization) is NOT the patch."""
        source = textwrap.dedent("""\
            class ScatterPlot(BasePlot):
                def __init__(self, x, y, data=None, **kwargs):
                    super().__init__(data=data, **kwargs)
                    self.y = y
        """)
        # The patch specifically checks for the isinstance pattern
        assert is_already_patched(source) is False

    def test_patched_with_extra_whitespace(self):
        """Detection works even with varied whitespace around the assignment."""
        source = "        self.y  =  [y]  if  isinstance(y, str)  else  y\n"
        # The regex allows flexible whitespace
        assert is_already_patched(source) is True


# ---------------------------------------------------------------------------
# Tests for apply_patch (using tmp_path to simulate source file)
# ---------------------------------------------------------------------------


class TestApplyPatch:
    """Tests for the patch application logic."""

    def test_patch_inserts_self_y_assignment(self, tmp_path, monkeypatch):
        """Patch inserts `self.y = [y] if isinstance(y, str) else y` after super().__init__()."""
        source_file = tmp_path / "scatter.py"
        source_file.write_text(UNPATCHED_SOURCE)

        # Monkeypatch get_scatter_source_path to return our temp file
        monkeypatch.setattr(
            "scripts.patch_scatter.get_scatter_source_path",
            lambda: str(source_file),
        )

        apply_patch()

        result = source_file.read_text()
        assert "self.y = [y] if isinstance(y, str) else y" in result

    def test_patch_inserts_after_super_init(self, tmp_path, monkeypatch):
        """The self.y assignment is inserted immediately after super().__init__()."""
        source_file = tmp_path / "scatter.py"
        source_file.write_text(UNPATCHED_SOURCE)

        monkeypatch.setattr(
            "scripts.patch_scatter.get_scatter_source_path",
            lambda: str(source_file),
        )

        apply_patch()

        lines = source_file.read_text().splitlines()
        # Find the super().__init__ line and verify self.y is the next line
        for i, line in enumerate(lines):
            if "super().__init__(" in line:
                assert "self.y = [y] if isinstance(y, str) else y" in lines[i + 1]
                break
        else:
            pytest.fail("super().__init__() not found in patched source")

    def test_idempotency_patch_twice_no_duplicate(self, tmp_path, monkeypatch):
        """Running the patch twice does not duplicate the self.y assignment."""
        source_file = tmp_path / "scatter.py"
        source_file.write_text(UNPATCHED_SOURCE)

        monkeypatch.setattr(
            "scripts.patch_scatter.get_scatter_source_path",
            lambda: str(source_file),
        )

        # Apply patch twice
        apply_patch()
        apply_patch()

        result = source_file.read_text()
        # Count occurrences of the assignment
        count = result.count("self.y = [y] if isinstance(y, str) else y")
        assert count == 1, f"Expected exactly 1 occurrence, found {count}"

    def test_idempotency_already_patched_source(self, tmp_path, monkeypatch):
        """Applying patch to already-patched source leaves it unchanged."""
        source_file = tmp_path / "scatter.py"
        source_file.write_text(PATCHED_SOURCE)

        monkeypatch.setattr(
            "scripts.patch_scatter.get_scatter_source_path",
            lambda: str(source_file),
        )

        apply_patch()

        result = source_file.read_text()
        assert result == PATCHED_SOURCE

    def test_patch_preserves_existing_code(self, tmp_path, monkeypatch):
        """Patch does not remove or alter existing lines in the source."""
        source_file = tmp_path / "scatter.py"
        source_file.write_text(UNPATCHED_SOURCE)

        monkeypatch.setattr(
            "scripts.patch_scatter.get_scatter_source_path",
            lambda: str(source_file),
        )

        apply_patch()

        result = source_file.read_text()
        # All original lines should still be present
        assert "self.x = x" in result
        assert "super().__init__(data=data, **kwargs)" in result
        assert "def plot(self):" in result
        assert "class ScatterPlot(BasePlot):" in result

    def test_patch_uses_correct_indentation(self, tmp_path, monkeypatch):
        """The inserted line uses the same indentation as surrounding code."""
        source_file = tmp_path / "scatter.py"
        source_file.write_text(UNPATCHED_SOURCE)

        monkeypatch.setattr(
            "scripts.patch_scatter.get_scatter_source_path",
            lambda: str(source_file),
        )

        apply_patch()

        lines = source_file.read_text().splitlines()
        for i, line in enumerate(lines):
            if "self.y = [y] if isinstance(y, str) else y" in line:
                # Check indentation matches the self.x line
                self_y_indent = len(line) - len(line.lstrip())
                # Find self.x line
                for other_line in lines:
                    if "self.x = x" in other_line:
                        self_x_indent = len(other_line) - len(other_line.lstrip())
                        assert self_y_indent == self_x_indent
                        break
                break
