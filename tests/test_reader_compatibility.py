"""Reader discovery works with both supported MonetIO registry interfaces."""

import sys
import types
from types import SimpleNamespace

import pytest

from mdt.tasks.data import _expand_local_grib_files, _reader_accepts_as_xarray, _reader_supports_virtualizarr, _resolve_reader_class


@pytest.mark.parametrize(("name", "as_xarray", "virtualizarr"), [("cmaq", False, True), ("aeronet", True, False)])
def test_installed_reader_capabilities(name, as_xarray, virtualizarr):
    """Pinned MonetIO reader classes expose the correct point/gridded contract."""
    assert isinstance(_resolve_reader_class(name), type)
    assert _reader_accepts_as_xarray(name) is as_xarray
    assert _reader_supports_virtualizarr(name) is virtualizarr


def test_modern_reader_alias_resolution(monkeypatch, mocker):
    """Modern lazy imports resolve aliases through the manifest's canonical name."""
    from monetio.readers.base import READER_REGISTRY, GriddedReader

    class Reader(GriddedReader):
        pass

    lazy = types.ModuleType("monetio._lazy")
    lazy.import_reader = mocker.Mock()
    manifest = types.ModuleType("monetio._manifest")
    manifest.resolve = lambda name: SimpleNamespace(name="canonical_test")
    monkeypatch.setitem(sys.modules, "monetio._lazy", lazy)
    monkeypatch.setitem(sys.modules, "monetio._manifest", manifest)
    monkeypatch.setitem(READER_REGISTRY, "canonical_test", Reader)
    assert _resolve_reader_class("alias_test") is Reader
    lazy.import_reader.assert_called_once_with("alias_test")


def test_local_grib_expansion_is_sorted_and_deduplicated(tmp_path):
    """Existing files reach mocked or real readers in deterministic order."""
    paths = [tmp_path / "b.grib2", tmp_path / "a.grib2"]
    for path in paths:
        path.touch()
    result = _expand_local_grib_files({"files": [str(tmp_path / "*.grib2"), str(paths[0])]}, "gefs", "model")
    assert result["files"] == sorted(str(path) for path in paths)


def test_missing_grib_input_fails_before_reader(tmp_path):
    """Unit reader doubles must not hide a missing local input path."""
    with pytest.raises(FileNotFoundError, match="No local GRIB2 files"):
        _expand_local_grib_files({"files": str(tmp_path / "*.grib2")}, "gefs", "model")
