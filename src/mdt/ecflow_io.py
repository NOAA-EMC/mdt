"""Durable intermediate artifacts shared between ecFlow task processes."""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import xarray as xr


def _write_value(value: Any, directory: Path) -> dict[str, Any]:
    """Write one supported MDT task result and return its manifest entry."""
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(value, xr.Dataset):
        value.to_zarr(directory / "data.zarr", mode="w")
        return {"kind": "xarray_dataset"}
    if isinstance(value, xr.DataArray):
        variable = value.name or "_mdt_dataarray"
        value.to_dataset(name=variable).to_zarr(directory / "data.zarr", mode="w")
        return {"kind": "xarray_dataarray", "variable": variable}
    if isinstance(value, dict):
        entries = []
        for index, (key, item) in enumerate(value.items()):
            child_dir = directory / f"item-{index}"
            entries.append({"key": key, "manifest": _write_value(item, child_dir)})
        return {"kind": "mapping", "items": entries}
    if value is None:
        return {"kind": "none"}
    raise TypeError(f"Unsupported ecFlow task result type: {type(value).__name__}")


def _read_value(directory: Path, manifest: dict[str, Any]) -> Any:
    """Load a task result from its manifest entry."""
    kind = manifest["kind"]
    if kind == "xarray_dataset":
        return xr.open_zarr(directory / "data.zarr")
    if kind == "xarray_dataarray":
        return xr.open_zarr(directory / "data.zarr")[manifest["variable"]]
    if kind == "mapping":
        return {item["key"]: _read_value(directory / f"item-{index}", item["manifest"]) for index, item in enumerate(manifest["items"])}
    if kind == "none":
        return None
    raise ValueError(f"Unknown ecFlow artifact type: {kind!r}")


def _artifact_directory(data_dir: str | Path, node_id: str) -> Path:
    """Return a safe artifact directory name for an arbitrary DAG node ID."""
    import hashlib

    digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
    return Path(data_dir) / digest


def save_artifact(data_dir: str | Path, node_id: str, value: Any) -> None:
    """Atomically persist an output for downstream ecFlow tasks."""
    destination = _artifact_directory(data_dir, node_id)
    temporary = destination.with_name(f"{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        manifest = _write_value(value, temporary)
        (temporary / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_artifact(data_dir: str | Path, node_id: str) -> Any:
    """Load an upstream task result, failing clearly if it was not produced."""
    directory = _artifact_directory(data_dir, node_id)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No ecFlow artifact found for upstream task {node_id!r} at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _read_value(directory, manifest)
