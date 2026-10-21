# Testing MDT

Use the locked development environment described in the developer guide.

## Offline CI suite

```bash
uv run --active --frozen --all-extras --inexact pytest tests/ -ra
uv run --active --frozen --all-extras --inexact mypy src/mdt
uv run --active --frozen --all-extras --inexact ruff check src/ tests/
```

Ordinary pytest runs skip tests marked `network`, `ecflow`, or `hpc` before
fixtures run. The remaining tests use synthetic data, local files, and isolated
service doubles. Optional Icechunk tests report a skip if its bindings are absent.

## Live integration suites

Opt into each resource explicitly. Selecting a marker alone does not enable it.
Tests carrying several resource markers require all corresponding switches.

```bash
# Requires working NOAA data access and the reader/regridding dependencies
uv run --active --frozen --all-extras --inexact pytest tests/ -m network --run-network -ra

# Requires ecFlow Python bindings and ecflow_server on PATH; starts a local server
uv run --active --frozen --all-extras --inexact pytest tests/ -m ecflow --run-ecflow -ra

# Enable all external integration categories in a suitably configured environment
uv run --active --frozen --all-extras --inexact pytest tests/ --run-network --run-ecflow --run-hpc -ra
```

Install ecFlow and ESMF using Conda Forge. Network data tests may also require
reader-specific packages such as `s3fs` and GRIB backends. HPC tests require the
site's scheduler, queues, and worker network configuration. The `hpc` marker and
switch reserve an explicit category for such tests; currently no test submits
real HPC jobs. These commands do not claim the remote integrations have passed.

CI runs the offline suite on Python 3.11 and 3.12. Run live integrations separately
in an environment with the required services; they are not part of ordinary PR
validation. Existing expected-failure annotations on live Prefect/Dask tests
remain visible when those tests are enabled.

## Legacy patch tests

Tests of `scripts/patch_radar.py` execute its generated source in an isolated
namespace. They do not assume the installed `monet-plots` package has been patched
and never overwrite site-packages. Current installed radar output is separately
checked against its Dataset and normalized metric interface.
