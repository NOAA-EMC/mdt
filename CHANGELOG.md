# CHANGELOG


## v1.1.0 (2026-10-21)

### Bug Fixes

- Enhance GFS and GEFS examples with Icechunk access and improvements
  ([#37](https://github.com/NOAA-EMC/mdt/pull/37),
  [`50c8f34`](https://github.com/NOAA-EMC/mdt/commit/50c8f3421f142c412b20f25f04d8bd381093992f))

* feat: update gefs and gfs examples to use icechunk access with grib2io

This commit updates the GFS and GEFS verification examples to utilize the new Icechunk
  virtualization capability. It also enhances the core DAG builder to support Icechunk-specific
  parameters and robustness settings for cloud-native GRIB2 access.

Key changes: - Modified `src/mdt/dag.py` to automatically set `use_icechunk=True` and propagate
  robustness parameters (`max_scan_attempts`, `network_timeout`, `max_concurrent_requests`) from the
  YAML configuration to the data loading tasks. - Updated `src/mdt/tasks/data.py` to correctly
  handle the new virtualization keys during fallback scenarios. - Updated
  `docs/examples/gefs_aeronet.yaml` and `docs/examples/gfs_ish_lite.yaml` to demonstrate the new
  `icechunk` backend configuration. - Enhanced `docs/yaml_format.md` and `docs/examples.md` with
  descriptions and examples for the new Icechunk and grib2io-related settings.

Key changes: - Modified `src/mdt/dag.py` to automatically set `use_icechunk=True` and propagate
  robustness parameters (`max_scan_attempts`, `network_timeout`, `max_concurrent_requests`) from the
  YAML configuration to the data loading tasks. - Updated `src/mdt/tasks/data.py` to correctly
  handle the new virtualization keys during fallback scenarios by adding them to `vz_keys`. -
  Updated `docs/examples/gefs_aeronet.yaml` and `docs/examples/gfs_ish_lite.yaml` to demonstrate the
  new `icechunk` backend configuration. - Enhanced `docs/yaml_format.md` and `docs/examples.md` with
  descriptions and examples for the new Icechunk and grib2io-related settings. - Updated unit tests
  to verify proper handling of new Icechunk and robustness parameters.

Tests were verified using a pip-installed environment with dependencies fetched from GitHub as
  specified in `pyproject.toml`.

* feat: support icechunk-grib2io and direct loading of existing zarr/icechunk stores

This commit fulfills two major enhancements: 1. Updates GEFS and GFS examples to utilize Icechunk
  virtualization via grib2io, including support for robustness parameters (retries, timeouts). 2.
  Implements a generic mechanism to load any data source directly from an existing Zarr or Icechunk
  store (e.g., preprocessed AERONET), bypassing standard monetio readers.

Key changes: - Modified `src/mdt/config.py` to allow 'zarr' as a valid backend. - Enhanced
  `src/mdt/dag.py` to propagate 'existing_zarr', 'zarr_kwargs', and robustness settings from YAML to
  tasks. - Updated `src/mdt/tasks/data.py` with a prioritized direct-loading path for existing
  stores and improved fallback keyword stripping. - Updated documentation (`yaml_format.md`,
  `examples.md`) and example YAML files. - Expanded unit tests to cover direct loading logic and
  fixed Hypothesis test deadline issues in the development environment.

Tests were verified in a local environment with all dependencies satisfied.

* Refactor code for improved readability and maintainability

- Added missing import statements and removed unnecessary blank lines in various files. - Simplified
  conditional checks and error messages for better clarity. - Consolidated multi-line list
  comprehensions into single lines where appropriate. - Enhanced type checking and error handling in
  data processing functions. - Updated test assertions for consistency and clarity. - Improved
  region name sanitization logic and ensured proper handling of unsafe characters. - Ensured that
  all changes adhere to PEP 8 style guidelines for better code quality.

* fix: update linting rules and improve code readability in tests and engine

* feat: Add AOD statistics documentation and improve icechunk handling

- Introduced new markdown file for AOD statistics (`gefs_aeronet_stats.md`). - Updated
  `ConfigParser` to require `icechunk_url` or `icechunk_repo` for the `icechunk` backend. - Enhanced
  `DAGBuilder` to correctly handle `icechunk_url` and streamline virtualizarr parameters. - Modified
  `EcFlowEngine` to replace `icechunk_repo` with `icechunk_url` in task variables. - Refactored
  `load_data` function to prefer `icechunk_url` and improve error handling for date ranges. - Added
  functions to deduplicate time entries in datasets before pairing. - Updated plotting functions to
  handle new title formatting and ensure proper variable extraction. - Enhanced tests to cover new
  functionality and ensure backward compatibility with legacy keys.

* feat: Enhance data processing and plotting capabilities

- Added Prefect task wrappers for calculate_reduction and save_data functions in engine.py. -
  Implemented save_data function to support saving xarray Datasets to Icechunk and Zarr formats in
  data.py. - Enhanced load_data function to dynamically calculate wind speed from UGRD and VGRD or
  u_wind and v_wind components. - Improved pair_data function to sort paired datasets by time and
  preserve duplicate time entries for point observation datasets. - Updated _generate_static_plot to
  handle profile plotting with interpolation for GFS and IGRA datasets. - Introduced unit tests for
  save_data and profile plotting functionalities to ensure reliability and correctness.

* feat: Update GFS model configuration and add wind statistics and plots

* fix: update monet-stats dependency URL in pyproject.toml

* feat: ability to save observations and paired data

* Add comprehensive tests for data handling and task execution

- Introduced new tests for artifact loading and saving in `test_ecflow_io.py`, ensuring
  compatibility with xarray and handling of None values. - Updated `test_ecflow_zarr_store.py` to
  verify correct script generation for Zarr store variables. - Enhanced `test_extended_tasks.py`
  with additional checks for metric validation and plot title generation. - Modified
  `test_gefs_openaq.py` to clarify mock data handling for OpenAQ. - Refined
  `test_gfs_ish_lite_e2e.py` to ensure task counts and trigger expressions align with the configured
  DAG. - Improved `test_load_data_zarr_store.py` with new tests for data loading and processing,
  including handling of Icechunk repositories. - Added tests in `test_pairing_mask.py` for mask
  application and exception handling. - Created `test_save_dataarray.py` to validate DataArray
  persistence. - Updated `test_tasks.py` to reject invalid data types and ensure proper provenance
  tracking. - Enhanced `test_ufs_airnow.py` and `test_viirs_aod.py` with mock ESMF manager to
  isolate tests from global state. - Updated dependencies in `uv.lock` to reflect changes in the
  prefect package version.

* feat: update Python version to 3.12 and improve configuration validation messages

* fix: improve error handling and logging in load_data function

* feat: update documentation and statistics files; enhance .gitignore for new directories

* feat: enhance load_data function to support as_xarray and VirtualiZarr ingestion options for
  readers

* feat: enhance load_data to retry without rejected options and add corresponding tests

* feat: add time coordinate promotion for concatenated GRIB2 files and corresponding tests

* feat: enhance YAML configuration and data handling for local GRIB2 files; improve pairing logic
  and time coordinate normalization

* refactor: improve error handling and variable naming in configuration and data processing
  functions

* fix(ecflow): load pairing artifacts and preserve model dimensions

* fix(statistics): require explicit consent for unweighted fallback

* build: pin MONET revisions and use frozen development environments

* fix(pairing): forward masks through Prefect and ecFlow engines

* fix(prefect): resolve nested model futures before serial pairing

* fix(types): resolve strict mypy errors across workflow tasks

* fix(data): repair reader compatibility and offline regression coverage

* test: separate offline CI from opt-in live integrations

* fix(ci): update checkout step to handle pull requests from forks

---------

Co-authored-by: bbakernoaa <22104759+bbakernoaa@users.noreply.github.com>

- Release CI ([#38](https://github.com/NOAA-EMC/mdt/pull/38),
  [`b7d43c1`](https://github.com/NOAA-EMC/mdt/commit/b7d43c15745972dda0bfcb2a02aa4c895f904999))

* feat: update gefs and gfs examples to use icechunk access with grib2io

This commit updates the GFS and GEFS verification examples to utilize the new Icechunk
  virtualization capability. It also enhances the core DAG builder to support Icechunk-specific
  parameters and robustness settings for cloud-native GRIB2 access.

Key changes: - Modified `src/mdt/dag.py` to automatically set `use_icechunk=True` and propagate
  robustness parameters (`max_scan_attempts`, `network_timeout`, `max_concurrent_requests`) from the
  YAML configuration to the data loading tasks. - Updated `src/mdt/tasks/data.py` to correctly
  handle the new virtualization keys during fallback scenarios. - Updated
  `docs/examples/gefs_aeronet.yaml` and `docs/examples/gfs_ish_lite.yaml` to demonstrate the new
  `icechunk` backend configuration. - Enhanced `docs/yaml_format.md` and `docs/examples.md` with
  descriptions and examples for the new Icechunk and grib2io-related settings.

Key changes: - Modified `src/mdt/dag.py` to automatically set `use_icechunk=True` and propagate
  robustness parameters (`max_scan_attempts`, `network_timeout`, `max_concurrent_requests`) from the
  YAML configuration to the data loading tasks. - Updated `src/mdt/tasks/data.py` to correctly
  handle the new virtualization keys during fallback scenarios by adding them to `vz_keys`. -
  Updated `docs/examples/gefs_aeronet.yaml` and `docs/examples/gfs_ish_lite.yaml` to demonstrate the
  new `icechunk` backend configuration. - Enhanced `docs/yaml_format.md` and `docs/examples.md` with
  descriptions and examples for the new Icechunk and grib2io-related settings. - Updated unit tests
  to verify proper handling of new Icechunk and robustness parameters.

Tests were verified using a pip-installed environment with dependencies fetched from GitHub as
  specified in `pyproject.toml`.

* feat: support icechunk-grib2io and direct loading of existing zarr/icechunk stores

This commit fulfills two major enhancements: 1. Updates GEFS and GFS examples to utilize Icechunk
  virtualization via grib2io, including support for robustness parameters (retries, timeouts). 2.
  Implements a generic mechanism to load any data source directly from an existing Zarr or Icechunk
  store (e.g., preprocessed AERONET), bypassing standard monetio readers.

Key changes: - Modified `src/mdt/config.py` to allow 'zarr' as a valid backend. - Enhanced
  `src/mdt/dag.py` to propagate 'existing_zarr', 'zarr_kwargs', and robustness settings from YAML to
  tasks. - Updated `src/mdt/tasks/data.py` with a prioritized direct-loading path for existing
  stores and improved fallback keyword stripping. - Updated documentation (`yaml_format.md`,
  `examples.md`) and example YAML files. - Expanded unit tests to cover direct loading logic and
  fixed Hypothesis test deadline issues in the development environment.

Tests were verified in a local environment with all dependencies satisfied.

* Refactor code for improved readability and maintainability

- Added missing import statements and removed unnecessary blank lines in various files. - Simplified
  conditional checks and error messages for better clarity. - Consolidated multi-line list
  comprehensions into single lines where appropriate. - Enhanced type checking and error handling in
  data processing functions. - Updated test assertions for consistency and clarity. - Improved
  region name sanitization logic and ensured proper handling of unsafe characters. - Ensured that
  all changes adhere to PEP 8 style guidelines for better code quality.

* fix: update linting rules and improve code readability in tests and engine

* feat: Add AOD statistics documentation and improve icechunk handling

- Introduced new markdown file for AOD statistics (`gefs_aeronet_stats.md`). - Updated
  `ConfigParser` to require `icechunk_url` or `icechunk_repo` for the `icechunk` backend. - Enhanced
  `DAGBuilder` to correctly handle `icechunk_url` and streamline virtualizarr parameters. - Modified
  `EcFlowEngine` to replace `icechunk_repo` with `icechunk_url` in task variables. - Refactored
  `load_data` function to prefer `icechunk_url` and improve error handling for date ranges. - Added
  functions to deduplicate time entries in datasets before pairing. - Updated plotting functions to
  handle new title formatting and ensure proper variable extraction. - Enhanced tests to cover new
  functionality and ensure backward compatibility with legacy keys.

* feat: Enhance data processing and plotting capabilities

- Added Prefect task wrappers for calculate_reduction and save_data functions in engine.py. -
  Implemented save_data function to support saving xarray Datasets to Icechunk and Zarr formats in
  data.py. - Enhanced load_data function to dynamically calculate wind speed from UGRD and VGRD or
  u_wind and v_wind components. - Improved pair_data function to sort paired datasets by time and
  preserve duplicate time entries for point observation datasets. - Updated _generate_static_plot to
  handle profile plotting with interpolation for GFS and IGRA datasets. - Introduced unit tests for
  save_data and profile plotting functionalities to ensure reliability and correctness.

* feat: Update GFS model configuration and add wind statistics and plots

* fix: update monet-stats dependency URL in pyproject.toml

* feat: ability to save observations and paired data

* Add comprehensive tests for data handling and task execution

- Introduced new tests for artifact loading and saving in `test_ecflow_io.py`, ensuring
  compatibility with xarray and handling of None values. - Updated `test_ecflow_zarr_store.py` to
  verify correct script generation for Zarr store variables. - Enhanced `test_extended_tasks.py`
  with additional checks for metric validation and plot title generation. - Modified
  `test_gefs_openaq.py` to clarify mock data handling for OpenAQ. - Refined
  `test_gfs_ish_lite_e2e.py` to ensure task counts and trigger expressions align with the configured
  DAG. - Improved `test_load_data_zarr_store.py` with new tests for data loading and processing,
  including handling of Icechunk repositories. - Added tests in `test_pairing_mask.py` for mask
  application and exception handling. - Created `test_save_dataarray.py` to validate DataArray
  persistence. - Updated `test_tasks.py` to reject invalid data types and ensure proper provenance
  tracking. - Enhanced `test_ufs_airnow.py` and `test_viirs_aod.py` with mock ESMF manager to
  isolate tests from global state. - Updated dependencies in `uv.lock` to reflect changes in the
  prefect package version.

* feat: update Python version to 3.12 and improve configuration validation messages

* fix: improve error handling and logging in load_data function

* feat: update documentation and statistics files; enhance .gitignore for new directories

* feat: enhance load_data function to support as_xarray and VirtualiZarr ingestion options for
  readers

* feat: enhance load_data to retry without rejected options and add corresponding tests

* feat: add time coordinate promotion for concatenated GRIB2 files and corresponding tests

* feat: enhance YAML configuration and data handling for local GRIB2 files; improve pairing logic
  and time coordinate normalization

* refactor: improve error handling and variable naming in configuration and data processing
  functions

* fix(ecflow): load pairing artifacts and preserve model dimensions

* fix(statistics): require explicit consent for unweighted fallback

* build: pin MONET revisions and use frozen development environments

* fix(pairing): forward masks through Prefect and ecFlow engines

* fix(prefect): resolve nested model futures before serial pairing

* fix(types): resolve strict mypy errors across workflow tasks

* fix(data): repair reader compatibility and offline regression coverage

* test: separate offline CI from opt-in live integrations

* fix(ci): update checkout step to handle pull requests from forks

* fix(release): set changelog template directory to 'templates'

* fix(dependencies): update ruff and mypy versions in pre-commit config; improve code reliability
  fix(data): correct valid_time assignment in promote_time_coordinate function fix(statistics):
  streamline call_kwargs construction and improve logging for unweighted metrics test: add
  comprehensive tests for pairing configuration and model statistics handling

fix(tests): enhance test coverage for Prefect integration and xarray type contracts

---------

Co-authored-by: bbakernoaa <22104759+bbakernoaa@users.noreply.github.com>

- Resolve linting error in ecflow_engine.py
  ([`f018c25`](https://github.com/NOAA-EMC/mdt/commit/f018c25c8af43a0d026d1d41ec479a82ff07f84a))

- Moved `logger` definition after imports to satisfy E402.

### Chores

- Correct PSR dry-run and checkout in CI
  ([`5588fcc`](https://github.com/NOAA-EMC/mdt/commit/5588fccb2de47cd7bf0083626ba5f5ef64983b16))

- Replaced additional_options with args for --dry-run - Added github.head_ref to checkout to avoid
  Detached HEAD in dry-run

- Fix semantic release dry-run option in CI
  ([`1131e13`](https://github.com/NOAA-EMC/mdt/commit/1131e13946292906c6b01a5fcceaa40ad1eca4aa))

- Moved --dry-run from root_options to additional_options in ci.yml - Ensures correct argument
  passing to python-semantic-release action

- Harden tests, add extended e2e integration, and drop Python < 3.11
  ([`c54136e`](https://github.com/NOAA-EMC/mdt/commit/c54136efcc59c9c2d24f0f299eca89e9f673d5fa))

- Hardened statistics module with robust metric discovery and weighted fallback tests. - Added
  comprehensive end-to-end integration tests for: - GEFS-Aerosol vs OpenAQ - VIIRS AOD vs ICAP-MME -
  UFS-AQM vs AirNow - Updated `pyproject.toml` to require Python >= 3.11 and bumped Ruff target
  version. - Streamlined CI matrix to support only Python 3.11 and 3.12. - Verified all 98 tests
  pass and adhere to Aero Protocol standards.

- Harden tests, add extended e2e integration, drop Python < 3.11, and fix linting
  ([`ac61571`](https://github.com/NOAA-EMC/mdt/commit/ac61571f69e1aa61b2889fff7c4c144c25e35168))

- Hardened statistics module with robust metric discovery and weighted fallback tests. - Added
  comprehensive end-to-end integration tests for: - GEFS-Aerosol vs OpenAQ - VIIRS AOD vs ICAP-MME -
  UFS-AQM vs AirNow - Updated `pyproject.toml` to require Python >= 3.11 and bumped Ruff target
  version. - Streamlined CI matrix to support only Python 3.11 and 3.12. - Fixed existing linting
  issues in test files (missing docstrings, unused variables, formatting). - Verified all 98 tests
  pass and adhere to Aero Protocol standards.

- Harden tests, extend e2e integration, and ensure import isolation
  ([`96a9e0b`](https://github.com/NOAA-EMC/mdt/commit/96a9e0bc19827e1e987421ca2a8ae304366ea3b9))

- Moved `dask`, `distributed`, and `dask-jobqueue` imports into methods in `src/mdt/engine.py` and
  `src/mdt/hpc.py` to fix lazy-import isolation issues. - Hardened statistics module with robust
  metric discovery and weighted fallback tests. - Added comprehensive end-to-end integration tests
  for GEFS, OpenAQ, VIIRS, UFS, and AirNow. - Updated `pyproject.toml` to require Python >= 3.11 and
  bumped Ruff target version. - Streamlined CI matrix to support only Python 3.11 and 3.12. -
  Verified all 98 tests pass and adhere to Aero Protocol standards.

- Harden tests, extend e2e integration, drop Python < 3.11
  ([`5957f4c`](https://github.com/NOAA-EMC/mdt/commit/5957f4cfa01e42a593bef79f9878672b7edd302a))

- Hardened statistics module with robust metric discovery and weighted fallback tests. - Added
  comprehensive end-to-end integration tests for GEFS, OpenAQ, VIIRS, UFS, and AirNow. - Updated
  `pyproject.toml` to require Python >= 3.11 and bumped Ruff target version. - Streamlined CI matrix
  to support only Python 3.11 and 3.12. - Fixed lazy-import isolation in `src/mdt/engine.py` for
  Dask/Distributed. - Resolved numerous linting issues (missing docstrings, unused variables) in
  test suite. - Verified all 98 tests pass and adhere to Aero Protocol standards.

- Ignore D102 in tests and finalize CI hardening
  ([`771f692`](https://github.com/NOAA-EMC/mdt/commit/771f6925fd31bec0006cdfc1878d659d9dd7d6b2))

- Added D102 (missing-docstring-in-public-method) to Ruff ignores for tests/ - Verified all linting
  and type checking passes across src/ and tests/ - Finalized hardened CI/CD pipeline and semantic
  versioning setup

### Continuous Integration

- Streamline release workflow and update semantic release configuration
  ([`192e7c1`](https://github.com/NOAA-EMC/mdt/commit/192e7c1e093815446a6d2fa2617ada149a466557))

- Remove repository owner check to allow releases from forks - Replace manual Python setup and pip
  install with python-semantic-release action v9.21.1 - Consolidate release steps into dedicated
  action with inline configuration - Add conditional GitHub Release publication step based on
  release output - Update pyproject.toml semantic_release config to use new action format - Remove
  deprecated version_toml and branch settings - Add explicit main branch configuration and changelog
  template settings - Simplifies CI/CD pipeline maintenance and ensures consistent release behavior

### Features

- Extend VirtualiZarr support to Prefect and fix monetio dependency
  ([`83853a6`](https://github.com/NOAA-EMC/mdt/commit/83853a67b9e384cabb289f7a3d39d7832f89df62))

- Update `PrefectEngine` to recognize VirtualiZarr-enabled loading tasks. - Apply `virtualizarr`
  tags and custom `task_run_name` to Prefect tasks via `.with_options()`. - Enhance
  `prefect_load_data` with detailed VirtualiZarr logging (backend, store path). - Fix installation
  error by updating `monetio` dependency branch to `@develop` in `pyproject.toml`. - Add
  comprehensive mock-based unit tests in `tests/test_prefect_zarr_store.py`.

- Extend VirtualiZarr support to Prefect execution path
  ([`8850117`](https://github.com/NOAA-EMC/mdt/commit/885011792447b9226a86d33f8a95ded70b88159e))

- Update `PrefectEngine` to recognize VirtualiZarr-enabled loading tasks. - Apply `virtualizarr`
  tags and custom `task_run_name` to Prefect tasks. - Enhance `prefect_load_data` with detailed
  VirtualiZarr logging. - Add comprehensive mock-based unit tests in
  `tests/test_prefect_zarr_store.py`.

- Extend VirtualiZarr support to Prefect path for full parity with ecFlow
  ([`1d3afc8`](https://github.com/NOAA-EMC/mdt/commit/1d3afc877dd8cdda6941b2d6736d676dbe01c339))

- Updated `PrefectEngine` in `src/mdt/engine.py` to identify VirtualiZarr-enabled tasks. - Applied
  Prefect tags ('virtualizarr') and custom task run names for better observability. - Enhanced
  `prefect_load_data` with detailed logging of backend, store path, and icechunk repository. - Fixed
  a blocking installation error in `pyproject.toml` by updating the `monetio` dependency branch. -
  Resolved multiple `ruff` linting issues in the test suite. - Added a new integration test suite
  `tests/test_prefect_zarr_store.py`.

- Extend VirtualiZarr support to Prefect with robust tests
  ([`31bd992`](https://github.com/NOAA-EMC/mdt/commit/31bd99200c10efdec9e6f77b51bda5aff1f29cac))

- Update `PrefectEngine` to support VirtualiZarr tagging and custom task names. - Enhance logging
  for VirtualiZarr tasks in `prefect_load_data`. - Fix monetio dependency branch in
  `pyproject.toml`. - Add integration tests with robust mocking for `dask.distributed`. - Resolve
  linting issues in test files.

- Full VirtualiZarr support for Prefect with dependency fix
  ([`df0bd2d`](https://github.com/NOAA-EMC/mdt/commit/df0bd2d022028c5b298b028fc4876e823a55d44b))

- Update `PrefectEngine` to support VirtualiZarr tagging, custom run names, and detailed logging
  (including `icechunk_repo`) for parity with ecFlow. - Fix blocking installation error by updating
  `monetio` dependency to `@develop` in `pyproject.toml`. - Resolve `ruff` linting issues (D205,
  D400, E402) across multiple test files. - Add and verify comprehensive integration tests in
  `tests/test_prefect_zarr_store.py`.

- Harden CI and introduce semantic versioning
  ([`4a6280d`](https://github.com/NOAA-EMC/mdt/commit/4a6280de6d6dd78c4332df4377043ef3a72c208e))

- Reorganized CI into parallel jobs for linting, type-checking, and testing - Implemented strict
  MyPy type checking and expanded Ruff ruleset - Added comprehensive type hints across the entire
  `src/mdt` codebase - Automated semantic versioning with `python-semantic-release` - Integrated
  Codecov for automated test coverage reporting - Implemented `pathlib.Path` for improved
  cross-platform support - Configured MyPy module overrides to handle environment-specific
  suppression - Configured targeted Ruff ignores for the test suite (added D102) - Resolved
  environment-dependent MyPy issues in Prefect/Dask integrations - Fixed Prefect lazy-import and
  task-creation logic to satisfy integration tests - Ensured consistent and robust tool access in CI
  via `uvx` and reusable workflows

- Harden CI and introduce semantic versioning
  ([`aafb18e`](https://github.com/NOAA-EMC/mdt/commit/aafb18e7c61218093726e7d00b0454c8db0a1e12))

- Reorganized CI into parallel jobs for linting, type-checking, and testing - Implemented strict
  MyPy type checking and expanded Ruff ruleset - Added comprehensive type hints across the entire
  `src/mdt` codebase - Automated semantic versioning with `python-semantic-release` - Integrated
  Codecov for automated test coverage reporting - Implemented `pathlib.Path` for improved
  cross-platform support - Configured MyPy module overrides to handle environment-specific
  suppression - Configured targeted Ruff ignores for the test suite to support idiomatic patterns -
  Resolved environment-dependent MyPy issues in Prefect/Dask integrations - Fixed Prefect
  lazy-import and task-creation logic to satisfy integration tests - Ensured consistent and robust
  tool access in CI via `uvx` and reusable workflows

- Harden CI and introduce semantic versioning
  ([`6d5163e`](https://github.com/NOAA-EMC/mdt/commit/6d5163e8ccc2b3bab03e3de0a6702cfc3d9c6cd5))

- Reorganized CI into parallel jobs for linting, type-checking, and testing - Implemented strict
  MyPy type checking and expanded Ruff ruleset - Added comprehensive type hints across the entire
  `src/mdt` codebase - Automated semantic versioning with `python-semantic-release` - Integrated
  Codecov for automated test coverage reporting - Implemented `pathlib.Path` for improved
  cross-platform support - Configured MyPy module overrides to handle environment-specific
  suppression - Configured targeted Ruff ignores for the test suite to support idiomatic patterns -
  Ensured consistent and robust tool access in CI via `uvx` and reusable workflows

- Harden CI and introduce semantic versioning
  ([`a229203`](https://github.com/NOAA-EMC/mdt/commit/a229203eb00da220d9ca3785f77e1d77ea051976))

- Reorganized CI into parallel jobs for linting, type-checking, and testing - Implemented strict
  MyPy type checking and expanded Ruff ruleset - Added comprehensive type hints across the entire
  `src/mdt` codebase - Automated semantic versioning with `python-semantic-release` - Integrated
  Codecov for automated test coverage reporting - Implemented `pathlib.Path` for improved
  cross-platform support - Resolved complex environment-dependent MyPy issues in Prefect/Dask
  integrations - Configured targeted Ruff ignores for the test suite to support idiomatic patterns -
  Ensured consistent and robust tool access in CI via `uvx` and reusable workflows

- Harden CI and introduce semantic versioning
  ([`5c2f800`](https://github.com/NOAA-EMC/mdt/commit/5c2f8008ddd91a783d20830a65ef0ca412222f38))

- Separates CI into parallel lint, type-check, and test jobs - Implements strict MyPy type checking
  and expanded Ruff rules - Adds comprehensive type hints across the `src/mdt` codebase - Automates
  semantic versioning via python-semantic-release - Adds test coverage reporting with Codecov
  integration - Supports pre-releases on the develop branch - Implements release dry-run for pull
  requests to main - Refactors file operations to use `pathlib.Path` - Configures Ruff per-file
  ignores for the test suite - Resolves complex MyPy issues in Dask/Prefect integrations with proper
  casting and type ignores where necessary

- Harden CI and introduce semantic versioning
  ([`2010a64`](https://github.com/NOAA-EMC/mdt/commit/2010a642ceb33f690ad4f5b5b2ff5c623f4e1934))

- Separates CI into lint, type-check, and test jobs - Configures strict MyPy type checking and
  expanded Ruff rules - Adds full type hints across the `src/mdt` codebase - Automates semantic
  versioning via python-semantic-release - Adds test coverage reporting with Codecov integration -
  Supports pre-releases on the develop branch - Implements release dry-run for pull requests to main
  - Configures Ruff ignores for the test suite - Fixes several type-related bugs and improves
  cross-platform Path handling - Resolves complex MyPy issues in Dask and Prefect integrations using
  proper casting

- Harden CI and introduce semantic versioning
  ([`a5ec9ea`](https://github.com/NOAA-EMC/mdt/commit/a5ec9ea9d7474c5ef7d7f25983120d02a818dd9d))

- Separates CI into lint, type-check, and test jobs - Configures strict MyPy type checking and
  expanded Ruff rules - Adds full type hints across the `src/mdt` codebase - Automates semantic
  versioning via python-semantic-release - Adds test coverage reporting with Codecov integration -
  Supports pre-releases on the develop branch - Implements release dry-run for pull requests to main
  - Configures Ruff ignores for the test suite to allow idiomatic test code - Fixes CI workflows to
  use reusable patterns and correct job dependencies

- Harden CI and introduce semantic versioning
  ([`a6d6457`](https://github.com/NOAA-EMC/mdt/commit/a6d6457b68c9056b352af35a171430156fda0bf9))

- Separates CI into lint, type-check, and test jobs - Configures strict MyPy type checking and
  expanded Ruff rules - Adds full type hints across the `src/mdt` codebase - Automates semantic
  versioning via python-semantic-release - Adds test coverage reporting with Codecov integration -
  Supports pre-releases on the develop branch - Implements release dry-run for pull requests to main
  - Fixes CI lint job to use uvx for ruff access

- Harden CI and introduce semantic versioning
  ([`722ff77`](https://github.com/NOAA-EMC/mdt/commit/722ff7733da2fcea4cc451107f9abc38a65b8949))

- Separates CI into lint, type-check, and test jobs - Configures strict MyPy type checking and
  expanded Ruff rules - Adds full type hints across the `src/mdt` codebase - Automates semantic
  versioning via python-semantic-release - Adds test coverage reporting with Codecov integration -
  Supports pre-releases on the develop branch - Implements release dry-run for pull requests to main

- Improve user experience with enhanced CLI and validation
  ([`86148c7`](https://github.com/NOAA-EMC/mdt/commit/86148c74df0b3246ce49e96122198e5accc1732a))

- Added `validate` subcommand to check YAML configs and DAG consistency. - Added `template`
  subcommand to generate sample configurations. - Added `--version` and `--debug` flags to the main
  CLI. - Implemented stricter configuration validation in `ConfigParser`. - Improved error handling
  to suppress stack traces by default. - Updated README.md with comprehensive documentation and
  quick-start guide. - Fixed existing tests to comply with new validation rules.

- Improve UX and enhance ecFlow suite generation
  ([`9571ffa`](https://github.com/NOAA-EMC/mdt/commit/9571ffa1a9d842f4e32638566edb67e1a9746cd2))

- Added `validate` and `template` subcommands to CLI. - Implemented stricter configuration
  validation. - Improved README documentation. - Enhanced `EcFlowEngine` to save `.def` suite files
  for easier inspection. - Fixed logger import and linting in `ecflow_engine.py`. - Verified all 98
  tests pass.

- Improve UX and enhance ecFlow suite generation
  ([`bff62e4`](https://github.com/NOAA-EMC/mdt/commit/bff62e4da204a634d783d871219888044a9c7d49))

- Added `validate` and `template` subcommands to CLI. - Implemented stricter configuration
  validation. - Improved README documentation. - Enhanced `EcFlowEngine` to save `.def` suite files
  for easier inspection. - Fixed logger import in `ecflow_engine.py`. - Updated unit tests for new
  validation rules.

- Improve weighted stats fallbacks and expand plotting support
  ([`e791c6f`](https://github.com/NOAA-EMC/mdt/commit/e791c6fd2ec92f7cc5aaa4df85be2473bedbc7c1))

- Add manual weighted fallbacks for MB, BIAS, MBIAS, and MAE in statistics task - Add support for
  'timeseries' and 'taylor' plot types in plotting task - Implement xarray-to-dataframe conversion
  for specific plot types - Add explicit resource management (close()) for static plots - Add
  comprehensive tests for new statistics and plotting features

- Streamline orchestrator logic and fix linting errors
  ([`2e050d8`](https://github.com/NOAA-EMC/mdt/commit/2e050d8f59da9b1bba207f6ceb089eccb4f9cd76))

- Remove manual weighted fallback statistics in `src/mdt/tasks/statistics.py` - Implement dynamic
  plot class discovery in `src/mdt/tasks/plotting.py` - Fix linting errors (unused imports and
  sorting) across src and tests - Update tests to reflect library delegation architecture

- Streamline statistics and plotting by delegating to specialized libraries
  ([`4b26708`](https://github.com/NOAA-EMC/mdt/commit/4b267083154604cae00a90d700bf2527ece0946d))

- Remove manual weighted fallback statistics in `src/mdt/tasks/statistics.py` to rely strictly on
  `monet-stats`. - Implement dynamic plot class discovery in `src/mdt/tasks/plotting.py` to
  automatically support all `monet-plots` types. - Generalize Track A/B logic in plotting task. -
  Update test suite to reflect new delegation-first architecture.

- **zarr-store**: Add VirtualiZarr support with multiple backend options
  ([`e2ba0bf`](https://github.com/NOAA-EMC/mdt/commit/e2ba0bf2a8b52af1c0618b3ef072bc6f295fd0a2))

- Add zarr_store configuration validation in config.py with support for kerchunk_json,
  kerchunk_parquet, and icechunk backends - Implement VirtualiZarr parameter passing through
  DAGBuilder to load_data tasks - Update EcFlowEngine to inject zarr_store configuration as task
  variables for ecFlow execution - Enhance load_data function to accept and handle VirtualiZarr
  parameters with fallback to standard monetio loading - Add hypothesis to dev dependencies for
  property-based testing - Update monetio dependency to feature/virtualizarr branch for VirtualiZarr
  compatibility - Add comprehensive test coverage for zarr_store configuration across config, dag,
  ecflow, and data loading modules - Enables users to virtualize large datasets using kerchunk or
  icechunk backends for improved performance and reduced memory footprint

### Testing

- Harden statistics tests and ensure end-to-end integration
  ([`17f2e78`](https://github.com/NOAA-EMC/mdt/commit/17f2e7804322dc457be805eb1683074883473777))

- Add `tests/test_aero_weighted_fallbacks.py` to verify metric discovery and manual weighted
  fallback logic. - Update `tests/test_merra2_icap_aeronet.py` to use real `monet-stats` library
  functions instead of mocks. - Refine `tests/test_tasks.py` to include explicit checks for Dask
  laziness and provenance metadata. - Ensure all 95 tests pass and comply with Aero Protocol
  standards.

- Harden tests and add extended integration tests for GEFS, OpenAQ, VIIRS, UFS, and AirNow
  ([`d524980`](https://github.com/NOAA-EMC/mdt/commit/d524980e24bc4fbcf965f66fb0c8efd145155430))

- Add `tests/test_aero_weighted_fallbacks.py` for robust metric discovery and fallback. - Update
  `tests/test_merra2_icap_aeronet.py` with real library calls. - Add `tests/test_gefs_openaq.py` for
  GEFS-Aerosol vs OpenAQ workflow. - Add `tests/test_viirs_aod.py` for VIIRS AOD vs ICAP-MME
  workflow. - Add `tests/test_ufs_airnow.py` for UFS-AQM vs AirNow workflow. - Ensure all 98 tests
  pass and preserve Dask laziness.


## v1.0.0 (2026-04-09)

### Continuous Integration

- Migrate to Python matrix testing and semantic versioning
  ([`ce7c16c`](https://github.com/NOAA-EMC/mdt/commit/ce7c16c0bcaee2324cdd2b588e3de91b5b4a168e))

- Simplify CI workflow to test against Python 3.9, 3.11, and 3.12 matrix - Replace micromamba
  environment with standard pip installation - Remove develop branch from CI triggers, focus on main
  only - Add release workflow with python-semantic-release for automated versioning - Migrate
  version management from static pyproject.toml to dynamic __version__ in __init__.py - Configure
  semantic-release to parse conventional commits and auto-generate changelog - Streamline linting to
  check only src/ and tests/ directories - Add verbose pytest output with short traceback format for
  better test visibility

### Documentation

- Add MDT documentation using Zensical and GitHub Pages deployment
  ([`08c2b37`](https://github.com/NOAA-EMC/mdt/commit/08c2b3798eccea83ea69fff136c5f3fce709d6e9))

- Added 'docs/' directory with intro, quick start, and user guide. - Added 'docs/yaml_format.md'
  with detailed reference and multiple model comparison example. - Added 'docs/hpc_usage.md' with
  NOAA platform profiles (Hera, Jet, Orion, etc.). - Created 'zensical.toml' for documentation
  project configuration. - Added '.github/workflows/deploy-docs.yml' for automated GitHub Pages
  deployment.

- Finalize MDT documentation with Zensical and NOAA HPC guide
  ([`23044ab`](https://github.com/NOAA-EMC/mdt/commit/23044ab671e89da7ee098c309389f4e2a2e9d49c))

- Added documentation structure using Zensical (Intro, Quick Start, User Guide, YAML Format, HPC
  Usage). - Included detailed reference for YAML configuration with multiple model comparison
  example. - Added NOAA HPC platform profiles (Hera, Jet, Orion, Hercules, Gaea, Ursa, WCOSS2). -
  Removed legacy references to 'Aero Protocol' and 'Two-Track Visualization'. - Added GitHub Action
  for automated deployment to GitHub Pages.

### Features

- Add area-weighted spatial_mean task following Aero Protocol
  ([`8def554`](https://github.com/NOAA-EMC/mdt/commit/8def55493a146edbcf1e939abe79f7618ab99344))

- Implement `spatial_mean` in `src/mdt/tasks/reductions.py` using cosine of latitude weighting. -
  Ensure backend-agnostic support for NumPy and Dask via Xarray's native `.weighted()` API. - Add
  `tests/test_reductions.py` with a Double-Check test verifying identical results for Eager and Lazy
  backends. - Maintain data provenance by updating `history` attribute. - Ensure all files adhere to
  ruff formatting and linting standards.

- Add combine functionality to merge paired datasets in MDT
  ([`001b5d6`](https://github.com/NOAA-EMC/mdt/commit/001b5d6d714717d81ebd251567ec7c1b2a3102ce))

- Add pip dependency to environment configuration
  ([`93e7294`](https://github.com/NOAA-EMC/mdt/commit/93e729495c419f50284e172a139c2995f2850cf0))

- Add scipy dependency to environment configuration
  ([`ae90548`](https://github.com/NOAA-EMC/mdt/commit/ae905480f333d7a429caf14f5ebc76dbdfe9f794))

- Delegate all statistics to monet-stats with weighted support
  ([`715678f`](https://github.com/NOAA-EMC/mdt/commit/715678fbcd1a80b7df1a822361e4d96c830ad27d))

- Refactor `mdt.tasks.statistics` to delegate weighted MB, MAE, MSE, and RMSE to
  `monet_stats.weighted_spatial_mean`. - Implement `mdt.tasks.reductions.spatial_mean` as a wrapper
  around `monet-stats`. - Add `tests/test_weighted_stats.py` with Aero Protocol Double-Check
  verification. - Ensure backend-agnostic behavior and preservation of Dask laziness. - Fix
  dimension mapping for `monet_stats.weighted_spatial_mean` calls.

- Enhance pairing functionality to support model-to-model comparison with merge_target option
  ([`b5b2345`](https://github.com/NOAA-EMC/mdt/commit/b5b2345fd798979c382024b1d2f99c8e435435d8))

- Remove unnecessary git dependencies from environment configuration
  ([`7694ff8`](https://github.com/NOAA-EMC/mdt/commit/7694ff82609d44973f190813c75f468e6dd24bbd))

- Robust statistics dispatch via monet-stats weighted engine
  ([`ae70dda`](https://github.com/NOAA-EMC/mdt/commit/ae70ddaa874ff56c017c57af8fa55552f5bb47d1))

- Refactor `mdt.tasks.statistics` to robustly delegate weighted MB, MAE, MSE, and RMSE to
  `monet_stats.weighted_spatial_mean`. - Ensure provenance tracking (Rule 2.3) for all statistical
  outputs. - Implement intelligent coordinate mapping for generic `dim` to `lat_dim`/`lon_dim`. -
  Provide `mdt.tasks.reductions.spatial_mean` wrapper. - Add comprehensive Double-Check tests for
  both Dataset and DataArray inputs. - Comply with all ruff formatting and linting rules.

- Update dependencies in environment configuration and pyproject.toml
  ([`0223902`](https://github.com/NOAA-EMC/mdt/commit/022390289cb1e5a05d78f1af3d116e124e4ef926))

- Update dependencies in environment.yml and pyproject.toml for monet packages
  ([`3e62b94`](https://github.com/NOAA-EMC/mdt/commit/3e62b94da6fa745b2e27a5679b72092948f85297))

- **engine**: Add pluggable orchestrator registry and ecFlow support
  ([`1f16674`](https://github.com/NOAA-EMC/mdt/commit/1f16674f2ca1e33cc1b93026e51d5c68f68bda41))

- Add EngineRegistry for dynamic orchestrator backend selection - Implement EcFlowEngine to support
  ecFlow as an orchestration alternative - Add --orchestrator CLI flag to override backend specified
  in config - Refactor config.execution to preserve orchestrator-specific keys - Add optional
  dependency groups for prefect, ecflow, dask, cubed, and dev - Add comprehensive test coverage for
  CLI, config, registry, and ecFlow engine - Update .gitignore to exclude Kiro artifacts and test
  outputs - Enable lazy importing of ecflow to avoid hard dependency

- **stats**: Implement weighted Pearson Correlation in statistics task
  ([`f821a16`](https://github.com/NOAA-EMC/mdt/commit/f821a1634c49d9d7672fec851a90c7bb8469f2b4))

Implemented Weighted Pearson Correlation (CORR/PEARSONR/CORRELATION) in
  `src/mdt/tasks/statistics.py` by orchestrating calls to `monet_stats.weighted_spatial_mean`. This
  implementation follows the Aero Protocol: - Backend-agnostic: Works on both NumPy and Dask (Lazy)
  inputs. - No Hidden Computes: Preserves the Dask graph. - Orchestration: MDT acts as a dispatcher
  to monet_stats. - Provenance: Automatically updates `ds.attrs['history']`. - Validation: Added
  Double-Check test verifying Eager vs Lazy consistency. - Visualization: Provided demo for Track A
  (Static) and Track B (Interactive).

Complies with Pangeo philosophy and ensures accurate metrics on non-uniform geospatial grids.

- **stats**: Implement weighted Pearson Correlation in statistics task
  ([`62bc557`](https://github.com/NOAA-EMC/mdt/commit/62bc55773787fc2c6be8c7329b4c060c4b7f384a))

Implemented Weighted Pearson Correlation (CORR/PEARSONR/CORRELATION) in
  `src/mdt/tasks/statistics.py` using `monet_stats.weighted_spatial_mean`. This implementation
  follows the Aero Protocol: - Backend-agnostic: Works on both NumPy and Dask (Lazy) inputs. - No
  Hidden Computes: Preserves the Dask graph. - Provenance: Automatically updates
  `ds.attrs['history']`. - Validation: Added Double-Check test verifying Eager vs Lazy consistency.
  - Visualization: Provided demo for Track A (Static) and Track B (Interactive).

Complies with Pangeo philosophy and ensures accurate metrics on non-uniform geospatial grids.

- **tasks**: Refactor stats and plotting tasks as orchestrators
  ([`e3a695b`](https://github.com/NOAA-EMC/mdt/commit/e3a695bf01051479ab4d0246a849886715b9e65a))

Refactored `statistics.py` and `plotting.py` to strictly act as orchestrators for the MONET
  ecosystem: - Statistics: Delegated weighted calculations (MB, RMSE, PEARSONR) to
  `monet_stats.weighted_spatial_mean` components. - Plotting: Delegated all visualization to
  `monet-plots` class-based API (`SpatialImshowPlot`, `ScatterPlot`), supporting Track A (Static)
  and Track B (Interactive). - Aero Protocol: Maintained backend-agnosticism and Dask laziness
  across all refactored tasks. - Validation: Added Double-Check tests for weighted correlation and
  provided a comprehensive `visualization_demo.py`. - Documentation: Updated function signatures and
  logic to reflect the orchestration role of MDT.
