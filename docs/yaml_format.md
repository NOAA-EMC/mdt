# MDT YAML Configuration Format

MDT uses a structured YAML format to define verification workflows. Each section of the configuration defines a set of tasks that MDT will execute in order of their dependencies.

## Top-Level Sections

| Section | Description |
| :--- | :--- |
| `data` | Defines model and observation datasets to load. |
| `pairing` | Specifies how datasets are matched in time and space. |
| `combine` | Merges multiple paired datasets for comparison. |
| `statistics` | Lists the verification metrics to compute. |
| `plots` | Configures visualizations. |
| `execution` | Sets up the compute environment (e.g., local or HPC). |

---

## 1. `data` Section

The `data` section defines the sources of information for your verification.

*   **`type`**: The `monetio` dataset name (e.g., `cmaq`, `wrfchem`, `aeronet`, `airnow`).
*   **`kwargs`**: A dictionary of arguments passed to the `monetio` reader's open function.
*   **`zarr_store`**: (Optional) Configuration for VirtualiZarr-based cloud-optimized access.

### VirtualiZarr (Kerchunk / Icechunk)

MDT supports virtualizing large datasets to enable efficient, random access without needing to convert the original files to Zarr. This is particularly useful for large model outputs stored in NetCDF or GRIB format.

*   **`enabled`**: Set to `true` to enable virtualization.
*   **`backend`**: The virtualization engine to use (`kerchunk_json`, `kerchunk_parquet`, `icechunk`, or `zarr`).
*   **`store_path`**: The local or cloud path where the virtual index or Zarr store is located.
*   **`icechunk_url`**: (Required for `icechunk`) The path/URL to the Icechunk repository. This is user/environment specific (do not copy example values literally). Legacy key `icechunk_repo` is still accepted for backward compatibility.
*   **`existing`**: (Optional) If `true`, MDT will attempt to load an existing Zarr or Icechunk store directly instead of generating virtual references.
*   **`zarr_kwargs`**: (Optional) Additional keyword arguments passed to `xarray.open_zarr`.
*   **`max_scan_attempts`**: (Optional) Number of retry attempts when scanning GRIB2 files (default: reader-specific).
*   **`network_timeout`**: (Optional) Network timeout in seconds for remote requests.
*   **`max_concurrent_requests`**: (Optional) Maximum number of concurrent network requests for parallel scanning.

```yaml
data:
  # Example using Kerchunk (JSON)
  cmaq_virtual:
    type: "cmaq"
    kwargs:
      fname: "/path/to/large_cmaq_collection/*.nc"
    zarr_store:
      enabled: true
      backend: "kerchunk_json"
      store_path: "./stores/cmaq_index.json"

  # Example using Icechunk
  gefs_virtual:
    type: "gefs"
    kwargs:
      fname: "s3://noaa-gefs-pds/..."
    zarr_store:
      enabled: true
      backend: "icechunk"
      icechunk_url: "s3://<your-bucket>/<your-gefs-icechunk-repo>"

  # Example pointing to an existing Zarr store (e.g. preprocessed AERONET)
  aeronet_zarr:
    type: "aeronet"  # Type still required by schema
    zarr_store:
      enabled: true
      backend: "zarr"
      existing: true
      store_path: "s3://my-bucket/aeronet-zarr"
```

---

## 2. `pairing` Section

Pairing tasks align two datasets together.

*   **`source`**: The name of the source dataset (usually a model).
*   **`target`**: The name of the target dataset (usually observations).
*   **`method`**: The pairing algorithm (`interpolate` or `regrid`).
*   **`kwargs`**: Additional arguments for the pairing function.

For model-to-model comparison using `regrid`, you can use `merge_target: true` to include the target model in the output.

```yaml
pairing:
  # Model vs Obs (Interpolation)
  cmaq_airnow:
    source: "cmaq_output"
    target: "airnow_obs"
    method: "interpolate"

  # Model vs Model (Regridding)
  cmaq_vs_wrf:
    source: "cmaq_output"
    target: "wrfchem_model"  # Target model defines the grid
    method: "regrid"
    kwargs:
      merge_target: true
      suffix_source: "_cmaq"
      suffix_target: "_wrf"
```

---

## 3. `combine` Section

The `combine` section (optional) merges multiple paired datasets into a single dataset. This is useful for comparing multiple model runs against the same observations.

*   **`sources`**: A list of pairing task names to combine.
*   **`dim`**: The name of the new dimension to create (defaults to `model`).

```yaml
combine:
  all_runs:
    sources:
      - pair_run1
      - pair_run2
    dim: model
```

---

## 4. `statistics` Section

Compute verification metrics on paired data.

*   **`input`**: The name of a pairing or data task.
*   **`metrics`**: A list of metric names to compute (e.g., `rmse`, `bias`, `corr`).
*   **`kwargs`**: Arguments passed to the metric functions (e.g., `obs_var`, `mod_var`).

```yaml
statistics:
  airnow_stats:
    input: "cmaq_airnow"
    metrics: ["rmse", "bias", "corr"]
    kwargs:
      obs_var: "OZONE"
      mod_var: "OZONE"
```

---

## 5. `plots` Section

Generate visualizations.

*   **`input`**: The name of a pairing, statistics, or data task.
*   **`type`**: The type of plot (e.g., `spatial`, `scatter`, `timeseries`).
*   **`kwargs`**: Arguments passed to the plotting function.

```yaml
plots:
  spatial_cmaq:
    input: "cmaq_airnow"
    type: "spatial"
    kwargs:
      savename: "cmaq_map.png"
```

---

## 6. `execution` Section

Configure how and where tasks are executed.

*   **`default_cluster`**: The name of the cluster to use for tasks (defaults to `compute`).
*   **`clusters`**: A dictionary of cluster definitions.

```yaml
execution:
  default_cluster: "local_cpu"
  clusters:
    local_cpu:
      mode: "local"
      n_workers: 4
```

---

## Advanced Example: Multiple Models vs. Observations

This example shows how to compare two different models against a single set of observations.

```yaml
data:
  # Load two different model outputs
  cmaq_model:
    type: "cmaq"
    kwargs:
      fname: "cmaq_data.nc"
  wrfchem_model:
    type: "wrfchem"
    kwargs:
      fname: "wrfchem_data.nc"

  # Load observational data
  airnow_obs:
    type: "airnow"
    kwargs:
      fname: "airnow_data.nc"

pairing:
  # Pair each model to the same observations
  pair_cmaq:
    source: "cmaq_model"
    target: "airnow_obs"
    method: "interpolate"
  pair_wrfchem:
    source: "wrfchem_model"
    target: "airnow_obs"
    method: "interpolate"

statistics:
  # Compute stats for each model
  cmaq_stats:
    input: "pair_cmaq"
    metrics: ["rmse", "bias"]
    kwargs:
      obs_var: "OZONE"
      mod_var: "OZONE"
  wrfchem_stats:
    input: "pair_wrfchem"
    metrics: ["rmse", "bias"]
    kwargs:
      obs_var: "OZONE"
      mod_var: "OZONE"

plots:
  # Create comparative spatial plots
  plot_cmaq:
    input: "pair_cmaq"
    type: "spatial"
    kwargs:
      savename: "cmaq_spatial.png"
  plot_wrfchem:
    input: "pair_wrfchem"
    type: "spatial"
    kwargs:
      savename: "wrfchem_spatial.png"
```
