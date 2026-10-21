# Workflow Examples

This section provides complete YAML configuration examples for common verification workflows in MDT.

## Multi-Model Evaluation vs. AERONET

This example demonstrates how to compare two different aerosol products (**MERRA-2** and **ICAP-MME**) against **AERONET** ground observations. It also includes a direct model-to-model bias analysis.

### Features
- Loading gridded model data (MERRA-2 and ICAP).
- Loading point observations (AERONET).
- Pairing models to observation sites using interpolation.
- Direct model-to-model pairing via regridding.
- Computing RMSE and Mean Bias (MB).
- Generating spatial comparison and bias maps.

### YAML Configuration

```yaml
# docs/examples/merra2_icap_aeronet_comparison.yaml

data:
  merra2_model:
    type: "merra2"
    kwargs:
      dates: "2023-01-01"
      product: "inst1_2d_asm_Nx"

  icap_mme_model:
    type: "icap_mme"
    kwargs:
      date: "2023-01-01"
      product: "MMC"
      data_var: "modeaod550"

  aeronet_obs:
    type: "aeronet"
    kwargs:
      dates: "2023-01-01"

pairing:
  # Model vs Obs (Interpolation to AERONET sites)
  pair_merra2_obs:
    source: "merra2_model"
    target: "aeronet_obs"
    method: "nearest"

  pair_icap_obs:
    source: "icap_mme_model"
    target: "aeronet_obs"
    method: "nearest"

  # Model vs Model (Regrid MERRA-2 to ICAP-MME grid for bias plot)
  pair_merra2_icap:
    source: "merra2_model"
    target: "icap_mme_model"
    method: "bilinear"
    kwargs:
      merge: true
      suffix: "_icap"

combine:
  obs_comparison:
    sources:
      - pair_merra2_obs
      - pair_icap_obs
    dim: "model"

statistics:
  # Stats against AERONET
  obs_stats:
    input: "obs_comparison"
    metrics: ["rmse", "mb", "corr"]
    kwargs:
      obs_var: "aod_550nm"
      mod_var: "AOD"

  # Direct Model Bias (Gridded)
  model_bias_stats:
    input: "pair_merra2_icap"
    metrics: ["mb"]
    kwargs:
      obs_var: "modeaod550"  # ICAP acts as reference
      mod_var: "TOTEXTTAU"

plots:
  # Map of model performance at observation sites
  obs_spatial_map:
    input: "obs_comparison"
    type: "spatial"
    kwargs:
      savename: "obs_comparison_map.png"

  # Spatial bias map: MERRA-2 vs ICAP-MME
  model_bias_map:
    input: "model_bias_stats"
    type: "spatial"
    kwargs:
      savename: "merra2_vs_icap_bias.png"
      cmap: "RdBu_r"

execution:
  default_cluster: "local"
  clusters:
    local:
      mode: "local"
      workers: 4
```

## GEFS-Aerosol vs. AERONET

This example demonstrates how to evaluate **GEFS-Aerosol** model output against **AERONET** ground-based sun photometer observations.

### YAML Configuration

```yaml
# docs/examples/gefs_aeronet.yaml

data:
  gefs_model:
    type: "gefs"
    kwargs:
      dates: "2023-08-01"
      product: "aerosol"
      filters:
        typeOfFirstFixedSurface: 10
    zarr_store:
      enabled: true
      backend: "icechunk"
      icechunk_url: "s3://<your-bucket>/<your-gefs-icechunk-repo>"
      max_scan_attempts: 5
      network_timeout: 60.0
      max_concurrent_requests: 16

  aeronet_obs:
    type: "aeronet"
    kwargs:
      dates: "2023-08-01"

pairing:
  pair_gefs_aeronet:
    source: "gefs_model"
    target: "aeronet_obs"
    method: "nearest"

statistics:
  aod_stats:
    input: "pair_gefs_aeronet"
    metrics: ["rmse", "mb", "corr"]
    kwargs:
      obs_var: "aod_550nm"
      mod_var: "dust" # Example variable

plots:
  spatial_aod:
    input: "pair_gefs_aeronet"
    type: "spatial_bias_scatter"
    kwargs:
      savename: "gefs_aeronet_spatial.png"
      col1: "aod_550nm"
      col2: "totAOD550"
      fact: 0.2
      vmin: -0.5
      vmax: 0.5

  timeseries_aod:
    input: "pair_gefs_aeronet"
    type: "timeseries"
    kwargs:
      savename: "gefs_aeronet_ts.png"
```

## GFS vs. ISH-Lite

This example compares **GFS** meteorological output with **ISH-Lite** (Integrated Surface Database Lite) observations for surface temperature.

### YAML Configuration

```yaml
# docs/examples/gfs_ish_lite.yaml

data:
  gfs_model:
    type: "gfs"
    kwargs:
      dates: "2023-08-01"
      filters:
        typeOfFirstFixedSurface: 103
    zarr_store:
      enabled: true
      backend: "icechunk"
      icechunk_url: "s3://<your-bucket>/<your-gfs-icechunk-repo>"
      max_scan_attempts: 5
      network_timeout: 60.0
      max_concurrent_requests: 16

  ish_lite_obs:
    type: "ish_lite"
    kwargs:
      dates: "2023-08-01"

pairing:
  pair_gfs_ish:
    source: "gfs_model"
    target: "ish_lite_obs"
    method: "nearest"

statistics:
  met_stats:
    input: "pair_gfs_ish"
    metrics: ["rmse", "mb"]
    kwargs:
      obs_var: "t2m"
      mod_var: "TMP_2maboveground"

plots:
  timeseries_temp:
    input: "pair_gfs_ish"
    type: "timeseries"
    kwargs:
      savename: "gfs_ish_temp_ts.png"
      column: "t2m"
```
