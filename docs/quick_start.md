# Quick Start Guide

This guide will help you set up and run your first MDT verification workflow.

## Installation

MDT requires Python 3.11+ and several components from the MONET ecosystem. For the best experience, we recommend using a Conda or Mamba environment.

```bash
# Create and activate environment
mamba create -n mdt -c conda-forge python=3.11 esmpy uv c-compiler
mamba activate mdt

# Install MDT with tested dependencies from the committed lockfile
git clone https://github.com/NOAA-EMC/mdt.git
cd mdt
uv sync --active --frozen --extra prefect --extra dask --inexact
```

For development, add `--extra dev` to the `uv sync` command. MDT's `prefect` and `dask` extras provide the respective optional runtime packages. MDT does not publish an `ecflow` extra: ecFlow Python bindings and the client are provided by the operational environment.

The default pairing mode is `serial` to avoid concurrent ESMF calls. Set `execution.pairing_mode: concurrent` only when parallel ESMF pairing is supported in your environment.

## Running MDT

MDT is configured via a YAML file. You can use the `mdt` CLI to generate templates, validate configurations, and run workflows.

```bash
# 1. Generate a template configuration:
mdt template -o my_config.yaml

# 2. Validate your configuration:
mdt validate my_config.yaml

# 3. Run the workflow:
mdt run my_config.yaml
```

## A Minimal Working Example

Create a file named `simple_eval.yaml` with the following content (adjust paths as needed):

```yaml
data:
  my_model:
    type: "merra2"
    kwargs:
      dates: "2023-01-01"
      product: "inst1_2d_asm_Nx"

  my_obs:
    type: "aeronet"
    kwargs:
      dates: "2023-01-01"

pairing:
  eval_pair:
    source: "my_model"
    target: "my_obs"
    method: "nearest"

statistics:
  basic_stats:
    input: "eval_pair"
    metrics: ["rmse", "mb", "corr"]
    kwargs:
      obs_var: "aod_550nm"
      mod_var: "AOD"

plots:
  spatial_eval:
    input: "eval_pair"
    type: "spatial"
    kwargs:
      savename: "merra2_vs_aeronet.png"
```

Then run it:

```bash
mdt run simple_eval.yaml
```

MDT will automatically load the data, pair the model output to the observation locations, compute the requested statistics, and save a spatial plot.

The locked `uv sync` command reproduces the tested dependency set. `--active`
uses the activated Conda environment and `--inexact` retains its ESMF bindings.
A C compiler is required to build the MONET interpolation dependency. Ordinary
`pip install -e ".[prefect,dask]"` pins the direct MONET revisions but does not
lock all transitive dependencies. For ecFlow execution, install `ecflow` from
Conda Forge in the operational environment; MDT does not install its bindings.
