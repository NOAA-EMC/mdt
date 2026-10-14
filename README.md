# Model Development Tool (MDT)

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://NOAA-EMC.github.io/mdt/)
[![Disclaimer](https://img.shields.io/badge/disclaimer-read%20first-yellow)](DISCLAIMER.md)

The Model Development Tool (MDT) is a powerful, flexible, and scalable environmental verification system. It acts as an orchestrator for the MONET ecosystem, allowing users to define complex verification workflows using simple YAML configurations.

## Features

- **Declarative Workflows**: Define data loading, pairing, statistics, and plotting in a single YAML file.
- **Multiple Orchestrators**: Support for Prefect (for local and cloud execution) and ecFlow (for operational environments).
- **HPC Integration**: Built-in support for NOAA RDHPCS platforms (Hera, Jet, Orion, etc.) via Dask.
- **Dask-Powered Scalability**: Leverages Dask for parallel and distributed computing, adhering to the Aero Protocol for efficient memory management.
- **Two-Track Visualization**: Supports both static (publication-quality) and interactive (exploratory) plots.

## Installation

MDT requires Python 3.11+. We recommend using a Conda/Mamba environment.

```bash
# Create and activate environment
mamba create -n mdt python=3.11
mamba activate mdt

# Install dependencies from MONET ecosystem
pip install git+https://github.com/bbakernoaa/monetio.git@develop
pip install git+https://github.com/bbakernoaa/monet.git@feature/interp_improvements
pip install git+https://github.com/noaa-oar-arl/monet-stats
pip install git+https://github.com/bbakernoaa/monet-plots.git --no-deps

# Install MDT
git clone https://github.com/noaa-oar-arl/mvs.git
cd mvs
pip install -e '.[prefect,dask]'
```

The base install (`pip install -e .`) includes MDT's core dependencies. Add `prefect` for the Prefect runner and `dask` for local distributed or HPC execution; use `pip install -e '.[prefect,dask,dev]'` in a development environment. MDT does not publish an `ecflow` extra. ecFlow Python bindings and the client are supplied by the operational environment where the ecFlow server runs.

## Quick Start

1. **Generate a template configuration:**
   ```bash
   mdt template -o my_config.yaml
   ```

2. **Validate your configuration:**
   ```bash
   mdt validate my_config.yaml
   ```

3. **Run the workflow:**
   ```bash
   mdt run my_config.yaml
   ```

## CLI Usage

```bash
mdt --help
```

- `mdt run <config>`: Execute a verification workflow.
- `mdt validate <config>`: Check a configuration file for errors.
- `mdt template`: Generate a sample configuration file.
- `--version`: Show MDT version.
- `--debug`: Enable debug logging and show full stack traces.

## Documentation

For more detailed information, please visit our [official documentation](https://NOAA-EMC.github.io/mdt/).
