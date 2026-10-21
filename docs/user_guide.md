# User Guide

This guide describes the core architecture and usage principles of MDT.

## Core Concepts

MDT is designed to handle the complexity of atmospheric model verification workflows. It uses a **Directed Acyclic Graph (DAG)** to model dependencies between tasks.

### Tasks and Workflows

MDT breaks down verification into several key task types:

1.  **Data Loading:** Using `monetio` to read model output (e.g., CMAQ, WRF-Chem, GEOS-Chem) and observational data (e.g., AERONET, IMPROVE, EPA AirNow).
2.  **Pairing:** Aligning model and observation data in time and space. This may involve:
    *   **Interpolation:** Extracting model values at specific observation point locations.
    *   **Regridding:** Mapping model output to a standard reference grid using `xregrid`.
3.  **Combining (Optional):** Merging multiple paired datasets into a single object to facilitate inter-model comparisons.
4.  **Statistics:** Computing standard verification metrics (RMSE, Bias, Correlation, etc.) using `monet-stats`.
5.  **Plotting:** Visualizing the results using `monet-plots`.

### DAG Construction

When you run MDT with a configuration file, it automatically builds a dependency graph. For example, a `statistics` task that depends on a `pairing` task will wait for that pairing task to complete before starting. This allows MDT to optimize the execution order and run independent tasks in parallel.

## Scalability and HPC Support

MDT is built on **Prefect**, which provides a robust engine for task scheduling and monitoring. For large-scale verification tasks, MDT supports multiple execution clusters:

*   **Local:** Use local CPU cores for processing.
*   **Dask (SLURM/PBS/LSF):** Scale out to multiple nodes on an HPC cluster.

MDT's backend-agnostic design ensures that it can efficiently handle "big data" using **Dask** without requiring manual code changes.

## Statistical weights

When statistics specify `kwargs.weights`, MDT requires that weight variable to
exist and that the selected MONET metric support weights. Unsupported weighting
now raises an error instead of silently changing the calculation.

For workflows that deliberately accept an unweighted result for unsupported
metrics, set `kwargs.allow_unweighted_fallback: true`. MDT logs a warning and
records the unweighted calculation in the result's history. Missing named weight
variables still raise an error, even with fallback enabled. Existing workflows
that relied on implicit fallback must choose a weighted metric or explicitly
opt in to this behavior. MDT continues to delegate calculations to `monet-stats`.
