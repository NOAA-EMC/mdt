# HPC Usage Guide

MDT is designed to scale from local workstations to high-performance computing (HPC) clusters. It integrates with `dask-jobqueue` to automatically manage job submission and worker lifecycle on various NOAA platforms.

## Execution Configuration

The `execution` section of your YAML file controls where and how your tasks run.

```yaml
execution:
  default_cluster: "hera_batch"
  clusters:
    hera_batch:
      mode: "hera"
      account: "my_account"
      walltime: "02:00:00"
```

### Supported NOAA Platforms

MDT provides built-in profiles for major NOAA RDHPCS and production platforms. Specifying one of these in the `mode` field will automatically apply optimized defaults for that system.

| Platform | `mode` | Scheduler | Default Cores | Default Memory |
| :--- | :--- | :--- | :--- | :--- |
| **Hera** | `hera` | SLURM | 40 | 120GB |
| **Jet** | `jet` | SLURM | 24 | 60GB |
| **Orion** | `orion` | SLURM | 40 | 180GB |
| **Hercules** | `hercules` | SLURM | 80 | 250GB |
| **Gaea** | `gaea` | SLURM | 36 | 120GB |
| **Ursa** | `ursa` | SLURM | 36 | 120GB |
| **WCOSS2** | `wcoss2` | PBS | 128 | 256GB |

### Generic Schedulers

If you are on a platform not listed above, you can use generic scheduler modes:

*   `slurm`
*   `pbs`
*   `lsf`

## Customizing Cluster Parameters

You can override any platform default by providing additional keys in the cluster configuration. These are passed directly to the underlying `dask-jobqueue` cluster class (e.g., `SLURMCluster` or `PBSCluster`).

```yaml
execution:
  clusters:
    custom_jet:
      mode: "jet"
      cores: 12
      memory: "30GB"
      job_extra_directives: ["--qos=windfall"]
```

## Assigning Tasks to Specific Clusters

MDT allows you to run different tasks on different clusters within the same workflow. For example, you might want to load data on a local cluster but perform heavy pairing and statistics on an HPC cluster.

```yaml
data:
  obs_data:
    type: "airnow"
    cluster: "local"  # Run locally
    kwargs:
      fname: "obs.nc"

pairing:
  heavy_pairing:
    source: "model_data"
    target: "obs_data"
    cluster: "hera_batch"  # Run on HPC
    method: "regrid"
```

If no `cluster` is specified for a task, it uses the `default_cluster` defined in the `execution` section.
