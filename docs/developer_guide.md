# Developer Guide

This guide is intended for developers who wish to contribute to MDT or understand its internal architecture.

## Architecture Overview

MDT is designed as a modular orchestration layer. It translates a declarative YAML configuration into a Directed Acyclic Graph (DAG) of tasks and then delegates the execution of those tasks to a pluggable orchestration engine (like Prefect or ecFlow).

The following diagram illustrates the high-level flow of an MDT execution:

```mermaid
graph TD
    A[YAML Config] --> B[ConfigParser]
    B --> C[DAGBuilder]
    C --> D[networkx.DiGraph]
    D --> E[EngineRegistry]
    E --> F{Orchestrator Engine}
    F -->|Prefect| G[PrefectEngine]
    F -->|ecFlow| H[EcFlowEngine]
    G --> I[Dask Clusters]
    H --> J[ecFlow Server]
    I --> K[MONET Tasks]
    J --> K
```

## Core Components

### 1. Configuration Parsing (`mdt.config`)

The `ConfigParser` class is responsible for reading the YAML configuration, applying defaults, and performing initial validation. It ensures that the required sections (`data`, `execution`) exist and that task definitions are structurally sound.

### 2. DAG Construction (`mdt.dag`)

The `DAGBuilder` takes the validated configuration and constructs a `networkx.DiGraph`.
- **Nodes** represent tasks (loading, pairing, statistics, etc.).
- **Edges** represent data dependencies.
- **Attributes** on nodes store the parameters (kwargs) and execution requirements (e.g., cluster/partition).

### 3. Engine Abstraction (`mdt.engine_registry`)

MDT uses an abstract `Engine` class to decouple the workflow logic from the underlying orchestrator. This allows MDT to support diverse environments—from a researcher's laptop using Prefect/Dask to operational NOAA supercomputers using ecFlow.

### 4. Task Delegation (`mdt.tasks`)

MDT acts strictly as an orchestrator. It does not contain heavy scientific logic. Instead, it delegates to the MONET ecosystem.

#### VirtualiZarr Integration
For large-scale data handling, MDT integrates **VirtualiZarr**. When enabled in the `data` configuration, MDT uses `kerchunk` or `icechunk` to create a virtual Zarr representation of the source files. This allows the orchestrator to:
- Perform metadata-only discovery of massive datasets.
- Distribute data-parallel tasks more efficiently across Dask workers.
- Avoid expensive data conversion or staging steps.

The primary delegation remains:
- **Data Loading**: `monetio`
- **Pairing**: `monet`
- **Statistics**: `monet-stats`
- **Plotting**: `monet-plots`

## Engine Implementation Details

MDT supports different workflow paths through its Engine abstraction. The two primary implementations are the `PrefectEngine` and the `EcFlowEngine`.

### Prefect + Dask Path

The Prefect path is optimized for dynamic, pythonic workflows and scales using Dask.

1.  **Dask Cluster Initialization**: MDT starts a central Dask scheduler. If HPC profiles are requested, it manually submits worker jobs to the batch system (SLURM/PBS) that connect back to this central scheduler.
2.  **Resource Tagging**: Tasks are submitted to Prefect with Dask resource annotations (e.g., `resources={'COMPUTE': 1}`). This ensures tasks run on the appropriate hardware (e.g., download tasks on service nodes, heavy math on compute nodes).
3.  **Lazy Execution**: Prefect manages the task futures, and Dask handles the actual distributed computation.

```mermaid
graph TD
    subgraph "Prefect Orchestration"
        A[Prefect Flow] --> B[Task Submission]
        B --> C{Resource Tag?}
        C -->|SERVICE| D[Service Workers]
        C -->|COMPUTE| E[Compute Workers]
    end
    subgraph "Dask Distributed"
        D --> F[monetio.load]
        E --> G[monet.pair]
        E --> H[monet_stats.compute]
    end
```

### ecFlow Path

The ecFlow path is designed for operational environments where workflow state is managed by a centralized ecFlow server.

1.  **Suite Definition**: MDT translates the NetworkX DAG into an `ecflow.Defs` structure. Each DAG node becomes an `ecflow.Task` within an `ecflow.Family`.
2.  **Trigger Logic**: Data dependencies from the DAG are converted into ecFlow trigger expressions (e.g., `trigger ./load_data == complete`).
3.  **Wrapper Generation**: MDT generates `.ecf` Python scripts for every task. These scripts act as thin wrappers that call the MDT task functions and report status back to the server.
4.  **Submission**: The entire suite is loaded into the ecFlow server and started.

```mermaid
graph LR
    subgraph "MDT (Local)"
        A[nx.DiGraph] --> B[Suite Builder]
        A --> C[Wrapper Gen]
    end
    subgraph "ecFlow Server"
        B --> D[Suite Definition]
        D --> E[Task 1]
        D --> F[Task 2]
        E -.->|Trigger| F
    end
    subgraph "Execution Node"
        E --> G[.ecf Script]
        F --> H[.ecf Script]
        G --> I[MDT Tasks]
        H --> I
    end
```

## Execution Sequence

The following sequence diagram shows the interaction between components during a `mdt run` command:

```mermaid
sequenceDiagram
    participant CLI as mdt CLI
    participant CP as ConfigParser
    participant DB as DAGBuilder
    participant ER as EngineRegistry
    participant EN as Engine (Prefect/ecFlow)
    participant T as Tasks (MONET)

    CLI->>CP: load_config(path)
    CP-->>CLI: config object
    CLI->>DB: DAGBuilder(config).build()
    DB-->>CLI: nx.DiGraph
    CLI->>ER: get_engine(orchestrator_name)
    ER-->>CLI: Engine Class
    CLI->>EN: execute(dag)
    activate EN
    EN->>T: Dispatch Task 1
    T-->>EN: Result 1
    EN->>T: Dispatch Task 2 (depends on 1)
    T-->>EN: Result 2
    deactivate EN
    CLI->>CLI: Exit Success
```

## Adding New Features

### Adding a New Task Type
1. Define the core logic in a new or existing module in `src/mdt/tasks/`.
2. Update `src/mdt/dag.py` to recognize the new configuration section and add corresponding nodes to the graph.
3. Update the Engine implementations (`src/mdt/engine.py` and `src/mdt/ecflow_engine.py`) to handle the new task type.

### Adding a New Engine
1. Subclass `mdt.engine_registry.Engine`.
2. Implement the `execute()` method to translate the NetworkX DAG into the new orchestrator's native format.
3. Register the new engine in `src/mdt/engine_registry.py`.

## Reproducible development environment

Activate the Conda environment described in the quick start, then run:

```bash
uv sync --active --frozen --all-extras --inexact
uv run --active --frozen --all-extras --inexact ruff check src/ tests/
uv run --active --frozen --all-extras --inexact mypy src/mdt
uv run --active --frozen --all-extras --inexact pytest tests/
```

The four direct MONET Git dependencies use immutable revisions. When updating
them, change the revision in `pyproject.toml`, run `uv lock`, and commit the
manifest and lockfile together after testing. CI uses the same frozen lockfile.
The lightweight lint job installs only the locked `lint` dependency group.

See [Testing MDT](testing.md) for offline checks and explicit live integration opt-ins.
