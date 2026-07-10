"""Prefect execution engine and task wrappers for MDT."""

import logging
import subprocess
from typing import TYPE_CHECKING, Any, Dict, List, cast

import networkx as nx

from mdt.engine_registry import Engine
from mdt.hpc import HPCProfileFactory
from mdt.tasks.data import load_data, save_data
from mdt.tasks.pairing import combine_paired_data, pair_data
from mdt.tasks.plotting import generate_plot
from mdt.tasks.reductions import calculate_reduction
from mdt.tasks.statistics import compute_statistics

if TYPE_CHECKING:
    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)


def prefect_calculate_reduction(
    name: str,
    data: Any,
    method: str,
    dim: Any,
    force_weighted: bool,
    kwargs: Dict[str, Any],
) -> Any:
    """Prefect task wrapper for calculate_reduction."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Reducing dataset '{name}' using method '{method}' over dim: {dim}")
    return calculate_reduction(data, method, dim, force_weighted, **kwargs)


def prefect_save_data(name: str, data: Any, backend: str, url: str, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for save_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Saving dataset '{name}' using backend '{backend}' to '{url}'")
    return save_data(name, data, backend, url, kwargs)


def prefect_load_data(name: str, dataset_type: str, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for load_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    use_virtualizarr = kwargs.get("use_virtualizarr", False)
    if use_virtualizarr:
        backend = kwargs.get("virtualizarr_backend", "N/A")
        store_path = kwargs.get("store_path", "N/A")
        icechunk_repo = kwargs.get("icechunk_repo", "N/A")
        logger.info(
            f"Loading data with VirtualiZarr: {name} "
            f"(type: {dataset_type}, backend: {backend}, store: {store_path}, icechunk_repo: {icechunk_repo})"
        )
    else:
        logger.info(f"Loading data: {name} of type {dataset_type}")
    return load_data(name, dataset_type, kwargs)


def prefect_pair_data(name: str, method: str, source_data: Any, target_data: Any, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for pair_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Pairing data: {name} using {method}")
    return pair_data(name, method, source_data, target_data, kwargs)


def prefect_combine_paired_data(paired_data: Dict[str, Any], dim: str = "model") -> Any:
    """Prefect task wrapper for combine_paired_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Combining {len(paired_data)} paired datasets along '{dim}'")
    return combine_paired_data(paired_data, dim=dim)


def prefect_compute_statistics(name: str, metrics: List[str], input_data: Any, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for compute_statistics."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Computing statistics: {name} for metrics {metrics}")
    return compute_statistics(name, metrics, input_data, kwargs)


def prefect_generate_plot(name: str, plot_type: str, input_data: Any, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for generate_plot."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Generating plot: {name} of type {plot_type}")
    return generate_plot(name, plot_type, input_data, kwargs)


class PrefectEngine(Engine):
    """Executes a NetworkX DAG as a Prefect Flow."""

    def __init__(self, dag: nx.DiGraph, config: "ConfigParser"):
        self.dag = dag
        self.config = config
        self.clusters: Dict[str, Any] = {}

    def _setup_dask_clusters(self) -> Any:
        """
        Initializes Dask clusters based on the execution configuration.

        It starts a primary LocalCluster (scheduler) and then scales HPC clusters
        which connect their workers back to this primary scheduler.
        Each HPC cluster's workers are tagged with the cluster name using `resources`.
        """
        exec_cfg = self.config.execution
        clusters_cfg = exec_cfg.get("clusters", {})

        # If there's only one cluster and it's local, keep it simple
        import dask.distributed

        if len(clusters_cfg) == 1 and next(iter(clusters_cfg.values())).get("mode", "local") == "local":
            logger.info("Setting up a single local Dask cluster.")
            cluster_name = next(iter(clusters_cfg.keys()))
            cluster_cfg = next(iter(clusters_cfg.values()))
            workers = cluster_cfg.get("workers", 1)
            memory_limit = cluster_cfg.get("memory_limit", "auto")
            # Assign the requested numeric resource to the local workers to prevent hanging
            return dask.distributed.LocalCluster(
                n_workers=workers,
                resources={cluster_name.upper(): 1},
                memory_limit=memory_limit,
            )

        # Multi-cluster or HPC mode:
        # Create a central scheduler on the node where MDT is executed.
        logger.info("Setting up central Dask Scheduler.")
        primary_cluster = dask.distributed.LocalCluster(
            n_workers=0,  # The local cluster is just the scheduler
            host="0.0.0.0",  # noqa: S104 — Allow external connections
        )
        scheduler_address = primary_cluster.scheduler_address

        for cluster_name, cfg in clusters_cfg.items():
            mode = cfg.get("mode", "local")
            workers = cfg.get("workers", 1)
            kwargs = cfg.get("cluster_kwargs", {})

            # Let the HPCProfileFactory know the logical name of this cluster (e.g. 'service')
            # so it can choose appropriate partitions.
            kwargs["cluster_name"] = cluster_name

            # Assign resource annotations so Dask knows which workers belong to which cluster.
            res_name = cluster_name.upper()

            if mode == "local":
                logger.info(f"Scaling local workers for '{cluster_name}'")
                # Scale the central cluster and explicitly assign resources
                # We use scale(cores=...) or similar if possible, but scale(workers)
                # is standard for LocalCluster.
                primary_cluster.scale(workers)
                # Dask workers started via scale() on LocalCluster might not automatically
                # get the resource tag. For LocalCluster we usually set it in the constructor.
                # However, for multi-cluster we rely on the resources dict if possible.
            else:
                logger.info(f"Setting up HPC cluster '{cluster_name}' (mode: {mode}) connecting to {scheduler_address}")

                # To support multiple disparate HPC clusters sharing a single DAG and scheduler,
                # we bypass `dask-jobqueue`'s automatic localized scheduler creation to avoid
                # port conflicts ("Address already in use"). We construct an isolated jobqueue
                # object on a random port, extract its generated batch script template, inject
                # our central scheduler address, and manually submit the job script.

                # Tag the workers with this cluster's name as a numeric resource
                if "env_extra" not in kwargs:
                    kwargs["env_extra"] = []
                kwargs["env_extra"].append(f"export DASK_DISTRIBUTED__WORKER__RESOURCES__{res_name}=1")

                # Force the dummy scheduler to a random port to avoid conflicts
                kwargs["scheduler_options"] = {"port": 0}

                # Create the JobQueue object representing this specific profile/queue
                hpc_cluster = HPCProfileFactory.create_cluster(mode, **kwargs)

                # Retrieve the underlying job script formatted for the specific batch system
                hpc_job_script = hpc_cluster.job_script()

                # Dask job scripts contain a variable representing the scheduler it connects to.
                # E.g., `dask-worker tcp://127.0.0.1:8786 ...`
                # We replace this dummy scheduler address with our central primary scheduler address.
                dummy_address = hpc_cluster.scheduler_address
                hpc_job_script = hpc_job_script.replace(dummy_address, scheduler_address)

                # Determine the submission command based on the mode (slurm -> sbatch, pbs -> qsub)
                submit_cmd = "sbatch" if "slurm" in mode or mode in ["hera", "jet", "orion", "hercules", "gaea", "ursa"] else "qsub"
                if mode == "lsf":
                    submit_cmd = "bsub"

                # Manually submit the modified script for the requested number of workers
                logger.info(f"Submitting {workers} {submit_cmd} jobs to '{cluster_name}' queue.")
                for _ in range(workers):
                    try:
                        subprocess.run([submit_cmd], input=hpc_job_script.encode("utf-8"), check=True)  # noqa: S603
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to submit HPC worker job for cluster {cluster_name}: {e}")
                        raise

                self.clusters[cluster_name] = hpc_cluster

        return primary_cluster

    def execute(self) -> Dict[str, Any]:
        """
        Builds the Dask cluster topology and executes the Prefect flow.

        Returns
        -------
        dict
            A dictionary mapping node IDs from the task graph to the Dask futures
            representing their completion state and results.
        """
        from prefect import flow, task

        # Prefect Task Wrappers — defined here so the @task decorator is only
        # evaluated when Prefect is actually installed and execute() is called.
        # Requirement 3.5: We maintain lazy imports.
        from prefect.cache_policies import NONE as NO_CACHE

        p_load_data = task(name="Load Data", cache_policy=NO_CACHE)(prefect_load_data)
        p_pair_data = task(name="Pair Data", cache_policy=NO_CACHE)(prefect_pair_data)
        p_combine_paired_data = task(name="Combine Paired Data", cache_policy=NO_CACHE)(prefect_combine_paired_data)
        p_compute_statistics = task(name="Compute Statistics", cache_policy=NO_CACHE)(prefect_compute_statistics)
        p_generate_plot = task(name="Generate Plot", cache_policy=NO_CACHE)(prefect_generate_plot)
        p_calculate_reduction = task(name="Calculate Reduction", cache_policy=NO_CACHE)(prefect_calculate_reduction)
        p_save_data = task(name="Save Data", cache_policy=NO_CACHE)(prefect_save_data)

        from contextlib import nullcontext

        import dask

        # Determine if we need Dask resource annotations
        exec_cfg = self.config.execution
        clusters_cfg = exec_cfg.get("clusters", {})
        use_dask_runner = len(clusters_cfg) > 1 or any(
            cfg.get("mode", "local") != "local" or cfg.get("workers", 1) > 1 for cfg in clusters_cfg.values()
        )
        serialize_pair_tasks = all(cfg.get("mode", "local") == "local" for cfg in clusters_cfg.values())

        def _resolve_output(value: Any) -> Any:
            if hasattr(value, "result") and callable(value.result):
                return value.result()
            return value

        # Snapshot mutable engine state into local variables so the flow closure
        # does not capture `self` (which may hold non-picklable Dask client state).
        dag = self.dag

        # Define the Prefect flow inline to capture the instance variables
        @flow(name="MDT Verification Workflow")  # type: ignore
        def mdt_flow() -> Dict[str, Any]:
            # Dictionary to store the output futures of each task
            task_outputs: Dict[str, Any] = {}
            last_pair_future: Any = None

            # Use topological sort to process nodes in the correct dependency order
            for node_id in nx.topological_sort(dag):
                node_data = dag.nodes[node_id]
                task_type = node_data["task_type"]
                target_cluster = node_data.get("cluster")

                # Retrieve the inputs from the dependencies that have already run
                predecessors = list(dag.predecessors(node_id))

                # We will use Prefect's `.with_options` to inject standard Dask resources.
                # If target_cluster is None, default to "COMPUTE"
                res_key = target_cluster.upper() if target_cluster else "COMPUTE"
                logger.debug(f"Routing task '{node_id}' to cluster resource: {res_key}")

                # Route the task to the specific worker pool via Dask Resource Annotation
                # Only use dask.annotate when running with DaskTaskRunner
                annotation_ctx = dask.annotate(resources={res_key: 1}) if use_dask_runner else nullcontext()
                with annotation_ctx:
                    if task_type == "load_data":
                        name = node_data["name"]
                        kwargs = node_data["kwargs"]
                        use_virtualizarr = kwargs.get("use_virtualizarr", False)

                        task_options: Dict[str, Any] = {}
                        if use_virtualizarr:
                            task_options["tags"] = ["virtualizarr"]
                            task_options["task_run_name"] = f"Load Data: {name} [VirtualiZarr]"

                        future = p_load_data.with_options(**task_options).submit(name=name, dataset_type=node_data["dataset_type"], kwargs=kwargs)
                        task_outputs[node_id] = future

                    elif task_type == "pair_data":
                        # Resolve source and target node IDs from DAG metadata
                        source_name = node_data.get("source_name")
                        target_name = node_data.get("target_name")

                        # Resolve node IDs from the actual predecessors in the DAG
                        # We look for nodes where the 'name' attribute matches the requested source/target names
                        source_node_id = next((p for p in predecessors if dag.nodes[p].get("name") == source_name), None)
                        target_node_id = next((p for p in predecessors if dag.nodes[p].get("name") == target_name), None)

                        if not source_node_id or not target_node_id:
                            logger.error(
                                f"Failed to resolve predecessors for pairing '{node_id}'. Source: {source_node_id}, Target: {target_node_id}"
                            )

                        source_output = task_outputs.get(source_node_id) if source_node_id else None
                        target_output = task_outputs.get(target_node_id) if target_node_id else None

                        if serialize_pair_tasks:
                            if last_pair_future is not None:
                                _resolve_output(last_pair_future)

                            # Keep Monet pairing on the flow thread for local runs.
                            # This avoids ESMF VM/thread initialization issues in Prefect workers
                            # while preserving the same pairing implementation path.
                            pair_result = prefect_pair_data(
                                name=node_data["name"],
                                method=node_data["method"],
                                source_data=_resolve_output(source_output),
                                target_data=_resolve_output(target_output),
                                kwargs=node_data["kwargs"],
                            )
                            task_outputs[node_id] = pair_result
                            last_pair_future = pair_result
                        else:
                            future = p_pair_data.submit(
                                name=node_data["name"],
                                method=node_data["method"],
                                source_data=source_output,
                                target_data=target_output,
                                kwargs=node_data["kwargs"],
                            )
                            task_outputs[node_id] = future

                    elif task_type == "combine_paired_data":
                        # Collect all inputs from predecessors (pairing tasks)
                        paired_data_inputs = {}
                        # Get the list of pairing names executed
                        sources = node_data.get("sources", [])

                        for source_pair_name in sources:
                            # The node ID is constructed as pair_{name} in DAGBuilder
                            pred_node_id = f"pair_{source_pair_name}"
                            if pred_node_id in task_outputs:
                                # Use the pairing name (or a label) as the key
                                paired_data_inputs[source_pair_name] = task_outputs[pred_node_id]
                            else:
                                logger.warning(f"Predecessor {pred_node_id} not found for combine task {node_id}")

                        future = p_combine_paired_data.submit(
                            paired_data=paired_data_inputs,
                            dim=node_data.get("dim", "model"),
                        )
                        task_outputs[node_id] = future

                    elif task_type == "compute_statistics":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_compute_statistics.submit(
                            name=node_data["name"],
                            metrics=node_data["metrics"],
                            input_data=task_outputs.get(input_node) if input_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "generate_plot":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_generate_plot.submit(
                            name=node_data["name"],
                            plot_type=node_data["plot_type"],
                            input_data=task_outputs.get(input_node) if input_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "calculate_reduction":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_calculate_reduction.submit(
                            name=node_data["name"],
                            data=task_outputs.get(input_node) if input_node else None,
                            method=node_data["method"],
                            dim=node_data["dim"],
                            force_weighted=node_data["force_weighted"],
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "save_data":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_save_data.submit(
                            name=node_data["name"],
                            data=task_outputs.get(input_node) if input_node else None,
                            backend=node_data["backend"],
                            url=node_data["url"],
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

            logger.info("All tasks submitted to Prefect.")
            return task_outputs

        # Setup execution environment
        logger.info("Starting Prefect flow execution...")
        if use_dask_runner:
            # Setup Dask clusters and configure Prefect to use the central scheduler
            from prefect_dask.task_runners import DaskTaskRunner

            cluster = self._setup_dask_clusters()
            logger.info(f"Central Dask Scheduler address: {cluster.scheduler_address}")
            task_futures: Dict[str, Any] = mdt_flow.with_options(task_runner=cast(Any, DaskTaskRunner(address=cluster.scheduler_address)))()
        else:
            # Single local worker — use ConcurrentTaskRunner (no Dask serialization overhead)
            from prefect.task_runners import ConcurrentTaskRunner

            task_futures = mdt_flow.with_options(task_runner=ConcurrentTaskRunner())()

        # Requirement: Ensure the CLI waits for task completion.
        # Prefect .submit() returns futures. We need to wait for them.
        logger.info("Waiting for tasks to complete...")
        final_results = {}
        for node_id, future in task_futures.items():
            try:
                # This will block until the specific task is done.
                final_results[node_id] = _resolve_output(future)
            except Exception as e:
                logger.error(f"Task {node_id} failed: {e}")
                final_results[node_id] = e

        logger.info("All tasks completed.")
        return final_results
