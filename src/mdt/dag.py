import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)


class DAGBuilder:
    """Builds a NetworkX Directed Acyclic Graph based on MDT configuration."""

    def __init__(self, config: "ConfigParser"):
        self.config = config
        self.graph: nx.DiGraph = nx.DiGraph()

    def build(self) -> nx.DiGraph:
        """
        Constructs and validates the dependency graph.

        Returns
        -------
        networkx.DiGraph
            The fully constructed Directed Acyclic Graph representing the
            execution workflow.

        Raises
        ------
        ValueError
            If the generated graph contains cycles and is not a valid DAG.
        """
        logger.info("Building task graph...")
        self._add_data_nodes()
        self._add_pairing_nodes()
        self._add_combine_nodes()
        self._add_reduction_nodes()
        self._add_statistics_nodes()
        self._add_plotting_nodes()
        self._add_save_nodes()

        # Validate that the graph is indeed a DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The generated task graph contains cycles and is not a valid DAG.")

        logger.info(f"Built task graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph

    def _add_data_nodes(self) -> None:
        """Adds nodes for loading data (from monetio)."""
        data_cfg = self.config.data
        if not data_cfg:
            logger.warning("No 'data' section found in configuration.")
            return

        # Data retrieval typically requires external internet access.
        # Compute nodes on HPC environments (e.g. NOAA RDHPCS) typically block this.
        # Therefore, data tasks should default to a 'service' cluster representing the service node.
        default_cluster = "service"

        for name, details in data_cfg.items():
            node_id = f"load_{name}"
            # Type specifies the reader to use from monetio.readers
            dataset_type = details.get("type")
            if not dataset_type:
                logger.warning(f"Data source '{name}' is missing 'type'. Skipping.")
                continue

            kwargs: dict[str, Any] = details.get("kwargs") or {}
            if "use_kerchunk" in details:
                kwargs["use_kerchunk"] = details.get("use_kerchunk")
            if "kerchunk_file" in details:
                kwargs["kerchunk_file"] = details.get("kerchunk_file")

            zarr_store = details.get("zarr_store", {})
            if zarr_store.get("enabled", False):
                backend = zarr_store.get("backend", "kerchunk_json")
                if backend == "icechunk":
                    kwargs["use_icechunk"] = True
                    icechunk_url = zarr_store.get("icechunk_url") or zarr_store.get("icechunk_repo")
                    if icechunk_url:
                        kwargs["icechunk_url"] = icechunk_url
                else:
                    kwargs["use_virtualizarr"] = True
                    kwargs["virtualizarr_backend"] = backend
                    kwargs["store_path"] = zarr_store.get("store_path", f"./zarr_stores/{name}/")

                # Pass through robustness parameters if present
                for param in ["max_scan_attempts", "network_timeout", "max_concurrent_requests"]:
                    if param in zarr_store:
                        kwargs[param] = zarr_store[param]

                # Support loading existing Zarr/Icechunk stores
                if zarr_store.get("existing", False):
                    kwargs["existing_zarr"] = True

                if "zarr_kwargs" in zarr_store:
                    kwargs["zarr_kwargs"] = zarr_store["zarr_kwargs"]

            # Support explicit bypass / existing flags at the data source level
            if details.get("existing", False) or details.get("bypass_load", False):
                kwargs["existing_zarr"] = True

            target_cluster = details.get("cluster", default_cluster)
            logger.debug(f"Adding data node: {node_id} (type={dataset_type}, cluster={target_cluster})")
            self.graph.add_node(
                node_id,
                task_type="load_data",
                name=name,
                dataset_type=dataset_type,
                cluster=target_cluster,
                kwargs=kwargs,
            )

    def _find_node(self, name: str) -> str | None:
        """
        Attempts to resolve a logical name into a node ID in the graph.

        Searches for prefixes: plot_, stats_, combine_, pair_, load_.
        Prioritizes more processed nodes (plots/stats) over raw sources.

        Parameters
        ----------
        name : str
            The logical name of the task/dataset to resolve.

        Returns
        -------
        str or None
            The node ID if found, else None.
        """
        # Search in reverse order to prioritize downstream/processed nodes
        prefixes = ["plot", "stats", "combine", "pair", "reduction", "save", "load"]
        for p in prefixes:
            node_id = f"{p}_{name}"
            if node_id in self.graph:
                logger.debug(f"Resolved name '{name}' to node ID '{node_id}' using prefix '{p}_'")
                return node_id
        logger.warning(f"Failed to resolve name '{name}' to any node ID in the graph.")
        return None

    def _add_pairing_nodes(self) -> None:
        """Adds nodes for pairing datasets together (from monet)."""
        pairing_cfg = self.config.pairing
        if not pairing_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in pairing_cfg.items():
            node_id = f"pair_{name}"
            source = details.get("source")
            target = details.get("target")

            if not source or not target:
                logger.warning(f"Pairing '{name}' is missing 'source' or 'target'. Skipping.")
                continue

            source_node = self._find_node(source)
            target_node = self._find_node(target)

            if not source_node:
                logger.error(f"Source '{source}' for pairing '{name}' not found. Skipping pairing.")
                continue
            if not target_node:
                logger.error(f"Target '{target}' for pairing '{name}' not found. Skipping pairing.")
                continue

            target_cluster = details.get("cluster", default_cluster)
            logger.debug(f"Adding pairing node: {node_id} (source={source_node}, target={target_node}, cluster={target_cluster})")
            mask = details.get("mask")
            node_attrs = {
                "task_type": "pair_data",
                "name": name,
                "method": details.get("method", "interpolate"),
                "cluster": target_cluster,
                "kwargs": details.get("kwargs", {}),
                "source_name": source,
                "target_name": target,
            }
            if mask:
                node_attrs["mask"] = mask
            self.graph.add_node(node_id, **node_attrs)

            logger.debug(f"Adding edge: {source_node} -> {node_id}")
            self.graph.add_edge(source_node, node_id)
            logger.debug(f"Adding edge: {target_node} -> {node_id}")
            self.graph.add_edge(target_node, node_id)

    def _add_combine_nodes(self) -> None:
        """Adds nodes for combining multiple paired datasets."""
        combine_cfg = self.config.combine
        if not combine_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in combine_cfg.items():
            node_id = f"combine_{name}"
            sources = details.get("sources", [])
            dim = details.get("dim", "model")

            if not sources:
                logger.warning(f"Combine task '{name}' has no sources. Skipping.")
                continue

            # Check if all sources exist
            valid_sources = []
            for source in sources:
                source_node_id = self._find_node(source)
                if source_node_id:
                    valid_sources.append(source_node_id)
                else:
                    logger.warning(f"Source '{source}' for combine task '{name}' not found. Skipping source.")

            if not valid_sources:
                logger.error(f"Combine task '{name}' has no valid sources. Skipping.")
                continue

            target_cluster = details.get("cluster", default_cluster)
            logger.debug(f"Adding combine node: {node_id} (sources={valid_sources}, cluster={target_cluster})")
            self.graph.add_node(
                node_id,
                task_type="combine_paired_data",
                name=name,
                sources=sources,  # List of original names, Engine uses this to look them up
                dim=dim,
                cluster=target_cluster,
                kwargs=details.get("kwargs", {}),
            )

            for pair_node_id in valid_sources:
                logger.debug(f"Adding edge: {pair_node_id} -> {node_id}")
                self.graph.add_edge(pair_node_id, node_id)

    def _add_statistics_nodes(self) -> None:
        """Adds nodes for computing statistics on paired data (from monet-stats)."""
        stats_cfg = self.config.statistics
        if not stats_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in stats_cfg.items():
            node_id = f"stats_{name}"
            input_data = details.get("input")

            if not input_data:
                logger.warning(f"Statistics task '{name}' is missing 'input'. Skipping.")
                continue

            target_parent = self._find_node(input_data)

            if not target_parent:
                logger.error(f"Input '{input_data}' for statistics '{name}' not found. Skipping stats.")
                continue

            target_cluster = details.get("cluster", default_cluster)
            kwargs = details.get("kwargs", {})
            regions = kwargs.get("regions") if kwargs else None

            logger.debug(f"Adding statistics node: {node_id} (input={target_parent}, cluster={target_cluster})")
            node_attrs: dict[str, Any] = {
                "task_type": "compute_statistics",
                "name": name,
                "metrics": details.get("metrics", []),
                "cluster": target_cluster,
                "kwargs": kwargs,
            }
            if regions:
                node_attrs["regions"] = list(regions)
            self.graph.add_node(node_id, **node_attrs)

            logger.debug(f"Adding edge: {target_parent} -> {node_id}")
            self.graph.add_edge(target_parent, node_id)

    def _add_plotting_nodes(self) -> None:
        """Adds nodes for plotting data or statistics (from monet-plots)."""
        plots_cfg = self.config.plots
        if not plots_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in plots_cfg.items():
            node_id = f"plot_{name}"
            input_data = details.get("input")

            if not input_data:
                logger.warning(f"Plot task '{name}' is missing 'input'. Skipping.")
                continue

            target_parent = self._find_node(input_data)

            if not target_parent:
                logger.error(f"Input '{input_data}' for plot '{name}' not found. Skipping plot.")
                continue

            target_cluster = details.get("cluster", default_cluster)
            kwargs = details.get("kwargs", {})
            regions = kwargs.get("regions") if kwargs else None

            logger.debug(f"Adding plotting node: {node_id} (input={target_parent}, cluster={target_cluster})")
            node_attrs: dict[str, Any] = {
                "task_type": "generate_plot",
                "name": name,
                "plot_type": details.get("type", "spatial"),
                "cluster": target_cluster,
                "kwargs": kwargs,
            }
            if regions:
                node_attrs["regions"] = list(regions)
            self.graph.add_node(node_id, **node_attrs)

            logger.debug(f"Adding edge: {target_parent} -> {node_id}")
            self.graph.add_edge(target_parent, node_id)

    def _add_reduction_nodes(self) -> None:
        """Adds nodes for calculating data reductions (e.g. mean along dimensions)."""
        reduction_cfg = self.config.reductions
        if not reduction_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in reduction_cfg.items():
            node_id = f"reduction_{name}"
            input_data = details.get("input")

            if not input_data:
                logger.warning(f"Reduction task '{name}' is missing 'input'. Skipping.")
                continue

            target_parent = self._find_node(input_data)

            if not target_parent:
                logger.error(f"Input '{input_data}' for reduction '{name}' not found. Skipping reduction.")
                continue

            target_cluster = details.get("cluster", default_cluster)
            method = details.get("method", "mean")
            dim = details.get("dim")
            force_weighted = details.get("force_weighted", False)
            kwargs = details.get("kwargs", {})

            logger.debug(f"Adding reduction node: {node_id} (input={target_parent}, cluster={target_cluster})")
            self.graph.add_node(
                node_id,
                task_type="calculate_reduction",
                name=name,
                method=method,
                dim=dim,
                force_weighted=force_weighted,
                cluster=target_cluster,
                kwargs=kwargs,
            )

            logger.debug(f"Adding edge: {target_parent} -> {node_id}")
            self.graph.add_edge(target_parent, node_id)

    def _add_save_nodes(self) -> None:
        """Adds nodes for saving/writing datasets to Zarr or Icechunk."""
        save_cfg = self.config.save
        if not save_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in save_cfg.items():
            node_id = f"save_{name}"
            input_data = details.get("input")

            if not input_data:
                logger.warning(f"Save task '{name}' is missing 'input'. Skipping.")
                continue

            target_parent = self._find_node(input_data)

            if not target_parent:
                logger.error(f"Input '{input_data}' for save '{name}' not found. Skipping save.")
                continue

            target_cluster = details.get("cluster", default_cluster)
            backend = details.get("backend")
            url = details.get("url")
            kwargs = details.get("kwargs", {})

            logger.debug(f"Adding save node: {node_id} (input={target_parent}, cluster={target_cluster})")
            self.graph.add_node(
                node_id,
                task_type="save_data",
                name=name,
                backend=backend,
                url=url,
                cluster=target_cluster,
                kwargs=kwargs,
            )

            logger.debug(f"Adding edge: {target_parent} -> {node_id}")
            self.graph.add_edge(target_parent, node_id)
