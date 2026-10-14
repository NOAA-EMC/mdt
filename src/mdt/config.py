"""Module for parsing and validating MDT YAML configuration files."""

import copy
import logging
import pathlib
from typing import Any, Dict, Union, cast

import yaml

logger = logging.getLogger(__name__)

VALID_ZARR_BACKENDS = {"kerchunk_json", "kerchunk_parquet", "icechunk", "zarr"}


class ConfigParser:
    """
    Parses and validates MDT YAML configuration files.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path to the YAML configuration file.
    """

    def __init__(self, config_path: Union[str, pathlib.Path]):
        self.config_path = pathlib.Path(config_path)
        self.config: Dict[str, Any] = self._load_yaml()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open() as f:
            try:
                config = yaml.safe_load(f)
                if config is None:
                    return {}
                if not isinstance(config, dict):
                    raise ValueError(f"Configuration validation failed: top-level YAML must be a mapping, got {type(config).__name__}.")
                return cast(Dict[str, Any], config)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {e}") from e

    def _validate_config(self) -> None:
        """Basic validation to ensure required top-level keys exist."""
        required_keys = ["data"]
        for key in required_keys:
            if key not in self.config:
                msg = f"Configuration validation failed: missing top-level key '{key}'."
                logger.error(msg)
                raise ValueError(msg)

        section_errors = []
        for section_name in ("data", "pairing", "combine", "statistics", "plots", "reductions", "save", "execution"):
            if section_name in self.config and not isinstance(self.config[section_name], dict):
                section_errors.append(f"'{section_name}' section must be a dictionary, got {type(self.config[section_name]).__name__}.")
        if section_errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in section_errors))

        execution_cfg = self.config.get("execution") or {}
        pairing_mode = execution_cfg.get("pairing_mode", "serial")
        if not isinstance(pairing_mode, str) or pairing_mode not in {"serial", "concurrent"}:
            raise ValueError(f"Configuration validation failed: execution.pairing_mode must be 'serial' or 'concurrent', got {pairing_mode!r}.")
        clusters = execution_cfg.get("clusters", {})
        if clusters is None:
            clusters = {}
        if not isinstance(clusters, dict):
            raise ValueError("Configuration validation failed: execution.clusters must be a mapping.")
        if "clusters" in execution_cfg and not clusters:
            raise ValueError("Configuration validation failed: execution.clusters must contain at least one cluster.")
        valid_modes = {"local", "hera", "jet", "orion", "hercules", "gaea", "ursa", "wcoss2", "slurm", "pbs", "lsf"}
        for cluster_name, cluster_cfg in clusters.items():
            if not isinstance(cluster_cfg, dict):
                raise ValueError(f"Configuration validation failed: execution.clusters.{cluster_name} must be a mapping.")
            mode = cluster_cfg.get("mode", "local")
            if not isinstance(mode, str) or mode.lower() not in valid_modes:
                raise ValueError(
                    f"Configuration validation failed: execution.clusters.{cluster_name}.mode must be one of {sorted(valid_modes)}, got {mode!r}."
                )
            workers = cluster_cfg.get("workers", 1)
            if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
                raise ValueError(
                    f"Configuration validation failed: execution.clusters.{cluster_name}.workers must be a positive integer, got {workers!r}."
                )
            if "cluster_kwargs" in cluster_cfg and not isinstance(cluster_cfg["cluster_kwargs"], dict):
                raise ValueError(f"Configuration validation failed: execution.clusters.{cluster_name}.cluster_kwargs must be a mapping.")
        default_cluster = execution_cfg.get("default_cluster")
        if default_cluster is not None and default_cluster not in clusters:
            raise ValueError(
                f"Configuration validation failed: execution.default_cluster '{default_cluster}' is not defined in execution.clusters."
            )

        # Ensure 'data' is a dictionary if it exists
        if "data" in self.config:
            if not isinstance(self.config["data"], dict):
                raise ValueError(f"Configuration validation failed: 'data' section must be a dictionary, got {type(self.config['data']).__name__}.")
            for name, details in self.config["data"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Configuration validation failed: Data source '{name}' must be a mapping, got {type(details).__name__}.")
                if "type" not in details:
                    raise ValueError(f"Configuration validation failed: Data source '{name}' is missing the required 'type' field.")
                if "kwargs" in details and details["kwargs"] is not None and not isinstance(details["kwargs"], dict):
                    raise ValueError(f"Configuration validation failed: data.{name}.kwargs must be a mapping.")
                time_axis = (details.get("kwargs") or {}).get("time_axis")
                if time_axis is not None:
                    if not isinstance(time_axis, dict):
                        raise ValueError(f"Configuration validation failed: data.{name}.kwargs.time_axis must be a mapping.")
                    for role, candidates in time_axis.items():
                        if role not in {"valid_time", "forecast_reference_time", "lead_time"}:
                            raise ValueError(f"Configuration validation failed: data.{name}.kwargs.time_axis has unknown role '{role}'.")
                        values = [candidates] if isinstance(candidates, str) else candidates
                        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value.strip() for value in values):
                            raise ValueError(f"Configuration validation failed: data.{name}.kwargs.time_axis.{role} must be a name or non-empty list of names.")

                zarr_store = details.get("zarr_store")
                if zarr_store is not None:
                    if not isinstance(zarr_store, dict):
                        raise ValueError(
                            f"Configuration validation failed: Data source '{name}': "
                            f"'zarr_store' must be a mapping, got {type(zarr_store).__name__}."
                        )
                    enabled = zarr_store.get("enabled", False)
                    if enabled:
                        backend = zarr_store.get("backend", "kerchunk_json")
                        if backend not in VALID_ZARR_BACKENDS:
                            raise ValueError(
                                f"Configuration validation failed: Data source '{name}': "
                                f"unsupported zarr_store.backend '{backend}'. "
                                f"Supported: {sorted(VALID_ZARR_BACKENDS)}"
                            )
                        if backend == "icechunk" and not (zarr_store.get("icechunk_url") or zarr_store.get("icechunk_repo")):
                            raise ValueError(
                                f"Configuration validation failed: Data source '{name}': "
                                "'icechunk_repo' is required for 'icechunk' backend (preferred key: 'icechunk_url')."
                            )

        if "pairing" in self.config:
            if not isinstance(self.config["pairing"], dict):
                raise ValueError(
                    f"Configuration validation failed: 'pairing' section must be a dictionary, got {type(self.config['pairing']).__name__}."
                )
            for name, details in self.config["pairing"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Configuration validation failed: Pairing '{name}' must be a dictionary, got {type(details).__name__}.")
                sources = details.get("sources", details.get("source"))
                if sources is None or "target" not in details:
                    raise ValueError(
                        f"Configuration validation failed: Pairing '{name}' must specify 'source' (or non-empty 'sources') and 'target'."
                    )
                if "sources" in details and (
                    not isinstance(sources, list) or not sources or not all(isinstance(s, str) and s.strip() for s in sources)
                ):
                    raise ValueError(f"Configuration validation failed: pairing.{name}.sources must be a non-empty list of dataset names.")
                if "kwargs" in details and details["kwargs"] is not None and not isinstance(details["kwargs"], dict):
                    raise ValueError(f"Configuration validation failed: pairing.{name}.kwargs must be a mapping.")

        if "combine" in self.config:
            for name, details in self.config["combine"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Configuration validation failed: combine.{name} must be a mapping.")
                sources = details.get("sources")
                if not isinstance(sources, list) or not sources or not all(isinstance(source, str) and source.strip() for source in sources):
                    raise ValueError(f"Configuration validation failed: combine.{name}.sources must be a non-empty list of task names.")
                if "kwargs" in details and details["kwargs"] is not None and not isinstance(details["kwargs"], dict):
                    raise ValueError(f"Configuration validation failed: combine.{name}.kwargs must be a mapping.")

        if "statistics" in self.config:
            if not isinstance(self.config["statistics"], dict):
                raise ValueError(
                    f"Configuration validation failed: 'statistics' section must be a dictionary, got {type(self.config['statistics']).__name__}."
                )
            for name, details in self.config["statistics"].items():
                if not isinstance(details, dict):
                    raise ValueError(
                        f"Configuration validation failed: Statistics task '{name}' must be a dictionary, got {type(details).__name__}."
                    )
                if "input" not in details:
                    raise ValueError(f"Configuration validation failed: Statistics task '{name}' must specify an 'input' key.")
                metrics = details.get("metrics")
                if not isinstance(metrics, list) or not metrics or not all(isinstance(metric, str) and metric.strip() for metric in metrics):
                    raise ValueError(f"Configuration validation failed: Statistics task '{name}' must specify a non-empty list of metric names.")

        execution_cfg = self.config.get("execution") or {}
        if "ecflow_data_dir" in execution_cfg:
            data_dir = execution_cfg["ecflow_data_dir"]
            if not isinstance(data_dir, str) or not data_dir.strip():
                raise ValueError("Configuration validation failed: execution.ecflow_data_dir must be a non-empty path string.")

        if "plots" in self.config:
            if not isinstance(self.config["plots"], dict):
                raise ValueError(
                    f"Configuration validation failed: 'plots' section must be a dictionary, got {type(self.config['plots']).__name__}."
                )
            for name, details in self.config["plots"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Configuration validation failed: Plot task '{name}' must be a dictionary, got {type(details).__name__}.")
                if "input" not in details:
                    raise ValueError(f"Configuration validation failed: Plot task '{name}' must specify an 'input' key.")

        if "reductions" in self.config:
            if not isinstance(self.config["reductions"], dict):
                raise ValueError(
                    f"Configuration validation failed: 'reductions' section must be a dictionary, got {type(self.config['reductions']).__name__}."
                )
            for name, details in self.config["reductions"].items():
                if not isinstance(details, dict):
                    raise ValueError(
                        f"Configuration validation failed: Reduction task '{name}' must be a dictionary, got {type(details).__name__}."
                    )
                if "input" not in details or "dim" not in details:
                    raise ValueError(f"Configuration validation failed: Reduction task '{name}' must specify 'input' and 'dim' fields.")

        if "save" in self.config:
            if not isinstance(self.config["save"], dict):
                raise ValueError(f"Configuration validation failed: 'save' section must be a dictionary, got {type(self.config['save']).__name__}.")
            for name, details in self.config["save"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Configuration validation failed: Save task '{name}' must be a dictionary, got {type(details).__name__}.")
                if "input" not in details or "backend" not in details or "url" not in details:
                    raise ValueError(f"Configuration validation failed: Save task '{name}' must specify 'input', 'backend', and 'url' fields.")
                backend = details["backend"]
                if backend not in {"icechunk", "zarr"}:
                    raise ValueError(
                        f"Configuration validation failed: Save task '{name}': unsupported backend '{backend}'. Supported: 'icechunk', 'zarr'."
                    )

        self._validate_region_masking()

    def _validate_region_masking(self) -> None:
        """Validate mask and regions configuration."""
        # Validate mask keys in pairing section
        for name, details in self.pairing.items():
            if "mask" in details:
                mask_val = details["mask"]
                if not isinstance(mask_val, str) or not mask_val.strip():
                    raise ValueError(
                        f"Configuration validation failed: Pairing '{name}' has invalid 'mask' value \u2014 must be a non-empty string."
                    )

        # Validate regions in plots and statistics
        for section_name, section in [("plots", self.plots), ("statistics", self.statistics)]:
            for name, details in section.items():
                kwargs = details.get("kwargs", {}) or {}
                if "regions" in kwargs:
                    regions = kwargs["regions"]
                    if not isinstance(regions, list) or len(regions) == 0 or not all(isinstance(r, str) and r.strip() for r in regions):
                        raise ValueError(
                            f"Configuration validation failed: {section_name.title()} task "
                            f"'{name}' has invalid 'regions' \u2014 must be a list of non-empty strings."
                        )
                    # Cross-validate: referenced pairing must have mask
                    input_name = details.get("input", "")
                    pairing_details = self.pairing.get(input_name, {})
                    if "mask" not in pairing_details:
                        raise ValueError(
                            f"Configuration validation failed: {section_name.title()} task "
                            f"'{name}' specifies 'regions' but its input pairing "
                            f"'{input_name}' does not define a 'mask' key."
                        )

    @property
    def data(self) -> Dict[str, Any]:
        """dict: The 'data' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("data", {}))

    @property
    def pairing(self) -> Dict[str, Any]:
        """dict: The 'pairing' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("pairing", {}))

    @property
    def combine(self) -> Dict[str, Any]:
        """dict: The 'combine' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("combine", {}))

    @property
    def statistics(self) -> Dict[str, Any]:
        """dict: The 'statistics' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("statistics", {}))

    @property
    def plots(self) -> Dict[str, Any]:
        """dict: The 'plots' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("plots", {}))

    @property
    def reductions(self) -> Dict[str, Any]:
        """dict: The 'reductions' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("reductions", {}))

    @property
    def save(self) -> Dict[str, Any]:
        """dict: The 'save' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("save", {}))

    @property
    def orchestrator(self) -> str:
        """str: The orchestrator backend name from the execution section, defaults to 'prefect'."""
        return str(self.execution.get("orchestrator", "prefect"))

    @property
    def execution(self) -> Dict[str, Any]:
        """
        Retrieves the execution configuration, which can define multiple clusters.

        If not specified, defaults to a single local cluster named 'compute'.

        When the orchestrator is ``"ecflow"``, the following ecFlow-specific
        keys are accessible directly from the returned dict:

        - ``ecflow_host`` — ecFlow server hostname (default ``"localhost"``)
        - ``ecflow_port`` — ecFlow server port (default ``3141``)
        - ``suite_name``  — ecFlow suite name (default ``"mdt"``)
        - ``task_script_dir`` — directory for generated ``.ecf`` wrapper
          scripts (default ``"./ecflow_tasks/"``)

        These keys are read by
        :class:`~mdt.ecflow_engine.EcFlowEngine` at construction time.

        Returns
        -------
        dict
            The execution configuration section, including any
            orchestrator-specific keys defined in the YAML.
        """
        exec_cfg: Dict[str, Any] = copy.deepcopy(self.config.get("execution", {}))
        if not exec_cfg:
            logger.debug("No 'execution' section found; using default local 'compute' cluster.")
            return {"default_cluster": "compute", "clusters": {"compute": {"mode": "local"}}}

        # Backwards compatibility: if they just provided 'mode', wrap it in a 'compute' cluster.
        # Preserve any non-cluster keys (e.g. orchestrator, ecflow_host, etc.)
        # at the top level so engine implementations can still access them.
        if "mode" in exec_cfg and "clusters" not in exec_cfg:
            logger.debug("Legacy 'execution' format detected (missing 'clusters' key). Mapping to 'compute' cluster.")
            cluster_cfg = {k: v for k, v in exec_cfg.items() if k == "mode" or k == "workers"}
            result = {k: v for k, v in exec_cfg.items() if k not in ("mode", "workers")}
            result["default_cluster"] = "compute"
            result["clusters"] = {"compute": cluster_cfg if cluster_cfg else exec_cfg}
            exec_cfg = result

        # Ensure default_cluster is set if clusters are defined
        if "clusters" in exec_cfg and "default_cluster" not in exec_cfg:
            exec_cfg["default_cluster"] = next(iter(exec_cfg["clusters"].keys()))
            logger.info(f"Using '{exec_cfg['default_cluster']}' as the default cluster.")

        # Ensure a service cluster is always defined for service-node tasks (like data download)
        if "service" not in exec_cfg.get("clusters", {}):
            # Attempt to derive the service mode from the default compute cluster's mode
            compute_mode = "local"
            if "clusters" in exec_cfg and exec_cfg["default_cluster"] in exec_cfg["clusters"]:
                compute_mode = exec_cfg["clusters"][exec_cfg["default_cluster"]].get("mode", "local")

            # If the primary execution is an HPC scheduler, we should map the service cluster to it
            if compute_mode != "local":
                service_mode = compute_mode
            else:
                service_mode = "local"

            logger.info(f"Automatically creating 'service' cluster with mode='{service_mode}' for data orchestration.")
            exec_cfg.setdefault("clusters", {})["service"] = {"mode": service_mode, "workers": 1}

        return exec_cfg

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a specific key from the raw configuration.

        Parameters
        ----------
        key : str
            The configuration key to retrieve.
        default : any, optional
            The default value to return if the key is not found, by default None.

        Returns
        -------
        any
            The value associated with the key, or the default value.
        """
        return self.config.get(key, default)


def load_config(config_path: Union[str, pathlib.Path]) -> ConfigParser:
    """
    Helper function to quickly initialize a ConfigParser from a path.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path to the YAML configuration file.

    Returns
    -------
    ConfigParser
        An instantiated and validated parser object.
    """
    return ConfigParser(config_path)
