"""ecFlow orchestrator engine for MDT.

Translates a NetworkX DAG into an ecFlow suite definition, generates
per-task wrapper scripts, and submits the suite to an ecFlow server.

The ``ecflow`` package is imported lazily so that this module can be
loaded without ecFlow installed.
"""

import base64
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import networkx as nx

from mdt.engine_registry import Engine

if TYPE_CHECKING:
    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)

#: Maps DAG ``task_type`` values to ecFlow family names.
_FAMILY_MAP: dict[str, str] = {
    "load_data": "load",
    "pair_data": "pair",
    "combine_paired_data": "combine",
    "compute_statistics": "statistics",
    "generate_plot": "plot",
    "calculate_reduction": "reduction",
    "save_data": "save",
}


#: Template for the dispatch block inside each wrapper script.
#: Maps ``TASK_TYPE`` values to the import + call statements.
_DISPATCH_BLOCKS: dict[str, str] = {
    "load_data": (
        "    if zarr_enabled == 'true':\n"
        "        if zarr_backend == 'icechunk':\n"
        "            kwargs['use_icechunk'] = True\n"
        "        else:\n"
        "            kwargs['use_virtualizarr'] = True\n"
        "            kwargs['virtualizarr_backend'] = zarr_backend\n"
        "        if zarr_path and zarr_backend != 'icechunk':\n"
        "            kwargs['store_path'] = zarr_path\n"
        "        if icechunk_url:\n"
        "            kwargs['icechunk_url'] = icechunk_url\n"
        "    from mdt.tasks.data import load_data\n"
        "    result = load_data(name=task_name, dataset_type=dataset_type, kwargs=kwargs)"
    ),
    "pair_data": (
        "    from mdt.tasks.pairing import pair_data\n"
        "    source_inputs = input_node_ids['source']\n"
        "    if isinstance(source_inputs, dict):\n"
        "        source_inputs = {name: load_artifact(data_dir, node_id) for name, node_id in source_inputs.items()}\n"
        "        if len(source_inputs) == 1:\n"
        "            source_inputs = next(iter(source_inputs.values()))\n"
        "    else:\n"
        "        source_inputs = load_artifact(data_dir, source_inputs)\n"
        "    kwargs = {**kwargs, 'model_dim': model_dim}\n"
        "    result = pair_data(name=task_name, method=method,\n"
        "                       source_data=source_inputs, target_data=inputs['target'], kwargs=kwargs, mask=mask)"
    ),
    "combine_paired_data": (
        "    from mdt.tasks.pairing import combine_paired_data\n"
        "    paired_data_inputs = {}\n"
        "    for source, node_id in input_node_ids['pairs'].items():\n"
        "        paired_data_inputs[source] = load_artifact(data_dir, node_id)\n"
        "    result = combine_paired_data(paired_data=paired_data_inputs, dim=dim)"
    ),
    "compute_statistics": (
        "    from mdt.tasks.statistics import compute_statistics\n"
        "    result = compute_statistics(name=task_name, metrics=metrics,\n"
        "                                input_data=inputs['input'], kwargs=kwargs)"
    ),
    "generate_plot": (
        "    from mdt.tasks.plotting import generate_plot\n"
        "    generate_plot(name=task_name, plot_type=plot_type,\n"
        "                  input_data=inputs['input'], kwargs=kwargs)\n"
        "    result = None  # plots are side effects; they are not data artifacts"
    ),
    "calculate_reduction": (
        "    from mdt.tasks.reductions import calculate_reduction\n"
        "    result = calculate_reduction(obj=inputs['input'], method=method, dim=dim,\n"
        "                                  force_weighted=force_weighted, **kwargs)"
    ),
    "save_data": (
        "    from mdt.tasks.data import save_data\n"
        "    result = inputs['input']\n"
        "    save_data(name=task_name, data=result, backend=backend, url=url, kwargs=kwargs)"
    ),
}


def _build_wrapper_script(node_id: str, python_executable: str) -> str:  # noqa: ARG001
    """Return the full text of an ``.ecf`` wrapper script for *node_id*.

    The script uses ecFlow ``%VAR%`` substitution tokens that the ecFlow
    server replaces at runtime with the variable values set on the task
    node.
    """
    # node_id is used by the caller to name the script file; not needed in the script body.
    lines = [
        f"#!{python_executable}",
        '"""Auto-generated ecFlow wrapper for task %TASK_NAME%."""',
        "import base64, json, os, sys",
        "import ecflow",
        "",
        'client = ecflow.Client(os.environ.get("ECF_HOST", "localhost"),',
        '                       int(os.environ.get("ECF_PORT", "3141")))',
        'child_path = "%ECF_NAME%"',
        'child_pass = "%ECF_PASS%"',
        'child_try_no = int("%ECF_TRYNO%")',
        "client.set_child_path(child_path)",
        "client.set_child_password(child_pass)",
        "client.set_child_try_no(child_try_no)",
        "client.set_child_pid(os.getpid())",
        "try:",
        "    client.child_init()",
        "",
        "    # --- task parameters (substituted by ecFlow) ---",
        "    decode = lambda value: base64.b64decode(value).decode('utf-8')",
        "    task_type = decode('%TASK_TYPE_B64%')",
        "    task_name = decode('%TASK_NAME_B64%')",
        "    dataset_type = decode('%DATASET_TYPE_B64%')",
        "    kwargs = json.loads(decode('%TASK_KWARGS_B64%'))",
        "    metrics = json.loads(decode('%METRICS_B64%'))",
        "    plot_type = decode('%PLOT_TYPE_B64%')",
        "    method = decode('%METHOD_B64%')",
        "    dim = json.loads(decode('%DIM_B64%'))",
        "    model_dim = decode('%MODEL_DIM_B64%')",
        "    mask = json.loads(decode('%MASK_B64%'))",
        "    force_weighted = decode('%FORCE_WEIGHTED_B64%') == 'true'",
        "    backend = decode('%BACKEND_B64%')",
        "    url = decode('%URL_B64%')",
        "    zarr_enabled = decode('%ZARR_STORE_ENABLED_B64%')",
        "    zarr_backend = decode('%ZARR_STORE_BACKEND_B64%')",
        "    zarr_path = decode('%ZARR_STORE_PATH_B64%')",
        "    icechunk_url = decode('%ZARR_STORE_ICECHUNK_URL_B64%')",
        "    sources = json.loads(decode('%SOURCES_B64%'))",
        "    node_id = decode('%NODE_ID_B64%')",
        "    data_dir = decode('%DATA_DIR_B64%')",
        "    input_node_ids = json.loads(decode('%INPUT_NODE_IDS_B64%'))",
        "    from mdt.ecflow_io import load_artifact, save_artifact",
        "    inputs = {key: load_artifact(data_dir, upstream_id) for key, upstream_id in input_node_ids.items() if key not in ('pairs', 'source')}",
        "",
        "    # --- dispatch to the correct MDT task function ---",
    ]

    # Build an if/elif chain for all known task types.
    first = True
    for task_type, block in _DISPATCH_BLOCKS.items():
        keyword = "if" if first else "elif"
        lines.append(f'    {keyword} task_type == "{task_type}":')
        for block_line in block.splitlines():
            lines.append(f"    {block_line}")
        first = False
    lines.append("    else:")
    lines.append('        raise ValueError(f"Unknown task_type: {task_type}")')
    lines.append("    save_artifact(data_dir, node_id, result)")

    lines += [
        "",
        "    client.child_complete()",
        "except Exception as e:",
        "    client.child_abort(str(e))",
        "    sys.exit(1)",
        "",
    ]
    return "\n".join(lines) + "\n"


class EcFlowEngine(Engine):
    """Engine implementation that executes DAGs via ECMWF ecFlow.

    Parameters
    ----------
    dag : nx.DiGraph
        The task dependency graph built by :class:`~mdt.dag.DAGBuilder`.
    config : object
        The parsed MDT configuration object (typically
        :class:`~mdt.config.ConfigParser`).

    Raises
    ------
    ImportError
        If the ``ecflow`` package is not installed.
    """

    def __init__(self, dag: nx.DiGraph, config: "ConfigParser"):
        try:
            import ecflow  # lazy import — keep ecflow optional
        except ImportError as e:
            raise ImportError(
                "ecFlow Python bindings are not installed. MDT does not provide an ecFlow extra; install the ecFlow client package "
                "through your site's operational environment and ensure its Python bindings are available."
            ) from e

        self.dag = dag
        self.config = config
        self.ecflow = ecflow

        exec_cfg = config.execution
        self.host: str = exec_cfg.get("ecflow_host", "localhost")
        self.port: int = int(exec_cfg.get("ecflow_port", 3141))
        self.suite_name: str = exec_cfg.get("suite_name", "mdt")
        self.task_script_dir: str = exec_cfg.get("task_script_dir", "./ecflow_tasks/")
        data_root = Path(exec_cfg.get("ecflow_data_dir", Path(self.task_script_dir) / self.suite_name / "data")).expanduser().resolve()
        self.data_dir = data_root / uuid.uuid4().hex
        self.python_executable: str = sys.executable
        self.poll_interval: float = float(exec_cfg.get("ecflow_poll_interval", 5))
        timeout = exec_cfg.get("ecflow_timeout", 86400)
        self.timeout: float | None = float(timeout) if timeout is not None else None
        if self.poll_interval <= 0:
            raise ValueError("execution.ecflow_poll_interval must be greater than zero")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("execution.ecflow_timeout must be greater than zero or null")
        self.client: Any = None

    # ------------------------------------------------------------------
    # Engine ABC
    # ------------------------------------------------------------------

    def execute(self) -> Dict[str, Any]:
        """Build the suite, start it, and wait for its terminal outcome.

        Returns
        -------
        dict
            ``{"suite": <suite_name>, "status": "complete"}``
        """
        try:
            if self.dag.number_of_nodes() == 0:
                logger.info("Workflow '%s' has no tasks; skipping ecFlow submission.", self.suite_name)
                return {"suite": self.suite_name, "status": "complete"}

            logger.info(f"Starting ecFlow workflow orchestration for suite '{self.suite_name}'")
            defs = self.build_suite()
            self.generate_task_wrappers()

            # Save the suite definition to a file for user inspection
            Path(self.task_script_dir).mkdir(parents=True, exist_ok=True)
            def_path = Path(self.task_script_dir) / f"{self.suite_name}.def"
            logger.info(f"Saving ecFlow suite definition to {def_path}")
            defs.save_as_defs(str(def_path))

            self._load_and_start(defs)
            self._wait_for_suite()
            return {"suite": self.suite_name, "status": "complete"}
        finally:
            self.close()

    def _wait_for_suite(self) -> None:
        """Poll the server until the suite completes or fails."""
        if self.client is None:
            raise RuntimeError("Cannot wait for ecFlow suite outcome without a connected client")

        started_at = time.monotonic()
        suite_path = f"/{self.suite_name}"
        while True:
            self.client.sync_local()
            defs = self.client.get_defs()
            suite = defs.find_suite(self.suite_name)
            if suite is None:
                raise RuntimeError(f"ecFlow suite '{self.suite_name}' disappeared before completion")

            state = str(suite.get_state()).lower()
            if "complete" in state:
                logger.info("ecFlow suite '%s' completed successfully.", self.suite_name)
                return
            if "abort" in state or "failed" in state:
                failed_nodes = []
                get_nodes = getattr(suite, "get_all_nodes", None)
                if callable(get_nodes):
                    for node in get_nodes():
                        node_state = str(node.get_state()).lower()
                        if "abort" in node_state or "failed" in node_state:
                            get_path = getattr(node, "get_abs_node_path", None)
                            failed_nodes.append(get_path() if callable(get_path) else str(getattr(node, "name", lambda: "unknown")()))
                failed_context = ", ".join(failed_nodes) or suite_path
                raise RuntimeError(f"ecFlow workflow failed for task(s): {failed_context} (suite state: {state})")

            if self.timeout is not None and time.monotonic() - started_at >= float(self.timeout):
                raise TimeoutError(f"Timed out waiting for ecFlow suite '{self.suite_name}' after {self.timeout} seconds")
            time.sleep(self.poll_interval)

    def close(self) -> None:
        """Close the ecFlow client connection when its API supports it."""
        if self.client is None:
            return
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("Failed to close ecFlow client cleanly: %s", exc)
        self.client = None

    # ------------------------------------------------------------------
    # Stubs — implemented in later tasks
    # ------------------------------------------------------------------

    def build_suite(self) -> Any:
        """Translate the DAG into an ``ecflow.Defs`` suite definition.

        Creates one ecFlow family per task type and one ecFlow task node
        per DAG node.  Trigger expressions are derived from DAG edges and
        ecFlow variables carry the parameters each wrapper script needs.

        Returns
        -------
        ecflow.Defs
            The fully constructed suite definition ready to be loaded into
            an ecFlow server.
        """
        ecflow = self.ecflow

        defs = ecflow.Defs()
        suite = ecflow.Suite(self.suite_name)

        # Let ecFlow resolve scripts from task_script_dir/<suite>/<family>/<task>.ecf
        # regardless of server ECF_HOME location.
        suite.add_variable("ECF_FILES", str(Path(self.task_script_dir).resolve()))

        # Create one family per task type.
        families: dict[str, Any] = {}
        for family_name in _FAMILY_MAP.values():
            logger.debug(f"Creating ecFlow family: {family_name}")
            family = suite.add_family(family_name)
            families[family_name] = family

        # Build lookups for trigger generation.
        node_family: dict[str, str] = {}
        node_tasks: dict[str, Any] = {}

        logger.info(f"Adding {self.dag.number_of_nodes()} tasks to ecFlow suite definition")
        for node_id, data in self.dag.nodes(data=True):
            task_type = data["task_type"]
            family_name = _FAMILY_MAP[task_type]
            node_family[node_id] = family_name

            task_node = families[family_name].add_task(node_id)
            node_tasks[node_id] = task_node

            # --- ecFlow variables ---
            task_node.add_variable("TASK_TYPE", data["task_type"])
            task_node.add_variable("TASK_NAME", data["name"])
            task_node.add_variable("DATASET_TYPE", data.get("dataset_type") or "")
            task_node.add_variable("TASK_KWARGS", json.dumps(data.get("kwargs") or {}))
            task_node.add_variable("METRICS", json.dumps(data.get("metrics") or []))
            task_node.add_variable("PLOT_TYPE", data.get("plot_type") or "")
            task_node.add_variable("METHOD", data.get("method") or "")
            task_node.add_variable("DIM", data.get("dim") or "")
            task_node.add_variable("FORCE_WEIGHTED", str(data.get("force_weighted") or "").lower())
            task_node.add_variable("SOURCES", json.dumps(data.get("sources") or []))
            safe_variables = {
                "TASK_TYPE_B64": data["task_type"],
                "TASK_NAME_B64": data.get("name") or "",
                "DATASET_TYPE_B64": data.get("dataset_type") or "",
                "TASK_KWARGS_B64": json.dumps(data.get("kwargs") or {}),
                "METRICS_B64": json.dumps(data.get("metrics") or []),
                "PLOT_TYPE_B64": data.get("plot_type") or "",
                "METHOD_B64": data.get("method") or "",
                "DIM_B64": json.dumps(data.get("dim") or ""),
                "MODEL_DIM_B64": data.get("model_dim", "model"),
                "MASK_B64": json.dumps(data.get("mask")),
                "SOURCES_B64": json.dumps(data.get("sources") or []),
                "FORCE_WEIGHTED_B64": str(data.get("force_weighted") or "").lower(),
                "BACKEND_B64": data.get("backend") or "",
                "URL_B64": data.get("url") or "",
                "NODE_ID_B64": node_id,
                "DATA_DIR_B64": str(self.data_dir),
                "INPUT_NODE_IDS_B64": json.dumps(self._input_node_ids(node_id, data)),
            }
            for variable, value in safe_variables.items():
                encoded = base64.b64encode(str(value).encode("utf-8")).decode("ascii")
                task_node.add_variable(variable, encoded)
            task_node.add_variable("CLUSTER", data.get("cluster") or "")
            task_node.add_variable("BACKEND", data.get("backend") or "")
            task_node.add_variable("URL", data.get("url") or "")

            # Always define ZARR_* so ecFlow token substitution succeeds for all
            # generated wrappers, even for non-load tasks.
            zarr_variables = {
                "ZARR_STORE_ENABLED": "false",
                "ZARR_STORE_BACKEND": "",
                "ZARR_STORE_PATH": "",
                "ZARR_STORE_ICECHUNK_URL": "",
                "ZARR_STORE_ICECHUNK_REPO": "",
            }

            if task_type == "load_data":
                node_kwargs = data.get("kwargs") or {}
                zarr_variables["ZARR_STORE_ENABLED"] = str(node_kwargs.get("use_virtualizarr", False)).lower()
                zarr_variables["ZARR_STORE_BACKEND"] = (
                    "icechunk" if node_kwargs.get("use_icechunk", False) else node_kwargs.get("virtualizarr_backend", "")
                )
                zarr_variables["ZARR_STORE_PATH"] = node_kwargs.get("store_path", "")
                icechunk_url = node_kwargs.get("icechunk_url") or node_kwargs.get("icechunk_repo", "")
                zarr_variables["ZARR_STORE_ICECHUNK_URL"] = icechunk_url
                zarr_variables["ZARR_STORE_ICECHUNK_REPO"] = icechunk_url

            for variable, value in zarr_variables.items():
                task_node.add_variable(variable, value)
            safe_zarr_variables = {
                "ZARR_STORE_ENABLED_B64": zarr_variables["ZARR_STORE_ENABLED"],
                "ZARR_STORE_BACKEND_B64": zarr_variables["ZARR_STORE_BACKEND"],
                "ZARR_STORE_PATH_B64": zarr_variables["ZARR_STORE_PATH"],
                "ZARR_STORE_ICECHUNK_URL_B64": zarr_variables["ZARR_STORE_ICECHUNK_URL"],
            }
            for variable, value in safe_zarr_variables.items():
                encoded = base64.b64encode(str(value).encode("utf-8")).decode("ascii")
                task_node.add_variable(variable, encoded)

        # --- trigger expressions from DAG edges ---
        for node_id in self.dag.nodes:
            predecessors = list(self.dag.predecessors(node_id))
            if not predecessors:
                continue

            parts = [f"/{self.suite_name}/{node_family[pred]}/{pred} == complete" for pred in sorted(predecessors)]
            trigger_expr = " and ".join(parts)
            node_tasks[node_id].add_trigger(trigger_expr)

        defs.add_suite(suite)
        return defs

    def _input_node_ids(self, node_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Map a task's logical inputs to the DAG nodes that produce them."""
        predecessors = list(self.dag.predecessors(node_id))
        task_type = data["task_type"]
        if task_type == "pair_data":
            result = {}
            for role, logical_name in [("target", data.get("target_name"))]:
                matches = [parent for parent in predecessors if self.dag.nodes[parent].get("name") == logical_name]
                if len(matches) != 1:
                    raise ValueError(f"ecFlow task {node_id!r} cannot resolve its {role} input {logical_name!r}")
                result[role] = matches[0]
            sources = {
                name: next(parent for parent in predecessors if self.dag.nodes[parent].get("name") == name)
                for name in data.get("source_names", [data.get("source_name")])
            }
            result["source"] = sources if len(sources) > 1 else next(iter(sources.values()))
            return result
        if task_type == "combine_paired_data":
            pairs = {}
            for source in data.get("sources", []):
                matches = [parent for parent in predecessors if self.dag.nodes[parent].get("name") == source]
                if len(matches) != 1:
                    raise ValueError(f"ecFlow task {node_id!r} cannot resolve paired input {source!r}")
                pairs[source] = matches[0]
            return {"pairs": pairs}
        if task_type in {"compute_statistics", "generate_plot", "calculate_reduction", "save_data"}:
            if len(predecessors) != 1:
                raise ValueError(f"ecFlow task {node_id!r} requires exactly one upstream input")
            return {"input": predecessors[0]}
        return {}

    def generate_task_wrappers(self) -> List[str]:
        """Generate one ``.ecf`` wrapper script per DAG node.

        Each script is a self-contained Python program that:

        1. Connects to the ecFlow server and calls ``client.init()``.
        2. Dispatches to the correct ``mdt.tasks.*`` function based on
           the ``TASK_TYPE`` ecFlow variable.
        3. Calls ``client.complete()`` on success.
        4. Calls ``client.abort(error_message)`` and exits with code 1
           on any exception.

        Returns
        -------
        list[str]
            Paths of the generated ``.ecf`` files.
        """
        generated: List[str] = []
        logger.info(f"Generating ecFlow wrapper scripts in {self.task_script_dir}")
        for node_id, data in self.dag.nodes(data=True):
            task_type = data["task_type"]
            family_name = _FAMILY_MAP[task_type]

            # Match ecFlow's default task script lookup: /suite/family/task.ecf
            script_path = Path(self.task_script_dir) / self.suite_name / family_name / f"{node_id}.ecf"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Writing wrapper script: {script_path}")
            with script_path.open("w") as fh:
                fh.write(_build_wrapper_script(node_id, self.python_executable))
            generated.append(str(script_path))

        return generated

    def _load_and_start(self, defs: Any) -> None:
        """Load the suite definition into the ecFlow server and begin it.

        Parameters
        ----------
        defs : ecflow.Defs
            The fully constructed suite definition.

        Raises
        ------
        RuntimeError
            If the connection to the ecFlow server fails.
        """
        try:
            client = self.ecflow.Client(self.host, self.port)
        except Exception as e:
            # Re-raise original exception as __cause__ to satisfy tests
            msg = f"Failed to connect to ecFlow server at {self.host}:{self.port} — {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        self.client = client

        logger.info(f"Connecting to ecFlow server at {self.host}:{self.port}")

        try:
            client.ping()
        except Exception as e:
            msg = f"Could not connect to ecFlow server at {self.host}:{self.port}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        try:
            logger.info(f"Loading suite '{self.suite_name}' to server.")
            client.load(defs)
        except Exception as e:
            msg = f"Failed to load suite definition to ecFlow server at {self.host}:{self.port}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        try:
            logger.info(f"Beginning suite '{self.suite_name}'.")
            client.begin_suite(self.suite_name)
        except Exception as e:
            msg = f"Failed to begin suite '{self.suite_name}' on ecFlow server: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
