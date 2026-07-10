"""ecFlow orchestrator engine for MDT.

Translates a NetworkX DAG into an ecFlow suite definition, generates
per-task wrapper scripts, and submits the suite to an ecFlow server.

The ``ecflow`` package is imported lazily so that this module can be
loaded without ecFlow installed.
"""

import json
import logging
import sys
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
        "    zarr_enabled = '%ZARR_STORE_ENABLED%'\n"
        "    if zarr_enabled == 'true':\n"
        "        zarr_backend = '%ZARR_STORE_BACKEND%'\n"
        "        if zarr_backend == 'icechunk':\n"
        "            kwargs['use_icechunk'] = True\n"
        "        else:\n"
        "            kwargs['use_virtualizarr'] = True\n"
        "            kwargs['virtualizarr_backend'] = zarr_backend\n"
        "        zarr_path = '%ZARR_STORE_PATH%'\n"
        "        if zarr_path and zarr_backend != 'icechunk':\n"
        "            kwargs['store_path'] = zarr_path\n"
        "        icechunk_url = '%ZARR_STORE_ICECHUNK_URL%'\n"
        "        if icechunk_url:\n"
        "            kwargs['icechunk_url'] = icechunk_url\n"
        "    from mdt.tasks.data import load_data\n"
        "    load_data(name=task_name, dataset_type=dataset_type, kwargs=kwargs)"
    ),
    "pair_data": (
        "    from mdt.tasks.pairing import pair_data\n"
        "    pair_data(name=task_name, method=method,\n"
        "             source_data=None, target_data=None, kwargs=kwargs)"
    ),
    "combine_paired_data": (
        "    from mdt.tasks.pairing import combine_paired_data\n"
        "    paired_data_inputs = {}\n"
        "    for source in sources:\n"
        "        paired_data_inputs[source] = None  # In ecFlow, data passing is via disk/db\n"
        "    combine_paired_data(paired_data=paired_data_inputs, dim=dim)"
    ),
    "compute_statistics": (
        "    from mdt.tasks.statistics import compute_statistics\n"
        "    compute_statistics(name=task_name, metrics=metrics,\n"
        "                       input_data=None, kwargs=kwargs)"
    ),
    "generate_plot": (
        "    from mdt.tasks.plotting import generate_plot\n"
        "    generate_plot(name=task_name, plot_type=plot_type,\n"
        "                  input_data=None, kwargs=kwargs)"
    ),
    "calculate_reduction": (
        "    from mdt.tasks.reductions import calculate_reduction\n"
        "    calculate_reduction(obj=None, method=method, dim=dim,\n"
        "                        force_weighted=force_weighted, **kwargs)"
    ),
    "save_data": (
        "    from mdt.tasks.data import save_data\n"
        "    save_data(name=task_name, data=None, backend=backend, url=url, kwargs=kwargs)"
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
        "import json, os, sys",
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
        '    task_type = "%TASK_TYPE%"',
        '    task_name = "%TASK_NAME%"',
        '    dataset_type = "%DATASET_TYPE%"',
        "    kwargs = json.loads('%TASK_KWARGS%')",
        "    metrics = json.loads('%METRICS%')",
        '    plot_type = "%PLOT_TYPE%"',
        '    method = "%METHOD%"',
        '    dim = "%DIM%"',
        "    sources = json.loads('%SOURCES%')",
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
            raise ImportError("ecFlow is not installed. Install with: pip install mdt[ecflow]") from e

        self.dag = dag
        self.config = config
        self.ecflow = ecflow

        exec_cfg = config.execution
        self.host: str = exec_cfg.get("ecflow_host", "localhost")
        self.port: int = int(exec_cfg.get("ecflow_port", 3141))
        self.suite_name: str = exec_cfg.get("suite_name", "mdt")
        self.task_script_dir: str = exec_cfg.get("task_script_dir", "./ecflow_tasks/")
        self.python_executable: str = sys.executable

    # ------------------------------------------------------------------
    # Engine ABC
    # ------------------------------------------------------------------

    def execute(self) -> Dict[str, Any]:
        """Build the suite, generate wrappers, and start the suite.

        Returns
        -------
        dict
            ``{"suite": <suite_name>, "status": "started"}``
        """
        logger.info(f"Starting ecFlow workflow orchestration for suite '{self.suite_name}'")
        defs = self.build_suite()
        self.generate_task_wrappers()

        # Save the suite definition to a file for user inspection
        Path(self.task_script_dir).mkdir(parents=True, exist_ok=True)
        def_path = Path(self.task_script_dir) / f"{self.suite_name}.def"
        logger.info(f"Saving ecFlow suite definition to {def_path}")
        defs.save_as_defs(str(def_path))

        self._load_and_start(defs)
        return {"suite": self.suite_name, "status": "started"}

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
            task_node.add_variable("CLUSTER", data.get("cluster") or "")
            task_node.add_variable("BACKEND", data.get("backend") or "")
            task_node.add_variable("URL", data.get("url") or "")

            # Always define ZARR_* so ecFlow token substitution succeeds for all
            # generated wrappers, even for non-load tasks.
            task_node.add_variable("ZARR_STORE_ENABLED", "false")
            task_node.add_variable("ZARR_STORE_BACKEND", "")
            task_node.add_variable("ZARR_STORE_PATH", "")
            task_node.add_variable("ZARR_STORE_ICECHUNK_URL", "")

            if task_type == "load_data":
                node_kwargs = data.get("kwargs") or {}
                task_node.add_variable(
                    "ZARR_STORE_ENABLED",
                    str(node_kwargs.get("use_virtualizarr", False)).lower(),
                )
                task_node.add_variable(
                    "ZARR_STORE_BACKEND",
                    "icechunk" if node_kwargs.get("use_icechunk", False) else node_kwargs.get("virtualizarr_backend", ""),
                )
                task_node.add_variable(
                    "ZARR_STORE_PATH",
                    node_kwargs.get("store_path", ""),
                )
                task_node.add_variable(
                    "ZARR_STORE_ICECHUNK_URL",
                    node_kwargs.get("icechunk_url", ""),
                )

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
