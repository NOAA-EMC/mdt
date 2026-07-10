"""Command-line interface for the Model Development Tool (MDT)."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from mdt import __version__
from mdt.config import load_config
from mdt.dag import DAGBuilder
from mdt.engine_registry import EngineRegistry


def setup_logging(debug: bool = False) -> None:
    """Configures the standard logging format for MDT."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> None:
    """
    Entry point for the MDT CLI application.

    Parses command-line arguments and executes the requested subcommands.
    """
    parser = argparse.ArgumentParser(description="Monet Environmental Verification System (MDT)")
    parser.add_argument("--version", action="version", version=f"MDT {__version__}")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and show full stack traces on error.")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # The 'run' command
    run_parser = subparsers.add_parser("run", help="Run a verification workflow from a YAML config.")
    run_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    run_parser.add_argument(
        "--orchestrator",
        choices=["prefect", "ecflow"],
        default=None,
        help="Override the orchestrator backend specified in the YAML configuration.",
    )

    # The 'validate' command
    validate_parser = subparsers.add_parser("validate", help="Validate a YAML configuration file.")
    validate_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")

    # The 'template' command
    template_parser = subparsers.add_parser("template", help="Generate a sample YAML configuration file.")
    template_parser.add_argument("--output", "-o", type=str, default="config_template.yaml", help="Output filename.")

    args = parser.parse_args()
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    if args.command == "run":
        try:
            # 1. Load the configuration
            logger.info(f"Loading configuration from {args.config_path}")
            config = load_config(args.config_path)

            # 2. Build the Directed Acyclic Graph (DAG)
            logger.info("Constructing Task Graph")
            builder = DAGBuilder(config)
            dag = builder.build()

            # 3. Resolve orchestrator: CLI arg → YAML config → default ("prefect")
            orchestrator_name = args.orchestrator or config.orchestrator

            # 4. Initialize the execution engine via the registry and run the graph
            logger.info(f"Initializing '{orchestrator_name}' execution engine")
            engine_cls = EngineRegistry.get_engine(orchestrator_name)
            engine = engine_cls(dag, config)

            logger.info("Executing workflow")
            engine.execute()

            logger.info("Workflow execution completed successfully.")

            # Force clean shutdown to avoid segfault from Prefect server threads
            import os

            os._exit(0)

        except Exception as e:
            if args.debug:
                logger.exception("Error executing MDT workflow")
            else:
                logger.error(f"Error executing MDT workflow: {e}")
            sys.exit(1)

    elif args.command == "validate":
        try:
            logger.info(f"Validating configuration: {args.config_path}")
            config = load_config(args.config_path)
            # Try building the DAG as well, to catch reference errors
            builder = DAGBuilder(config)
            builder.build()
            print(f"SUCCESS: Configuration '{args.config_path}' is valid.")
        except Exception as e:
            if args.debug:
                logger.exception("Validation failed")
            else:
                logger.error(f"Validation failed: {e}")
            sys.exit(1)

    elif args.command == "template":
        template = {
            "data": {
                "my_model": {"type": "cmaq", "kwargs": {"fname": "path/to/cmaq_output.nc"}},
                "my_obs": {"type": "aeronet", "kwargs": {"fname": "path/to/aeronet_data.nc"}},
            },
            "pairing": {"eval_pair": {"source": "my_model", "target": "my_obs", "method": "interpolate"}},
            "statistics": {
                "basic_stats": {"input": "eval_pair", "metrics": ["rmse", "bias", "corr"], "kwargs": {"obs_var": "obs", "mod_var": "mod"}}
            },
            "plots": {"spatial_eval": {"input": "eval_pair", "type": "spatial", "kwargs": {"savename": "spatial_plot.png"}}},
            "execution": {"default_cluster": "local", "clusters": {"local": {"mode": "local", "workers": 2}}},
        }
        try:
            with Path(args.output).open("w") as f:
                yaml.dump(template, f, sort_keys=False)
            print(f"Generated template configuration: {args.output}")
        except Exception as e:
            logger.error(f"Failed to generate template: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
