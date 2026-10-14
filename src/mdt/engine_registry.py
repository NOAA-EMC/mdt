"""Engine abstraction layer and registry for MDT orchestrator backends."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Type

import networkx as nx

if TYPE_CHECKING:
    from mdt.config import ConfigParser


class Engine(ABC):
    """Base contract for all MDT orchestrator engines."""

    @abstractmethod
    def __init__(self, dag: nx.DiGraph, config: "ConfigParser"):
        """Initialize the engine with a DAG and configuration.

        Parameters
        ----------
        dag : nx.DiGraph
            The task dependency graph built by DAGBuilder.
        config : object
            The parsed MDT configuration object.
        """
        ...

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """Execute the workflow represented by the DAG.

        Returns
        -------
        dict
            A results dictionary whose shape depends on the engine.
        """
        ...


class EngineRegistry:
    """Maps orchestrator names to lazy engine factories.

    Factories are callables that import and return the concrete
    :class:`Engine` subclass on demand, raising :class:`ImportError`
    with install instructions when the optional dependency is missing.
    """

    _engines: ClassVar[Dict[str, Callable[[], Type[Engine]]]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], Type[Engine]]) -> None:
        """Register a lazy engine factory under *name*.

        Parameters
        ----------
        name : str
            Orchestrator name (e.g. ``"prefect"``, ``"ecflow"``).
        factory : callable
            A zero-argument callable that imports and returns the
            :class:`Engine` subclass.  It should raise
            :class:`ImportError` with install instructions if the
            dependency is not available.
        """
        cls._engines[name] = factory

    @classmethod
    def get_engine(cls, name: str) -> Type[Engine]:
        """Return the engine class for *name*, invoking the lazy factory.

        Parameters
        ----------
        name : str
            Orchestrator name previously passed to :meth:`register`.

        Returns
        -------
        type[Engine]
            The concrete engine class.

        Raises
        ------
        ValueError
            If *name* has not been registered.
        ImportError
            If the factory raises because the optional dependency is
            missing.
        """
        if name not in cls._engines:
            available = ", ".join(sorted(cls._engines)) or "(none)"
            raise ValueError(f"Unsupported orchestrator '{name}'. Available: {available}")
        return cls._engines[name]()


def _register_prefect() -> Type[Engine]:
    try:
        from mdt.engine import PrefectEngine
    except ImportError as e:
        raise ImportError("Prefect is not installed. Install with: pip install mdt[prefect]") from e
    return PrefectEngine


EngineRegistry.register("prefect", _register_prefect)


def _register_ecflow() -> Type[Engine]:
    try:
        from mdt.ecflow_engine import EcFlowEngine
    except ImportError as e:
        raise ImportError(
            "ecFlow Python bindings are not installed. MDT does not provide an ecFlow extra; install the ecFlow client package "
            "through your site's operational environment and ensure its Python bindings are available."
        ) from e
    return EcFlowEngine


EngineRegistry.register("ecflow", _register_ecflow)
