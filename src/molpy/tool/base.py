"""Base abstractions for computation and tool operations.

Provides:

- ``Compute``: frozen-dataclass ABC for analysis operations (MSD, correlations)
- ``Tool``: frozen-dataclass ABC for executable tools (builders, transforms)
- ``ToolRegistry``: auto-discovery registry for ``Tool`` subclasses
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ToolRegistry:
    """Auto-discovery registry for Tool subclasses.

    Concrete Tool subclasses register themselves automatically via
    ``__init_subclass__``.  The MCP server iterates this registry
    to discover and expose available tools.

    Usage::

        for name, cls in ToolRegistry.get_all().items():
            print(f"{name}: {cls.__doc__}")
    """

    _tools: dict[str, type] = {}

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Return all registered concrete Tool subclasses."""
        return dict(cls._tools)

    @classmethod
    def get(cls, name: str) -> type | None:
        """Look up a Tool subclass by class name."""
        return cls._tools.get(name)


@dataclass(frozen=True)
class Compute(ABC):
    """Base class for analysis operations (MSD, correlations).

    Duck-type compatible with pydantic-graph ``BaseNode``:

    * Dataclass fields = configuration parameters (like Node fields).
    * ``run()`` method = core logic (like ``Node.run()``).
    * ``get_node_id()`` classmethod = unique identifier.

    Usage::

        msd = MSD(max_lag=3000)
        result = msd(positions)          # __call__ delegates to run()
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Core computation logic. Subclasses must implement."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke run() directly."""
        return self.run(*args, **kwargs)

    @classmethod
    def get_node_id(cls) -> str:
        """Unique node identifier (pydantic-graph compatible)."""
        return cls.__name__


@dataclass(frozen=True)
class Tool(ABC):
    """Base class for executable tools (builders, transforms).

    Concrete subclasses are auto-registered in ``ToolRegistry`` and
    discovered by the MCP server.  Unlike ``Compute`` (analysis-only),
    ``Tool`` is intended for molecular operations that produce or
    transform structures.

    Usage::

        @dataclass(frozen=True)
        class MyTool(Tool):
            param: int = 10

            def run(self, input: str) -> dict:
                return {"result": input, "param": self.param}

        tool = MyTool(param=5)
        result = tool("hello")  # delegates to run()
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Core tool logic. Subclasses must implement."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke run() directly."""
        return self.run(*args, **kwargs)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Register only concrete subclasses (those that define run())
        if "run" in cls.__dict__:
            ToolRegistry._tools[cls.__name__] = cls
