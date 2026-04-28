"""molpy_mcp — molpy's MCP integration.

This package wires molpy into the Model Context Protocol via molmcp.
It does two things:

1. Re-exports :func:`create_server`, used by the ``molpy mcp`` CLI to
   launch a Model Context Protocol server backed by molmcp's
   source-introspection tools.
2. Re-exports :class:`MolPyProvider`, advertised via the
   ``molmcp.providers`` entry point so that any MCP host running molmcp
   discovers molpy automatically.

Future domain tools (build / parse / compute) will be registered on
:class:`MolPyProvider`.
"""

from __future__ import annotations

from molpy_mcp.provider import MolPyProvider
from molpy_mcp.server import create_server

__all__ = ["create_server", "MolPyProvider"]
