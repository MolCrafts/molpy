"""``MolPyProvider`` — molpy's domain-tool registration seam.

molmcp's ``IntrospectionProvider`` already exposes ``molpy`` source code to
LLM agents. This Provider is the home for *domain* tools — build polymers,
parse SMILES, compute observables — once they are added.

The class is currently a noop on register(); it exists so that:

* ``create_server()`` can register it explicitly without import gymnastics,
* the ``molmcp.providers`` entry point can advertise it for auto-discovery,
* future domain tools have an obvious place to live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP


class MolPyProvider:
    """molpy's Provider for the molmcp Provider Protocol."""

    name: str = "molpy"

    def register(self, mcp: "FastMCP") -> None:
        """Register molpy domain tools on the MCP server.

        Args:
            mcp: The host server to register tools onto.
        """
        # Future domain tools (build/parse/compute) attach here, e.g.:
        #
        #     @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
        #     def parse_smiles(smiles: str) -> dict: ...
        return None


__all__ = ["MolPyProvider"]
