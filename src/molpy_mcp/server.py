"""``create_server`` — the molpy MCP server factory."""

from __future__ import annotations

from molmcp import create_server as _molmcp_create_server

from molpy_mcp.provider import MolPyProvider


def create_server(name: str = "molpy"):
    """Build an MCP server exposing molpy via molmcp.

    The server registers:

    * molmcp's seven source-introspection tools, scoped to the ``molpy``
      package (``list_modules``, ``get_source``, ``search_source``, …).
    * :class:`MolPyProvider`, which currently has no domain tools but is
      the seam for future ``molpy``-specific MCP tools.

    Entry-point auto-discovery is disabled here so that
    :class:`MolPyProvider` is registered exactly once even when molpy is
    installed alongside other MolCrafts packages that also publish to the
    ``molmcp.providers`` group.

    Args:
        name: Server name advertised to MCP clients. Defaults to ``"molpy"``.

    Returns:
        A configured ``fastmcp.FastMCP`` instance ready to ``.run(...)``.
    """
    return _molmcp_create_server(
        name=name,
        import_roots=["molpy"],
        providers=[MolPyProvider()],
        discover_entry_points=False,
    )


__all__ = ["create_server"]
