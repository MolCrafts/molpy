"""Smoke tests for the molpy MCP integration.

These tests verify that ``molpy_mcp.create_server()`` correctly wires
molmcp's introspection tools to the molpy package — i.e., real molpy
modules and classes are reachable through the seven introspection tools.

Generic tool behavior (error paths, prefix filtering, response limiting,
path safety) is covered in molmcp's own test suite. We don't duplicate
those here; we only verify the molpy <-> molmcp integration.
"""

import json

import pytest

from molpy_mcp import create_server

pytestmark = pytest.mark.mcp


class TestMolPyMCPIntegration:
    """End-to-end checks that molpy is exposed correctly via molmcp."""

    @pytest.fixture(autouse=True)
    def server(self):
        self.server = create_server("test-molpy")

    async def _call(self, tool: str, args: dict | None = None):
        result = await self.server.call_tool(tool, args or {})
        if not result.content:
            sc = result.structured_content
            return sc.get("result") if sc else None
        text = result.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    @pytest.mark.asyncio
    async def test_molpy_modules_reachable(self):
        modules = await self._call("list_modules")
        assert "molpy" in modules
        assert "molpy.core" in modules
        assert "molpy.core.atomistic" in modules

    @pytest.mark.asyncio
    async def test_molpy_core_symbols_reachable(self):
        symbols = await self._call("list_symbols", {"module": "molpy.core.atomistic"})
        assert isinstance(symbols, dict)
        assert "Atomistic" in symbols
        assert "Atom" in symbols
        assert "Bond" in symbols

    @pytest.mark.asyncio
    async def test_get_source_for_real_molpy_class(self):
        src = await self._call(
            "get_source", {"symbol": "molpy.core.atomistic.Atomistic"}
        )
        assert "class Atomistic" in src

    @pytest.mark.asyncio
    async def test_get_docstring_for_real_molpy_function(self):
        doc = await self._call(
            "get_docstring", {"symbol": "molpy.parser.parse_molecule"}
        )
        assert "SMILES" in doc

    @pytest.mark.asyncio
    async def test_search_finds_real_molpy_class(self):
        hits = await self._call(
            "search_source",
            {"query": "class Reacter", "module_prefix": "molpy.reacter"},
        )
        assert any("class Reacter" in h["text"] for h in hits)

    @pytest.mark.asyncio
    async def test_read_file_reads_molpy_source(self):
        text = await self._call(
            "read_file",
            {"relative_path": "molpy/__init__.py", "start": 1, "end": 5},
        )
        assert text  # not an error string
        assert "not found" not in text.lower()

    @pytest.mark.asyncio
    async def test_introspection_tools_registered(self):
        tools = await self.server.list_tools()
        names = {t.name for t in tools}
        # molmcp provides 7 introspection tools; assert all 7 are present.
        # We use a subset check rather than equality so future molpy domain
        # tools (when added via MolPyProvider) won't break this assertion.
        expected = {
            "list_modules",
            "list_symbols",
            "get_source",
            "get_docstring",
            "get_signature",
            "search_source",
            "read_file",
        }
        assert expected <= names

    @pytest.mark.asyncio
    async def test_all_tools_marked_read_only(self):
        tools = await self.server.list_tools()
        for tool in tools:
            assert tool.annotations is not None, f"{tool.name} missing annotations"
            assert tool.annotations.readOnlyHint is True, f"{tool.name} not read-only"
