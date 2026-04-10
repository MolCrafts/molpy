"""Tests for the molpy-mcp source retrieval server.

Uses FastMCP's in-process ``call_tool`` — no subprocess, no HTTP, no curl.
"""

import json

import pytest

from molpy_mcp import create_server

pytestmark = pytest.mark.mcp


class TestMCPServer:
    """Test suite for the MolPy MCP server."""

    @pytest.fixture(autouse=True)
    def server(self):
        self.server = create_server("test-molpy")

    # -- helpers ----------------------------------------------------------

    async def _call(self, tool: str, args: dict | None = None):
        result = await self.server.call_tool(tool, args or {})
        if not result.content:
            return (
                result.structured_content.get("result")
                if result.structured_content
                else None
            )
        text = result.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    # -- list_modules -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_modules_returns_root(self):
        modules = await self._call("list_modules")
        assert "molpy" in modules
        assert "molpy.core" in modules
        assert "molpy.core.atomistic" in modules

    @pytest.mark.asyncio
    async def test_list_modules_prefix_filter(self):
        modules = await self._call("list_modules", {"prefix": "molpy.typifier"})
        assert all(m.startswith("molpy.typifier") for m in modules)
        assert "molpy.typifier" in modules
        assert len(modules) >= 3

    @pytest.mark.asyncio
    async def test_list_modules_no_match(self):
        modules = await self._call("list_modules", {"prefix": "nonexistent"})
        assert modules == []

    # -- list_symbols -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_symbols_core_atomistic(self):
        symbols = await self._call("list_symbols", {"module": "molpy.core.atomistic"})
        assert isinstance(symbols, dict)
        assert "Atomistic" in symbols
        assert "Atom" in symbols
        assert "Bond" in symbols

    @pytest.mark.asyncio
    async def test_list_symbols_bad_module(self):
        symbols = await self._call("list_symbols", {"module": "molpy.no_such_module"})
        assert "error" in symbols

    # -- get_source -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_source_class(self):
        src = await self._call("get_source", {"symbol": "molpy.tool.base.ToolRegistry"})
        assert "class ToolRegistry" in src
        assert "get_all" in src

    @pytest.mark.asyncio
    async def test_get_source_method(self):
        src = await self._call(
            "get_source", {"symbol": "molpy.core.atomistic.Atomistic.def_atom"}
        )
        assert "def def_atom" in src

    @pytest.mark.asyncio
    async def test_get_source_module(self):
        src = await self._call("get_source", {"symbol": "molpy.tool.base"})
        assert "ToolRegistry" in src

    @pytest.mark.asyncio
    async def test_get_source_not_found(self):
        src = await self._call(
            "get_source", {"symbol": "molpy.core.atomistic.NoSuchClass"}
        )
        assert "not found" in src.lower()

    # -- get_docstring ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_docstring_function(self):
        doc = await self._call(
            "get_docstring", {"symbol": "molpy.parser.parse_molecule"}
        )
        assert "SMILES" in doc

    @pytest.mark.asyncio
    async def test_get_docstring_class(self):
        doc = await self._call(
            "get_docstring", {"symbol": "molpy.core.atomistic.Atomistic"}
        )
        assert len(doc) > 10

    @pytest.mark.asyncio
    async def test_get_docstring_not_found(self):
        doc = await self._call("get_docstring", {"symbol": "molpy.fake.Fake"})
        assert "not found" in doc.lower()

    # -- get_signature ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_signature_function(self):
        sig = await self._call(
            "get_signature", {"symbol": "molpy.parser.parse_molecule"}
        )
        assert "smiles" in sig

    @pytest.mark.asyncio
    async def test_get_signature_method(self):
        sig = await self._call(
            "get_signature", {"symbol": "molpy.core.atomistic.Atomistic.def_atom"}
        )
        assert "def_atom" in sig
        assert "self" in sig

    @pytest.mark.asyncio
    async def test_get_signature_not_found(self):
        sig = await self._call("get_signature", {"symbol": "molpy.nope.Nope"})
        assert "not found" in sig.lower()

    # -- search_source ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_source_finds_class(self):
        hits = await self._call(
            "search_source",
            {
                "query": "class Reacter",
                "module_prefix": "molpy.reacter",
            },
        )
        assert len(hits) >= 1
        assert any("class Reacter" in h["text"] for h in hits)
        assert all({"file", "line", "text"} <= h.keys() for h in hits)

    @pytest.mark.asyncio
    async def test_search_source_respects_prefix(self):
        hits = await self._call(
            "search_source",
            {
                "query": "import",
                "module_prefix": "molpy.core.box",
            },
        )
        assert all(h["file"].startswith("molpy/core/box") for h in hits)

    @pytest.mark.asyncio
    async def test_search_source_no_match(self):
        hits = await self._call(
            "search_source",
            {
                "query": "xyzzy_impossible_string_42",
            },
        )
        assert hits == []

    @pytest.mark.asyncio
    async def test_search_source_limit(self):
        hits = await self._call("search_source", {"query": "import"})
        assert len(hits) <= 50

    # -- tool discovery ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_server_lists_all_tools(self):
        tools = await self.server.list_tools()
        names = {t.name for t in tools}
        assert names == {
            "list_modules",
            "list_symbols",
            "get_source",
            "get_docstring",
            "get_signature",
            "search_source",
        }
