"""MolPy MCP server — exposes source code retrieval for LLM agents."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

_MOLPY_SRC = Path(__file__).resolve().parent.parent / "molpy"


def _find_modules() -> list[str]:
    """Walk the molpy package tree and return all importable module paths."""
    import molpy

    modules: list[str] = ["molpy"]
    for info in pkgutil.walk_packages(molpy.__path__, prefix="molpy."):
        modules.append(info.name)
    return sorted(modules)


def _resolve_symbol(dotted: str) -> Any | None:
    """Import *dotted* name (module, module.attr, or module.Class.method)."""
    # Try as a module first
    try:
        return importlib.import_module(dotted)
    except ImportError:
        pass
    # Walk backwards through dots to find the deepest importable module,
    # then resolve remaining parts as attributes.
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        mod_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(mod_path)
        except ImportError:
            continue
        # Resolve remaining attribute chain
        for attr in parts[i:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj
    return None


def create_server(name: str = "molpy") -> FastMCP:
    """Create and return a configured MCP server with source retrieval tools."""
    mcp = FastMCP(
        name,
        instructions=(
            "MolPy source-code retrieval server. "
            "Use `list_modules` to discover packages, "
            "`get_source` to read implementation details."
        ),
    )

    @mcp.tool()
    def list_modules(prefix: str = "molpy") -> list[str]:
        """List all importable modules under the molpy package.

        Args:
            prefix: Filter modules whose name starts with this prefix.
                    Defaults to "molpy" (all modules).

        Returns:
            Sorted list of fully-qualified module names.
        """
        return [m for m in _find_modules() if m.startswith(prefix)]

    @mcp.tool()
    def list_symbols(module: str) -> dict[str, str]:
        """List public symbols exported by a module.

        Args:
            module: Fully-qualified module name, e.g. "molpy.core.atomistic".

        Returns:
            Dict mapping symbol name to a one-line summary (first line of
            docstring, or the symbol's type name if no docstring).
        """
        mod = _resolve_symbol(module)
        if mod is None or not inspect.ismodule(mod):
            return {"error": f"Module not found: {module}"}

        names = getattr(mod, "__all__", None)
        if names is None:
            names = [n for n in dir(mod) if not n.startswith("_")]

        result: dict[str, str] = {}
        for name in sorted(names):
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            doc = inspect.getdoc(obj)
            summary = doc.split("\n", 1)[0] if doc else type(obj).__name__
            result[name] = summary
        return result

    @mcp.tool()
    def get_source(symbol: str) -> str:
        """Retrieve the source code of a module, class, or function.

        Args:
            symbol: Fully-qualified dotted name.
                    Examples: "molpy.core.atomistic",
                              "molpy.core.atomistic.Atomistic",
                              "molpy.reacter.Reacter.run"

        Returns:
            Source code as a string, or an error message if not found.
        """
        obj = _resolve_symbol(symbol)
        if obj is None:
            return f"Symbol not found: {symbol}"
        try:
            return inspect.getsource(obj)
        except (TypeError, OSError):
            return f"Source not available for: {symbol}"

    @mcp.tool()
    def get_docstring(symbol: str) -> str:
        """Retrieve the docstring of a module, class, or function.

        Args:
            symbol: Fully-qualified dotted name.

        Returns:
            Cleaned docstring, or an error message if not found.
        """
        obj = _resolve_symbol(symbol)
        if obj is None:
            return f"Symbol not found: {symbol}"
        doc = inspect.getdoc(obj)
        return doc if doc else f"No docstring for: {symbol}"

    @mcp.tool()
    def get_signature(symbol: str) -> str:
        """Retrieve the call signature of a callable.

        Args:
            symbol: Fully-qualified dotted name of a class or function.

        Returns:
            Signature string, or an error message.
        """
        obj = _resolve_symbol(symbol)
        if obj is None:
            return f"Symbol not found: {symbol}"
        try:
            sig = inspect.signature(obj)
            return f"{symbol}{sig}"
        except (ValueError, TypeError):
            return f"No signature available for: {symbol}"

    @mcp.tool()
    def search_source(query: str, module_prefix: str = "molpy") -> list[dict[str, str]]:
        """Search for a string in molpy source files.

        Args:
            query: Text to search for (case-insensitive substring match).
            module_prefix: Restrict search to modules starting with this prefix.

        Returns:
            List of dicts with keys "file", "line", "text" for each match.
            Limited to 50 results.
        """
        if not _MOLPY_SRC.is_dir():
            return [{"error": "molpy source directory not found"}]

        results: list[dict[str, str]] = []
        q = query.lower()
        for py_file in sorted(_MOLPY_SRC.rglob("*.py")):
            rel = py_file.relative_to(_MOLPY_SRC.parent)
            mod_guess = str(rel).replace("/", ".").removesuffix(".py")
            if not mod_guess.startswith(module_prefix):
                continue
            try:
                lines = py_file.read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(lines, 1):
                if q in line.lower():
                    results.append(
                        {
                            "file": str(rel),
                            "line": str(i),
                            "text": line.strip(),
                        }
                    )
                    if len(results) >= 50:
                        return results
        return results

    return mcp
