"""Doc-example smoke tests — the "docs never rot" defense line.

Three classes of user-facing code are exercised here:

1. Every ``python`` code block in docs/api/builder.md and
   docs/api/reacter.md is executed block-by-block (shared namespace per
   document, so later blocks may reuse earlier definitions). Whole-file
   skips are not allowed; only an RDKit-missing environment skips the
   blocks that need 3D embedding.
2. A targeted doctest for src/molpy/reacter/base.py, plus source-level
   assertions that the class docstring examples reference only real symbols.
3. The ``examples/`` scripts' ``main()`` entry points.

Also locks the API surface: lazy top-level submodules and
Selector as the public extension point.
"""

import doctest
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_API = REPO_ROOT / "docs" / "api"
EXAMPLES = REPO_ROOT / "examples"

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None


def _python_blocks(md_path: Path) -> list[str]:
    """Extract ``python`` fenced code blocks from a markdown file."""
    text = md_path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def _exec_blocks(md_name: str) -> None:
    md_path = DOCS_API / md_name
    blocks = _python_blocks(md_path)
    assert blocks, f"{md_name} contains no python code blocks"
    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        code = compile(block, f"{md_name}[python block {index}]", "exec")
        # Executing our own documentation is the point of this test.
        exec(code, namespace)


class TestApiDocExamples:
    """docs/api/*.md python blocks must execute as written."""

    @pytest.mark.skipif(not _HAS_RDKIT, reason="builder examples embed 3D via RDKit")
    def test_builder_md_blocks_execute(self):
        _exec_blocks("builder.md")

    def test_reacter_md_blocks_execute(self):
        _exec_blocks("reacter.md")

    def test_reacter_md_references_real_symbols(self):
        text = (DOCS_API / "reacter.md").read_text(encoding="utf-8")
        for stale in ("TemplateReacter", "select_nothing", "product_info"):
            assert stale not in text, f"reacter.md references stale symbol {stale!r}"

    def test_builder_md_no_agent_infrastructure(self):
        text = (DOCS_API / "builder.md").read_text(encoding="utf-8")
        for leaked in ("ToolRegistry", "molpy.builder._tool"):
            assert leaked not in text, f"builder.md leaks agent infra {leaked!r}"

    def test_builder_md_documents_the_selector_extension_point(self):
        """Selection is the one variation point of the assembly kernel.

        (It replaces ReactionPresets, which existed to name Reacter chemistries;
        chemistry now lives in the reaction SMARTS itself.)
        """
        text = (DOCS_API / "builder.md").read_text(encoding="utf-8")
        assert "class FirstPairSelector(Selector)" in text
        assert "ReactionPresets" not in text


class TestDocstringDoctests:
    """Targeted doctests + stale-symbol grep on class docstring examples."""

    def test_reacter_base_doctests_pass(self):
        import importlib

        base_module = importlib.import_module("molpy.reacter.base")

        results = doctest.testmod(base_module, verbose=False)
        assert results.failed == 0, f"{results.failed} doctest failures in base.py"
        assert results.attempted > 0, "base.py should carry a runnable doctest"

    def test_reacter_base_example_uses_real_api(self):
        source = (REPO_ROOT / "src/molpy/reacter/base.py").read_text(encoding="utf-8")
        for stale in ("select_port_atom", "port_selector_left", "port_L="):
            assert stale not in source, f"base.py example references {stale!r}"


class TestTopLevelSurface:
    """Lazy top-level submodules + consolidated export tables."""

    def test_mp_top_level_submodules_available(self):
        import molpy as mp

        for name in ("builder", "reacter", "pack", "compute"):
            assert hasattr(mp, name), f"molpy.{name} not reachable as attribute"
            assert name in mp.__all__, f"{name!r} missing from molpy.__all__"

    def test_lazy_submodules_not_loaded_at_import(self):
        import subprocess

        probe = (
            "import sys, molpy; "
            "loaded = [m for m in ('molpy.builder', 'molpy.pack') "
            "if m in sys.modules]; "
            "assert not loaded, loaded"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_assembly_surface_is_public(self):
        """One kernel, one variation point; chemistry lives in the reaction.

        Replaces the ReactionPresets surface: presets existed to name Reacter
        chemistries, and a reaction SMARTS names its own.
        """
        import molpy as mp
        from molpy.builder import GraphAssembler, PolymerBuilder, Selector

        assert issubclass(PolymerBuilder, GraphAssembler)
        assert hasattr(Selector, "select")
        # the reaction is constructed by the caller, from molpy, never molrs
        assert isinstance(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]"), mp.Reaction)

    def test_find_port_is_the_only_exported_name(self):
        import molpy.reacter as reacter_pkg

        assert "find_port" in reacter_pkg.__all__
        assert "find_port_atom" not in reacter_pkg.__all__
        assert not hasattr(reacter_pkg, "find_port_atom")
        # find_port_atom_by_node is explicitly out of scope and stays.
        assert hasattr(reacter_pkg, "find_port_atom_by_node")


def _run_example_main(script_name: str) -> None:
    script = EXAMPLES / script_name
    assert script.exists(), f"missing example script {script}"
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


class TestExampleScripts:
    """examples/ scripts stay runnable. 01-05 are pure molpy (native molrs
    conformer, no RDKit); 06 needs AmberTools and is not run here."""

    @pytest.mark.parametrize(
        "script",
        [
            "01_parse_chemistry.py",
            "02_build_polymer.py",
            "03_polymer_topology.py",
            "04_crosslinking.py",
            "05_polydisperse.py",
        ],
    )
    def test_example_runs(self, script, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_example_main(script)
