"""Doc-example smoke tests — the "docs never rot" defense line.

Three classes of user-facing code are exercised here:

1. Every ``python`` code block in docs/api/builder.md and
   docs/api/reacter.md is executed block-by-block (shared namespace per
   document, so later blocks may reuse earlier definitions). Whole-file
   skips are not allowed; only an RDKit-missing environment skips the
   blocks that need 3D embedding.
2. Targeted doctests for src/molpy/reacter/base.py and
   src/molpy/builder/polymer/dsl.py, plus source-level assertions that
   the class docstring examples reference only real symbols.
3. The two ``examples/`` scripts' ``main()`` entry points.

Also locks the API surface consolidated by builder-reacter-04:
lazy top-level submodules, agent-only Tool classes out of user
``__all__``, ReactionPresets as the public extension point.
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

    def test_builder_md_documents_reaction_presets_register(self):
        text = (DOCS_API / "builder.md").read_text(encoding="utf-8")
        assert "ReactionPresets.register" in text


class TestDocstringDoctests:
    """Targeted doctests + stale-symbol grep on class docstring examples."""

    def test_reacter_base_doctests_pass(self):
        import importlib

        base_module = importlib.import_module("molpy.reacter.base")

        results = doctest.testmod(base_module, verbose=False)
        assert results.failed == 0, f"{results.failed} doctest failures in base.py"
        assert results.attempted > 0, "base.py should carry a runnable doctest"

    def test_dsl_doctests_pass(self):
        import importlib

        dsl_module = importlib.import_module("molpy.builder.polymer.dsl")

        results = doctest.testmod(dsl_module, verbose=False)
        assert results.failed == 0, f"{results.failed} doctest failures in dsl.py"

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

    def test_tool_classes_agent_only(self):
        import importlib

        import molpy.builder as builder_pkg

        # NOTE: the attribute molpy.builder.polymer is the polymer()
        # entry FUNCTION (star-import rebinds it over the subpackage);
        # the module itself is addressed via importlib/sys.modules.
        polymer_pkg = importlib.import_module("molpy.builder.polymer")
        tools_module = importlib.import_module("molpy.builder.polymer.tools")

        tool_names = {
            "PrepareMonomer",
            "BuildPolymer",
            "PlanSystem",
            "BuildSystem",
            "BuildPolymerAmber",
        }
        assert tool_names.isdisjoint(builder_pkg.__all__)
        assert tool_names.isdisjoint(polymer_pkg.__all__)
        for name in tool_names:
            assert hasattr(tools_module, name), f"tools.py missing {name}"

    def test_internal_components_demoted(self):
        import importlib

        polymer_pkg = importlib.import_module("molpy.builder.polymer")

        internal = {
            "GBigSmilesCompiler",
            "SystemPlanner",
            "PolydisperseChainGenerator",
        }
        assert internal.isdisjoint(polymer_pkg.__all__)

    def test_reaction_presets_public(self):
        from molpy.builder.polymer import ReactionPresets, ReactionPresetSpec

        assert callable(ReactionPresets.register)
        assert "dehydration" in ReactionPresets.list_presets()
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(ReactionPresetSpec)]
        assert "anchor_selector_left" in field_names
        assert "anchor_selector_right" in field_names

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
    """examples/ scripts stay runnable (RDKit-dependent paths skip)."""

    @pytest.mark.skipif(not _HAS_RDKIT, reason="polymer build embeds 3D via RDKit")
    def test_example_polymer_build(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_example_main("polymer_build.py")

    def test_example_reacter_bond_react_templates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_example_main("reacter_bond_react_templates.py")
