"""MolPy keeps both bridging patterns, and depends on neither.

:mod:`molpy.adapter` (in-memory object sync) and :mod:`molpy.wrapper` /
:mod:`molpy.pack` (shell out to a binary) each keep one worked example —
:class:`RDKitAdapter` and the Packmol packer. They are examples, not
dependencies: importing molpy must never require them, and no molpy code path
may route through them for something molpy does natively.

That makes the rule positional rather than absolute: ``adapter/`` is the *one*
place allowed to import a third-party chemistry library. The scan below is the
durable form of it. A per-call-site ``monkeypatch`` only proves one call site
does not reach for RDKit; this proves the rest of the tree cannot.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import molpy.adapter as adapter_mod

SRC = Path(__file__).resolve().parents[2] / "src" / "molpy"

#: Third-party chemistry that molpy must not import *outside* the sanctioned
#: bridge packages. AmberTools and Packmol are absent on purpose — they are
#: subprocesses, not imports, so no import scan would ever see them.
#:
#: :mod:`molpy.engine` is exempt (see :data:`DRIVER_PACKAGES`). An engine's job
#: *is* to drive an external simulator: LAMMPS and CP2K are driven by generated
#: script + subprocess, and so is OpenMM, whose Python import is deferred into
#: the one function that reads its output. Driving a simulator is not the same
#: as leaning on a library for molpy's own chemistry, which is what this rule
#: is about.
FORBIDDEN = frozenset(
    {
        "rdkit",
        "openbabel",
        "pybel",
        "openmm",
        "simtk",
        "parmed",
        "MDAnalysis",
        "ase",
        "pymatgen",
    }
)


#: Sub-packages allowed to name a third-party library: the two bridging
#: exemplars, plus the engines whose whole job is to drive an external program
#: (LAMMPS and CP2K by generated script + subprocess, and OpenMM the same, its
#: Python import deferred into the one function that reads its output).
BRIDGE_PACKAGES = ("adapter", "engine")


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by one module, from its AST."""
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestNoThirdPartyChemistry:
    def test_the_scan_actually_sees_the_source_tree(self):
        # Without this, every assertion below passes on an empty file list.
        modules = list(SRC.rglob("*.py"))
        assert len(modules) > 100, f"only found {len(modules)} modules under {SRC}"
        assert any(m.name == "atomistic.py" for m in modules)

    def test_no_module_imports_a_third_party_chemistry_library(self):
        offenders: list[str] = []
        for module in sorted(SRC.rglob("*.py")):
            if module.relative_to(SRC).parts[0] in BRIDGE_PACKAGES:
                continue
            hit = _imported_roots(module) & FORBIDDEN
            if hit:
                offenders.append(f"{module.relative_to(SRC)}: {sorted(hit)}")
        assert offenders == [], (
            "molpy source imports third-party chemistry:\n  "
            + "\n  ".join(offenders)
            + "\nUse the native path (Conformer / Perceive / SmilesIR), or shell "
            "out from molpy.wrapper if it is AmberTools."
        )

    def test_the_forbidden_list_is_not_empty_and_names_the_bridged_libraries(self):
        # A gate over an empty list is not a gate.
        assert {"rdkit", "openbabel"} <= FORBIDDEN

    def test_the_bridge_carve_out_stays_narrow(self):
        """The exemption is two packages, and it must not creep."""
        assert BRIDGE_PACKAGES == ("adapter", "engine")
        for package in BRIDGE_PACKAGES:
            assert (SRC / package).is_dir()

    def test_importing_molpy_does_not_import_the_bridged_library(self):
        """The example must not become a dependency — installed or not.

        Asserting rdkit is *absent from the environment* would make this a
        claim about the machine rather than about molpy: it goes red the
        moment someone installs the optional extra, and it cannot run at all
        in the environment where the adapter is exercised. What molpy owes is
        that a plain ``import molpy`` never pulls rdkit in, which is true
        either way. A subprocess is the only place to observe it — this
        interpreter may already have imported rdkit for another test.
        """
        probe = "import sys, molpy; print('rdkit' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False", (
            "importing molpy pulled rdkit into sys.modules; the adapter is an "
            "example, so its import must stay inside molpy.adapter"
        )

    def test_the_scan_would_catch_a_forbidden_import(self, tmp_path):
        """Proves the AST walk sees deferred imports, not just top-level ones."""
        planted = tmp_path / "planted.py"
        planted.write_text("def f():\n    import rdkit\n    return rdkit\n")
        assert _imported_roots(planted) & FORBIDDEN == {"rdkit"}

    def test_the_adapter_pattern_keeps_a_worked_example(self):
        assert "Adapter" in adapter_mod.__all__
        assert hasattr(adapter_mod, "RDKitAdapter"), (
            "RDKitAdapter is the worked example of the adapter pattern; it is "
            "optional, not absent"
        )
