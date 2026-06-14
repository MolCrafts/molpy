"""Import-hygiene tests for molpy.reacter (spec builder-reacter-02-template-io).

The reacter package must not depend on molpy.io (or the molpy top-level
package) at import time: serialization belongs to the io layer, and the
upcoming refactor moves LAMMPS fix bond/react file writing into
``molpy.io.data.lammps_bond_react``. Both tests are RED on the pre-refactor
HEAD by design:

- ``molpy/__init__.py`` eagerly imports ``molpy.io``, so importing
  ``molpy.reacter`` (which pulls in ``molpy``) loads the io layer.
- ``molpy/reacter/bond_react.py`` does a top-level ``import molpy as mp``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REACTER_DIR = Path(__file__).resolve().parents[2] / "src" / "molpy" / "reacter"


def test_import_reacter_does_not_load_io() -> None:
    """Importing molpy.reacter must not pull molpy.io into sys.modules."""
    code = "import sys, molpy.reacter; assert 'molpy.io' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"importing molpy.reacter loaded molpy.io as a side effect:\n{proc.stderr}"
    )


def _is_type_checking_test(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` guards."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _type_checking_guarded_imports(tree: ast.Module) -> set[int]:
    """ids of Import/ImportFrom nodes nested inside a TYPE_CHECKING block."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for stmt in node.body:
                for child in ast.walk(stmt):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        guarded.add(id(child))
    return guarded


def test_no_toplevel_molpy_imports_in_reacter() -> None:
    """No reacter module may import the molpy top-level package.

    Allowed: ``from molpy.core...`` (and other non-io sub-packages);
    ``from molpy.typifier...`` only inside an ``if TYPE_CHECKING:`` block.
    Forbidden: ``import molpy`` / ``import molpy as ...`` and
    ``from molpy import ...`` anywhere in src/molpy/reacter/.
    """
    source_files = sorted(REACTER_DIR.glob("*.py"))
    assert source_files, f"no reacter sources found under {REACTER_DIR}"

    violations: list[str] = []
    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded = _type_checking_guarded_imports(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "molpy":
                        violations.append(
                            f"{path.name}:{node.lineno}: import molpy"
                            + (f" as {alias.asname}" if alias.asname else "")
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module == "molpy":
                    violations.append(
                        f"{path.name}:{node.lineno}: from molpy import ..."
                    )
                elif module.startswith("molpy.typifier") and id(node) not in guarded:
                    violations.append(
                        f"{path.name}:{node.lineno}: runtime import of {module} "
                        f"(only allowed under TYPE_CHECKING)"
                    )
    assert not violations, "top-level molpy imports in reacter:\n" + "\n".join(
        violations
    )
