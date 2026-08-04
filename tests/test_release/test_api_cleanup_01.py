"""release-0-12-molpy-01-api-cleanup — dual names and factories are gone.

These tests import leaf modules via importlib so they still run when the
installed molrs wheel is only used for package metadata checks at molpy
package import time is skipped — they read source surfaces of the dual APIs.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "molpy"


def test_amber_result_has_forcefield_only() -> None:
    tree = ast.parse((SRC / "builder" / "ambertools.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AmberResult"
    )
    ann = {
        t.target.id
        for t in cls.body
        if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
    }
    assert "forcefield" in ann
    assert "ff" not in ann
    # no @property named ff
    props = [
        n.name
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(d, ast.Name) and d.id == "property" for d in n.decorator_list
        )
    ]
    assert "ff" not in props


def test_get_topo_signature_no_entity_or_link_type() -> None:
    tree = ast.parse((SRC / "core" / "atomistic.py").read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_topo"
    )
    names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "entity_type" not in names
    assert "link_type" not in names


def test_get_packer_not_exported() -> None:
    pack_init = (SRC / "pack" / "__init__.py").read_text()
    tree = ast.parse(pack_init)
    all_list: list[str] = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if (
                    isinstance(t, ast.Name)
                    and t.id == "__all__"
                    and isinstance(n.value, (ast.List, ast.Tuple))
                ):
                    all_list = [
                        e.value
                        for e in n.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
    assert "get_packer" not in all_list
    assert "from .packer import" in pack_init or "Packmol" in pack_init
    for path in (SRC / "pack").rglob("*.py"):
        if "def get_packer" in path.read_text():
            raise AssertionError(
                f"get_packer still defined in {path.relative_to(ROOT)}"
            )
