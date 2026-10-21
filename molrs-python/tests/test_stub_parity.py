"""Freshness guard for ``python/molrs/_native.pyi`` at class-name level.

It compares, in **both** directions, the top-level ``class`` names the stub
declares against the public classes the native module actually exports, so a
new pyclass that nobody declared — and a declaration whose class has been
deleted or moved to a ``.py`` module — both fail here rather than rotting.

Two facts decide how the comparison is set up:

* **The gate is the native wheel.** ``tox -e py`` builds the default-feature
  wheel with maturin and force-installs it before running pytest
  (``pyproject.toml:96-107``), so ``molrs._native`` here is that wheel's
  extension module and the class set is the default-feature surface.
* **The stub is read from the source tree**, not from ``molrs.__file__``.
  The same tox block asserts the imported package resolves under
  ``site-packages``: the install is non-editable, so resolving the stub next
  to the imported package would check a *copy* rather than the file a
  contributor edits.

The only exemptions are structural, never a name allowlist: a ``_native``
attribute that is a **module** (``_native.md``) may be declared in the stub as a
class, because a stub has no other way to describe a submodule; and private
(``_``-prefixed) names are compared on neither side.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest
from molrs import _native

STUB = Path(__file__).parents[1] / "python" / "molrs" / "_native.pyi"


def test_stub_declares_exactly_the_classes_native_exports() -> None:
    tree = ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))
    # Private names are excluded on both sides: a stub-only typing helper
    # (``_PotentialLike``) is no export, exactly as ``_native``'s private
    # attributes are not.
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    }

    exported: set[str] = set()
    submodules: set[str] = set()
    for name in dir(_native):
        if name.startswith("_"):
            continue
        value = getattr(_native, name)
        if isinstance(value, type):
            exported.add(name)
        elif inspect.ismodule(value):
            submodules.add(name)

    stub_only = declared - exported - submodules
    lib_only = exported - declared

    assert not stub_only and not lib_only, (
        f"{STUB} is out of date with molrs._native:\n"
        f"  declared in the stub but not exported by _native: {sorted(stub_only)}\n"
        f"  exported by _native but not declared in the stub: {sorted(lib_only)}"
    )


# ``molrs.core.keys`` / ``schema`` / ``constants`` copy their names from the
# native submodules at import time (``globals().update``), which a type
# checker cannot follow: each has a ``.pyi`` beside it declaring the names,
# and ``_native.pyi`` declares the submodules as classes. Both are held to
# the runtime names here, so a key, block or constant added in Rust fails
# until it is declared.
VOCABULARIES = ("keys", "schema", "constants")


def _declared(body: list[ast.stmt]) -> set[str]:
    """The public names a stub body binds (annotations, assignments, defs,
    classes)."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return {n for n in names if not n.startswith("_")}


def _runtime(name: str) -> set[str]:
    return set(getattr(importlib.import_module(f"molrs.core.{name}"), "__all__"))


@pytest.mark.parametrize("name", VOCABULARIES)
def test_the_vocabulary_stub_declares_exactly_the_runtime_names(name: str) -> None:
    stub = STUB.parent / "core" / f"{name}.pyi"
    declared = _declared(ast.parse(stub.read_text(encoding="utf-8")).body)
    runtime = _runtime(name)
    assert declared == runtime, (
        f"{stub} is out of date with molrs.core.{name}:\n"
        f"  declared only in the stub: {sorted(declared - runtime)}\n"
        f"  at run time only: {sorted(runtime - declared)}"
    )


@pytest.mark.parametrize("name", VOCABULARIES)
def test_the_native_stub_declares_exactly_the_runtime_names(name: str) -> None:
    tree = ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    declared = _declared(cls.body)
    runtime = {n for n in dir(getattr(_native, name)) if not n.startswith("_")}
    if name == "constants":
        runtime = {n for n in runtime if n.isupper()}
    assert declared == runtime, (
        f"{STUB} class {name} is out of date with molrs._native.{name}:\n"
        f"  declared only in the stub: {sorted(declared - runtime)}\n"
        f"  at run time only: {sorted(runtime - declared)}"
    )
