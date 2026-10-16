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
import inspect
from pathlib import Path

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
