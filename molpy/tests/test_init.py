"""Unit test for the ``molpy`` facade module (``src/molpy/__init__.py``)."""

from __future__ import annotations

from types import ModuleType

import pytest

import molpy as mp


def test_unregistered_name_raises_attribute_error() -> None:
    """``__getattr__`` resolves lazy submodules only; anything else fails.

    A neutral name that was never registered must raise ``AttributeError``
    rather than falling through to an import attempt.
    """
    with pytest.raises(AttributeError, match="not_a_submodule"):
        getattr(mp, "not_a_submodule")


def test_every_exported_name_resolves() -> None:
    """Every name an ``__all__`` lists, from ``mp`` down, is there to take."""
    missing: list[str] = []
    seen: set[str] = set()
    queue: list[tuple[str, ModuleType]] = [("mp", mp)]
    while queue:
        path, module = queue.pop()
        if module.__name__ in seen:
            continue
        seen.add(module.__name__)
        for name in getattr(module, "__all__", ()):
            if not hasattr(module, name):
                missing.append(f"{path}.{name}")
                continue
            value = getattr(module, name)
            if isinstance(value, ModuleType):
                queue.append((f"{path}.{name}", value))
    assert not missing
