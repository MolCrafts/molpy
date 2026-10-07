"""Unit test for the ``molpy`` facade module (``src/molpy/__init__.py``)."""

from __future__ import annotations

import importlib
from types import ModuleType

import molrs
import pytest

import molpy


def test_unregistered_name_raises_attribute_error() -> None:
    """``__getattr__`` resolves lazy submodules only; anything else fails.

    A neutral name that was never registered must raise ``AttributeError``
    rather than falling through to an import attempt.
    """
    with pytest.raises(AttributeError, match="not_a_submodule"):
        molpy.not_a_submodule


# --- The module map: molpy is a thin layer over molrs -------------------------

#: The molrs subsystems with a molpy module of the same name; molpy may add
#: names there (``mp.builder`` its crystals and polymers, ``mp.core`` its
#: selectors, …).
MIRRORED = tuple(sorted(molrs.__all__))


@pytest.mark.parametrize("sub", MIRRORED)
def test_a_mirrored_subsystem_keeps_every_native_name(sub: str) -> None:
    native = getattr(molrs, sub)
    mine = importlib.import_module(f"molpy.{sub}")
    assert getattr(molpy, sub) is mine
    for name in native.__all__:
        ours = getattr(mine, name)
        if isinstance(ours, ModuleType) and ours.__name__.startswith("molpy."):
            continue  # molpy's own mirror of a molrs submodule (mp.io.lammps, …)
        assert ours is getattr(native, name), (sub, name)
    assert set(native.__all__) <= set(mine.__all__)


def test_only_promoted_core_classes_are_on_the_root() -> None:
    for name in set(dir(molpy)) & {
        n for sub in molrs.__all__ for n in getattr(molrs, sub).__all__
    }:
        obj = getattr(molpy, name)
        if isinstance(obj, ModuleType):
            continue
        assert obj is getattr(molpy.core, name), name


def test_every_root_name_resolves() -> None:
    for name in molpy.__all__:
        assert hasattr(molpy, name), name
