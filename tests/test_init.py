"""Unit test for the ``molpy`` facade module (``src/molpy/__init__.py``)."""

from __future__ import annotations

import importlib

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

#: molrs subsystems flattened onto the molpy root.
FLATTENED = ("store", "system", "spatial", "units", "perceive", "optimize", "conformer")
#: molrs subsystems mirrored under the same name; molpy may add names there.
MIRRORED = ("io", "compute", "signal", "md", "op", "builder")
#: The native names molpy replaces with its own subclass.
SUBCLASSED = {"Box", "Trajectory"}


@pytest.mark.parametrize("sub", FLATTENED)
def test_a_flattened_subsystem_is_on_the_root_by_identity(sub: str) -> None:
    native = getattr(molrs, sub)
    for name in native.__all__:
        if name in SUBCLASSED:
            assert issubclass(getattr(molpy, name), getattr(native, name))
            continue
        assert getattr(molpy, name) is getattr(native, name), (sub, name)


@pytest.mark.parametrize("sub", MIRRORED)
def test_a_mirrored_subsystem_keeps_every_native_name(sub: str) -> None:
    native = getattr(molrs, sub)
    mine = importlib.import_module(f"molpy.{sub}")
    for name in native.__all__:
        assert getattr(mine, name) is getattr(native, name), (sub, name)
    assert set(native.__all__) <= set(mine.__all__)


def test_a_mirrored_name_is_not_also_on_the_root() -> None:
    mirrored = {name for sub in MIRRORED for name in getattr(molrs, sub).__all__}
    mirrored |= {
        name for sub in molrs.ff.__all__ for name in getattr(molrs.ff, sub).__all__
    }
    assert not (mirrored - SUBCLASSED) & set(molpy.__all__)


def test_every_root_name_resolves() -> None:
    for name in molpy.__all__:
        assert hasattr(molpy, name), name
