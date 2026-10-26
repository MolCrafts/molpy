"""Seam tests for ForceField special-bonds declaration.

A declared value is observed through ``merge``: two force fields that declare
equal special bonds merge; two that declare different ones raise
``ValueError``; an undeclaring one adopts what it merges in.
"""

from __future__ import annotations

import molpy as mp
import pytest

_AMBER = ([0.0, 0.0, 0.5], [0.0, 0.0, 1.0 / 1.2])
_OTHER = ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])


def _declaring(
    name: str, weights: tuple[list[float], list[float]]
) -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField(name)
    ff.set_special_bonds(*weights)
    return ff


def test_set_special_bonds_wrong_length_raises():
    ff = mp.ff.forcefield.ForceField("x")
    with pytest.raises(ValueError):
        ff.set_special_bonds([0.5], [0.0, 0.0, 0.8333])


def test_set_special_bonds_declares_them():
    ff = _declaring("x", _AMBER)
    assert ff.merge(_declaring("same", _AMBER)) is ff
    with pytest.raises(ValueError):
        ff.merge(_declaring("other", _OTHER))


def test_merge_into_an_undeclaring_force_field_adopts_special_bonds():
    ff = mp.ff.forcefield.ForceField("dst")
    ff.merge(_declaring("src", _AMBER))
    assert ff.merge(_declaring("same", _AMBER)) is ff
    with pytest.raises(ValueError):
        ff.merge(_declaring("other", _OTHER))
