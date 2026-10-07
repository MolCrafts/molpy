"""Unit tests for :class:`molpy.builder.PackingTemplate`."""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.builder import PackingTemplate


def _with_hydrogens(smiles: str) -> mp.Atomistic:
    return mp.perceive.Perceive().find_hydrogens(
        mp.io.smiles.SmilesIR(smiles).to_atomistic()
    )


def test_hydrogens_are_the_h_rows_of_the_frame() -> None:
    template = PackingTemplate(_with_hydrogens("CCOCC"))

    elements = [str(e) for e in template.frame["atoms"]["element"]]
    assert template.hydrogens == tuple(i for i, e in enumerate(elements) if e == "H")
    assert len(template.hydrogens) == 10


def test_frame_is_the_molecule_frame() -> None:
    mol = _with_hydrogens("CCO")
    template = PackingTemplate(mol)

    assert template.frame["atoms"].nrows == mol.n_atoms
    assert template.frame["bonds"].nrows == mol.to_frame()["bonds"].nrows


def test_molecule_without_hydrogens_has_none() -> None:
    mol = mp.Atomistic()
    mol.def_atom(element="Ar", x=0.0, y=0.0, z=0.0)

    template = PackingTemplate(mol)

    assert template.hydrogens == ()


def test_missing_element_raises() -> None:
    mol = mp.Atomistic()
    mol.def_atom(x=0.0, y=0.0, z=0.0)

    with pytest.raises(ValueError, match="element"):
        PackingTemplate(mol)
