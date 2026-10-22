"""Unit tests for :class:`molpy.builder.PackingTemplate`."""

from __future__ import annotations

import pytest

import molpy as mp


def _with_hydrogens(smiles: str) -> mp.Atomistic:
    return mp.perceive.add_hydrogens(mp.io.smiles.read_smiles(smiles).to_atomistic())


def test_hydrogens_are_the_h_rows_of_the_frame() -> None:
    template = mp.builder.PackingTemplate(_with_hydrogens("CCOCC"))

    atomic_numbers = template.frame["atoms"]["atomic_number"]
    assert template.hydrogens == tuple(
        i for i, number in enumerate(atomic_numbers) if number == 1
    )
    assert len(template.hydrogens) == 10


def test_frame_is_the_molecule_frame() -> None:
    mol = _with_hydrogens("CCO")
    template = mp.builder.PackingTemplate(mol)

    assert template.frame["atoms"].n_rows == mol.n_atoms
    assert template.frame["bonds"].n_rows == mol.to_frame()["bonds"].n_rows


def test_molecule_without_hydrogens_has_none() -> None:
    mol = mp.Atomistic()
    mol.add_atom(atomic_number=18, x=0.0, y=0.0, z=0.0)

    template = mp.builder.PackingTemplate(mol)

    assert template.hydrogens == ()


def test_missing_atomic_number_is_rejected_by_the_graph() -> None:
    mol = mp.Atomistic()
    with pytest.raises(ValueError, match="atomic_number"):
        mol.add_atom(x=0.0, y=0.0, z=0.0)
