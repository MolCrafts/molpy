"""Reading SMILES: ``SmilesIr.parse(text).to_atomistic()`` — one SMILES
molecule in, an ``Atomistic`` out, connectivity only."""

from __future__ import annotations

import molrs
import pytest

SmilesIr = molrs.io.smiles.SmilesIr


def test_reads_one_molecule_without_adding_hydrogens() -> None:
    mol = SmilesIr.parse("CCO").to_atomistic()
    assert isinstance(mol, molrs.core.Atomistic)
    assert mol.n_atoms == 3
    assert mol.n_bonds == 2


def test_aromatic_ring_reads_its_six_atoms() -> None:
    ir = SmilesIr.parse("c1ccccc1")
    assert ir.n_components == 1
    mol = ir.to_atomistic()
    assert mol.n_atoms == 6
    assert mol.n_bonds == 6


def test_a_set_of_molecules_splits_into_its_components() -> None:
    ir = SmilesIr.parse("CCO.O")
    assert ir.n_components == 2
    parts = ir.components()
    assert [part.n_atoms for part in parts] == [3, 1]
    assert all(isinstance(part, molrs.core.Atomistic) for part in parts)
    # The one graph holds every component.
    assert ir.to_atomistic().n_atoms == 4


def test_invalid_smiles_is_a_smiles_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        SmilesIr.parse("C(")
    assert isinstance(excinfo.value, molrs.io.smiles.SmilesError)


def test_a_bare_str_is_not_a_reader_source() -> None:
    # SMILES text goes through ``SmilesIr.parse``; there is no ``_str`` door.
    assert not hasattr(molrs.io, "read_smiles_str")
