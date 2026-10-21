"""Reading SMILES: ``read_smiles(text).to_atomistic()`` — one SMILES
molecule in, an ``Atomistic`` out, connectivity only."""

from __future__ import annotations

import molrs
import pytest

from molrs.io.smiles import read_smiles


def test_reads_one_molecule_without_adding_hydrogens() -> None:
    mol = read_smiles("CCO").to_atomistic()
    assert isinstance(mol, molrs.core.Atomistic)
    assert mol.n_atoms == 3
    assert mol.n_bonds == 2


def test_aromatic_ring_reads_its_six_atoms() -> None:
    ir = read_smiles("c1ccccc1")
    assert ir.n_components == 1
    mol = ir.to_atomistic()
    assert mol.n_atoms == 6
    assert mol.n_bonds == 6


def test_a_set_of_molecules_splits_into_its_components() -> None:
    ir = read_smiles("CCO.O")
    assert ir.n_components == 2
    parts = ir.components()
    assert [part.n_atoms for part in parts] == [3, 1]
    assert all(isinstance(part, molrs.core.Atomistic) for part in parts)
    # The one graph holds every component.
    assert ir.to_atomistic().n_atoms == 4


def test_invalid_smiles_is_a_smiles_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        read_smiles("C(")
    assert isinstance(excinfo.value, molrs.io.smiles.SmilesError)


def test_a_bare_str_is_not_a_reader_source() -> None:
    # SMILES text goes through ``read_smiles``; there is no ``_str`` door.
    assert not hasattr(molrs.io, "read_smiles_str")


@pytest.mark.parametrize(
    ("ir", "constructors"),
    [
        (molrs.io.smiles.SmilesIr, ("parse", "from_fragment", "from_atomistic")),
        (molrs.io.smiles.FragmentIr, ()),
        (molrs.io.cgsmiles.CgSmilesIr, ("parse",)),
    ],
)
def test_an_ir_is_handed_out_never_built_by_the_caller(ir, constructors) -> None:
    # The read_* / write_* doors are the only way in: no class constructor,
    # no static alternate constructor.
    with pytest.raises(TypeError):
        ir()
    for name in constructors:
        assert not hasattr(ir, name), f"{ir.__name__}.{name} is a second door"
