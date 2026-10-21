"""Python surface for SMILES emit (``write_smiles(mol)``) and the
SMARTS pattern of an atom's environment
(``molrs.perceive.SmartsPattern.from_environment``)."""

from __future__ import annotations

import molrs
import pytest


from molrs.io.smiles import read_smiles, write_smiles


def test_emit_round_trip():
    mol = read_smiles("CCO").to_atomistic()
    s = write_smiles(mol, canonical=True)
    assert isinstance(s, str) and s
    assert read_smiles(s).to_atomistic().n_atoms == 3


def test_emit_round_trips_aromatic():
    mol = read_smiles("c1ccccc1").to_atomistic()
    s = write_smiles(mol, canonical=True)
    assert s
    assert read_smiles(s).to_atomistic().n_atoms == 6


def test_emit_root_is_an_atom_view():
    mol = read_smiles("CCO").to_atomistic()
    oxygen = next(a for a in mol.atoms if a["atomic_number"] == 8)
    s = write_smiles(mol, canonical=False, root=oxygen)
    assert s.startswith("O")
    with pytest.raises(ValueError):
        write_smiles(mol, root=0)


def test_environment_pattern_matches():
    mol = molrs.io.smiles.read_smiles("CCO").to_atomistic()
    center = mol.atoms[0]
    from molrs.perceive import SmartsPattern

    env = SmartsPattern.from_environment(mol, center, reach=1, atomic_number=True)
    s = str(env)
    assert isinstance(s, str) and s
    assert env.has_match(mol)
    assert SmartsPattern(s).has_match(mol)


def test_bad_aromatic_flag():
    mol = molrs.io.smiles.read_smiles("CCO").to_atomistic()
    with pytest.raises((ValueError, TypeError)):
        molrs.io.smiles.write_smiles(mol, aromatic="nope")


def test_bad_neighbor_style():
    mol = molrs.io.smiles.read_smiles("CCO").to_atomistic()
    center = mol.canonical_order()[0]
    with pytest.raises((ValueError, TypeError)):
        molrs.perceive.SmartsPattern.from_environment(mol, center, neighbor_style="x")
