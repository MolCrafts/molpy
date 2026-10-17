"""Python surface for SMILES emit (``str(SmilesIr.from_atomistic(mol))``) and the
SMARTS pattern of an atom's environment
(``molrs.perceive.SmartsPattern.from_environment``)."""

from __future__ import annotations

import molrs
import pytest


SmilesIr = molrs.io.smiles.SmilesIr


def test_emit_round_trip():
    mol = SmilesIr.parse("CCO").to_atomistic()
    s = str(SmilesIr.from_atomistic(mol, canonical=True))
    assert isinstance(s, str) and s
    assert SmilesIr.parse(s).to_atomistic().n_atoms == 3


def test_emit_round_trips_aromatic():
    mol = SmilesIr.parse("c1ccccc1").to_atomistic()
    s = str(SmilesIr.from_atomistic(mol, canonical=True))
    assert s
    assert SmilesIr.parse(s).to_atomistic().n_atoms == 6


def test_emit_root_is_an_atom_view():
    mol = SmilesIr.parse("CCO").to_atomistic()
    oxygen = next(a for a in mol.atoms if a["atomic_number"] == 8)
    s = str(SmilesIr.from_atomistic(mol, canonical=False, root=oxygen))
    assert s.startswith("O")
    with pytest.raises(ValueError):
        SmilesIr.from_atomistic(mol, root=0)


def test_environment_pattern_matches():
    mol = molrs.io.smiles.SmilesIr.parse("CCO").to_atomistic()
    center = mol.atoms[0]
    from molrs.perceive import SmartsPattern

    env = SmartsPattern.from_environment(mol, center, reach=1, atomic_number=True)
    s = str(env)
    assert isinstance(s, str) and s
    assert env.has_match(mol)
    assert SmartsPattern(s).has_match(mol)


def test_bad_aromatic_flag():
    mol = molrs.io.smiles.SmilesIr.parse("CCO").to_atomistic()
    with pytest.raises((ValueError, TypeError)):
        molrs.io.smiles.SmilesIr.from_atomistic(mol, aromatic="nope")


def test_bad_neighbor_style():
    mol = molrs.io.smiles.SmilesIr.parse("CCO").to_atomistic()
    center = mol.canonical_order()[0]
    with pytest.raises((ValueError, TypeError)):
        molrs.perceive.SmartsPattern.from_environment(mol, center, neighbor_style="x")
