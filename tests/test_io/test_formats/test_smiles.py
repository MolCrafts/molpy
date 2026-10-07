"""``mp.io.smiles.SmilesIr(...).to_atomistic()`` and ``mp.perceive.SmartsPattern.from_environment``."""

from __future__ import annotations

import pytest

import molpy as mp


def test_smiles_to_atomistic_is_connectivity_only() -> None:
    mol = mp.io.smiles.SmilesIr("CCO").to_atomistic()
    assert isinstance(mol, mp.Atomistic)
    assert mol.n_atoms == 3


def test_smiles_refuses_brace_notation() -> None:
    with pytest.raises(ValueError):
        mp.io.smiles.SmilesIr("{[#EO]|3}").to_atomistic()


def test_smarts_pattern_from_local_environment() -> None:
    mol = mp.io.smiles.SmilesIr("CCO").to_atomistic()
    center = next(iter(mol.atoms))
    pattern = mp.perceive.SmartsPattern.from_environment(
        mol, center.handle, reach=1, atomic_number=True
    )
    assert "#" in str(pattern)
    assert pattern.find_matches(mol)
