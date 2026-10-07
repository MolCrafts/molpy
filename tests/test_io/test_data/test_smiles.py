"""``mp.io.SmilesIR(...).to_atomistic()`` and ``mp.io.write_smarts``."""

from __future__ import annotations

import pytest

import molpy as mp


def test_smiles_to_atomistic_is_connectivity_only() -> None:
    mol = mp.io.SmilesIR("CCO").to_atomistic()
    assert isinstance(mol, mp.Atomistic)
    assert mol.n_atoms == 3


def test_smiles_refuses_brace_notation() -> None:
    with pytest.raises(ValueError):
        mp.io.SmilesIR("{[#EO]|3}").to_atomistic()


def test_write_smarts_local_environment() -> None:
    mol = mp.io.SmilesIR("CCO").to_atomistic()
    center = next(iter(mol.atoms))
    pattern = mp.io.write_smarts(mol, center.handle, reach=1, atomic_number=True)
    assert isinstance(pattern, str) and pattern
    assert "#" in pattern
    assert mp.SmartsPattern(pattern).find_matches(mol)
