"""``mp.io.read_smiles`` and ``mp.io.write_smarts``."""

from __future__ import annotations

import pytest

import molpy as mp


def test_read_smiles_is_connectivity_only() -> None:
    mol = mp.io.read_smiles("CCO")
    assert isinstance(mol, mp.Atomistic)
    assert mol.n_atoms == 3


def test_read_smiles_refuses_brace_notation() -> None:
    with pytest.raises(ValueError):
        mp.io.read_smiles("{[#EO]|3}")


def test_write_smarts_local_environment() -> None:
    mol = mp.io.read_smiles("CCO")
    center = next(iter(mol.atoms))
    pattern = mp.io.write_smarts(mol, center.handle, reach=1, atomic_number=True)
    assert isinstance(pattern, str) and pattern
    assert "#" in pattern
    assert mp.SmartsPattern(pattern).find_matches(mol)
