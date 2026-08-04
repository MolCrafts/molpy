"""SmilesReader must use molrs (parse + Conformer)."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_MOLRS = importlib.util.find_spec("molrs") is not None

pytestmark = pytest.mark.skipif(not _HAS_MOLRS, reason="molrs required")


def test_smiles_reader_read_returns_frame() -> None:
    """``read()`` matches DataReader: tabular Frame, not Atomistic."""
    from molpy.io.data.smiles import SmilesReader
    from molrs import Frame

    frame = SmilesReader("CCO", add_hydrogens=True, optimize=False, seed=0).read()
    assert isinstance(frame, Frame)
    assert "atoms" in frame
    assert frame["atoms"].nrows >= 3


def test_smiles_reader_read_as_atomistic() -> None:
    from molpy.core.atomistic import Atomistic
    from molpy.io.data.smiles import SmilesReader

    mol = SmilesReader("CCO", add_hydrogens=True, optimize=False, seed=0).read_as(
        Atomistic
    )
    assert isinstance(mol, Atomistic)
    assert mol.n_atoms >= 3

    mol2 = SmilesReader("CCO", add_hydrogens=True, optimize=False, seed=0).read_as(
        "atomistic"
    )
    assert isinstance(mol2, Atomistic)


def test_smiles_reader_rejects_multi_component() -> None:
    from molpy.io.data.smiles import SmilesReader

    with pytest.raises(ValueError, match="single-component"):
        SmilesReader("CCO.O").read()
