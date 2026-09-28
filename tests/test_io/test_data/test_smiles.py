"""SmilesReader: molrs parse plus hydrogen perception, no coordinates."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_MOLRS = importlib.util.find_spec("molrs") is not None

pytestmark = pytest.mark.skipif(not _HAS_MOLRS, reason="molrs required")


def test_smiles_reader_read_returns_frame() -> None:
    """``read()`` matches DataReader: tabular Frame, not Atomistic."""
    from molpy.io.data.smiles import SmilesReader
    from molrs import Frame

    frame = SmilesReader("CCO", add_hydrogens=True).read()
    assert isinstance(frame, Frame)
    assert "atoms" in frame
    assert frame["atoms"].nrows >= 3


def test_smiles_reader_read_as_atomistic() -> None:
    from molpy.core.atomistic import Atomistic
    from molpy.io.data.smiles import SmilesReader

    mol = SmilesReader("CCO", add_hydrogens=True).read_as(Atomistic)
    assert isinstance(mol, Atomistic)
    assert mol.n_atoms >= 3

    mol2 = SmilesReader("CCO", add_hydrogens=True).read_as("atomistic")
    assert isinstance(mol2, Atomistic)


def test_write_smarts_local_environment() -> None:
    import molpy as mp
    from molpy.core.atomistic import Atomistic

    mol = mp.io.read_smiles("CCO")
    assert isinstance(mol, Atomistic)
    center = next(iter(mol.atoms))
    pattern = mp.io.write_smarts(mol, center, reach=1, atomic_number=True)
    assert isinstance(pattern, str) and pattern
    assert "#" in pattern
    from molpy import SmartsPattern

    hits = SmartsPattern(pattern).find_matches(mol)
    assert hits


def test_smiles_reader_rejects_multi_component() -> None:
    from molpy.io.data.smiles import SmilesReader

    with pytest.raises(ValueError, match="single-component"):
        SmilesReader("CCO.O").read()


def test_smiles_reader_brace_notation_points_to_cgsmilesir() -> None:
    """Brace notation is refused with a pointer to the CGsmiles parser type."""
    from molpy.io.data.smiles import SmilesReader

    with pytest.raises(ValueError) as excinfo:
        SmilesReader("{[#EO]|3}").read()
    message = str(excinfo.value)
    assert "mp.CGSmilesIR(" in message
    assert ".to_coarsegrain()" in message
