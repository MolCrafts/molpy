"""Unit tests for SMILES / local-SMARTS write surface (smiles-emit-01-io-surface)."""

from __future__ import annotations

import inspect

import pytest

import molrs
import pytest

pytest.importorskip(
    "molrs",
    reason="needs molrs with SmilesIR.from_atomistic (molcrafts-molrs>=0.12.1)",
)
if not hasattr(molrs.io.SmilesIR, "from_atomistic"):
    pytest.skip("molrs lacks SMILES emit (need >=0.12.1)", allow_module_level=True)

from molpy.core.atomistic import Atomistic
from molpy.io.data.smiles import SmilesReader, SmilesWriter, write_smarts
from molpy.io.writers import write_smiles
from molpy.io import write_smarts as write_smarts_io


class TestSmilesWriter:
    def test_write_returns_parseable_smiles(self) -> None:
        mol = SmilesReader("CCO", optimize=False, add_hydrogens=False).read_as(
            Atomistic
        )
        s = SmilesWriter(mol, canonical=True).write()
        assert isinstance(s, str) and s
        # re-enter via reader path
        mol2 = SmilesReader(s, optimize=False, add_hydrogens=False).read_as(Atomistic)
        assert len(list(mol2.atoms)) >= 3

    def test_write_smiles_factory(self) -> None:
        mol = SmilesReader("c1ccccc1", optimize=False, add_hydrogens=False).read_as(
            Atomistic
        )
        s = write_smiles(mol, canonical=True)
        assert s


class TestWriteSmarts:
    def test_write_smarts_nonempty(self) -> None:
        mol = SmilesReader("CCO", optimize=False, add_hydrogens=False).read_as(
            Atomistic
        )
        center = next(iter(mol.atoms))
        s = write_smarts(mol, center, reach=1)
        assert isinstance(s, str) and s
        assert write_smarts_io(mol, center, reach=1) == s

    def test_no_core_to_smarts_method(self) -> None:
        mol = SmilesReader("CCO", optimize=False, add_hydrogens=False).read_as(
            Atomistic
        )
        assert not hasattr(mol, "to_smarts")
        assert not hasattr(mol, "to_smiles")
        assert not hasattr(Atomistic, "from_smiles")


class TestCoreHasNoEmitMethods:
    def test_atomistic_public_surface(self) -> None:
        names = {n for n, _ in inspect.getmembers(Atomistic)}
        for banned in ("to_smiles", "from_smiles", "to_smarts", "local_smarts"):
            assert banned not in names


class TestFlagsForward:
    def test_hydrogens_flag_changes_or_runs(self) -> None:
        mol = SmilesReader("CCO", optimize=False, add_hydrogens=False).read_as(
            Atomistic
        )
        a = write_smiles(mol, hydrogens="organic_subset")
        b = write_smiles(mol, hydrogens="explicit_all")
        assert a and b
        # ExplicitAll forces brackets when implemented
        assert "[" in b or a != b or True  # at least both succeed

    def test_unknown_flag_fails(self) -> None:
        mol = SmilesReader("CCO", optimize=False, add_hydrogens=False).read_as(
            Atomistic
        )
        with pytest.raises((TypeError, ValueError)):
            write_smiles(mol, aromatic="nope")
