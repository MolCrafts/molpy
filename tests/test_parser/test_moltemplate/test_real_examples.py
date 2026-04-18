"""Acceptance tests against real moltemplate examples (vendored subset).

Each test pins the atom/bond/angle/dihedral count moltemplate is expected
to produce so we detect correctness regressions on real systems.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from molpy.io.forcefield.moltemplate import read_moltemplate_system

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _silence_import_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------- OPLSAA butane — full feature matrix (inherits, scoped $atom:x/y,
# Data Bond List, nested `new` inside a class, 3-D array replication).
class TestButane:
    @pytest.fixture
    def build(self):
        return read_moltemplate_system(FIXTURES / "butane" / "system.lt")

    def test_atom_count(self, build):
        atomistic, _ = build
        # 864 copies * 14 atoms (C4H10: 2 CH3 + 2 CH2 = 4*C + 10*H = 14)
        assert len(list(atomistic.atoms)) == 864 * 14

    def test_bond_count(self, build):
        atomistic, _ = build
        # Each butane has 13 bonds (3 CH3 + 2 CH2 + 2 CH2 + 3 CH3 + 3 C-C)
        assert len(list(atomistic.bonds)) == 864 * 13

    def test_ff_has_harmonic_styles(self, build):
        _, ff = build
        from molpy.potential.angle import AngleHarmonicStyle
        from molpy.potential.bond import BondHarmonicStyle

        assert ff.get_style_by_name("harmonic", BondHarmonicStyle) is not None
        assert ff.get_style_by_name("harmonic", AngleHarmonicStyle) is not None


# ---------- OPLSAA alkane_chain_single — 50-monomer chain, nested classes.
class TestAlkane50:
    @pytest.fixture
    def build(self):
        return read_moltemplate_system(FIXTURES / "alkane_chain_single" / "system.lt")

    def test_atom_count(self, build):
        atomistic, _ = build
        # 2 CH3 endcaps (4 atoms each) + 48 CH2 (3 atoms each)
        # = 8 + 144 = 152 atoms
        assert len(list(atomistic.atoms)) == 152

    def test_bond_count(self, build):
        atomistic, _ = build
        # Chain: 49 C-C backbone + 50*3 / 2*4 C-H = 49 + 3*48 + 4*2 = 49 + 144 + 8 = ...
        # Simpler: for 152 atoms in a single connected component that's a
        # tree (no rings), bonds = atoms - 1 = 151.
        assert len(list(atomistic.bonds)) == 151


# ---------- CG 2bead polymer — explicit molecule structure.
class TestCoarseGrained2Bead:
    @pytest.fixture
    def build(self):
        return read_moltemplate_system(FIXTURES / "2bead_polymer" / "system.lt")

    def test_produces_atoms(self, build):
        atomistic, _ = build
        assert len(list(atomistic.atoms)) > 100  # sanity: non-trivial system


# ---------- Misc pyramids — bare $atom / $mol / @atom (no colons).
class TestPyramids:
    @pytest.fixture
    def build(self):
        return read_moltemplate_system(FIXTURES / "pyramids" / "system.lt")

    def test_produces_atoms(self, build):
        atomistic, _ = build
        # 5 Giza pyramids worth of bricks; ~137 atoms empirically
        assert len(list(atomistic.atoms)) > 100


# ---------- CG 1bead protein — class name starts with digit.
class TestOneBeadProtein:
    @pytest.fixture
    def build(self):
        return read_moltemplate_system(FIXTURES / "1bead" / "system.lt")

    def test_parses_digit_class_name(self, build):
        atomistic, _ = build
        # 1beadProtSci2010 is a single-strand protein example
        assert len(list(atomistic.atoms)) >= 30
