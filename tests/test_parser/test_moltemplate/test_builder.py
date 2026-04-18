"""Tests for the MolTemplate builder: IR -> ForceField + Atomistic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import AtomStyle, AtomType, ForceField
from molpy.parser.moltemplate import build_forcefield, build_system, parse_file
from molpy.potential.angle import AngleHarmonicStyle
from molpy.potential.bond import BondHarmonicStyle

FIXTURES = Path(__file__).parent / "fixtures"


class TestBuildForceField:
    def test_tip3p_masses(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        ff = build_forcefield(doc, base_dir=FIXTURES)
        astyle = ff.get_style_by_name("full", AtomStyle)
        assert astyle is not None
        by_name = {t.name: t for t in astyle.types.bucket(AtomType)}
        assert "O" in by_name
        assert abs(by_name["O"].get("mass") - 15.9994) < 1e-6
        assert abs(by_name["H"].get("mass") - 1.008) < 1e-6

    def test_tip3p_charges(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        ff = build_forcefield(doc, base_dir=FIXTURES)
        astyle = ff.get_style_by_name("full", AtomStyle)
        assert astyle is not None
        by_name = {t.name: t for t in astyle.types.bucket(AtomType)}
        assert abs(by_name["O"].get("charge") - (-0.834)) < 1e-6
        assert abs(by_name["H"].get("charge") - 0.417) < 1e-6

    def test_tip3p_has_bond_style(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        ff = build_forcefield(doc, base_dir=FIXTURES)
        assert ff.get_style_by_name("harmonic", BondHarmonicStyle) is not None

    def test_tip3p_has_angle_style(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        ff = build_forcefield(doc, base_dir=FIXTURES)
        assert ff.get_style_by_name("harmonic", AngleHarmonicStyle) is not None


class TestBuildSystem:
    def test_tip3p_two_waters(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        system, ff = build_system(doc, base_dir=FIXTURES)
        assert isinstance(system, Atomistic)
        # 2 water molecules x 3 atoms = 6 atoms
        assert len(system.atoms) == 6
        # Each water has 2 bonds, 1 angle
        assert len(system.bonds) == 4
        assert len(system.angles) == 2

    def test_tip3p_second_water_translated(self):
        doc = parse_file(FIXTURES / "tip3p.lt")
        system, _ff = build_system(doc, base_dir=FIXTURES)
        # The second `new` has .move(3, 0, 0): its O atom should be at x=3.0
        ox_positions = [a.get("x", 0.0) for a in system.atoms if a.get("type") == "O"]
        assert sorted(ox_positions) == [0.0, 3.0]
