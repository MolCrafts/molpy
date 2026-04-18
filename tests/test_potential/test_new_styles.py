"""Phase-2 style classes: registration + type storage."""

from __future__ import annotations

import pytest

from molpy.core.forcefield import (
    AngleType,
    AtomStyle,
    BondType,
    DihedralType,
    ForceField,
    ImproperType,
    PairType,
)
from molpy.potential.angle import (
    AngleClass2BondAngleStyle,
    AngleClass2BondBondStyle,
    AngleClass2Style,
)
from molpy.potential.bond import BondClass2Style, BondMorseStyle
from molpy.potential.dihedral import (
    DihedralCharmmStyle,
    DihedralClass2Style,
    DihedralMultiHarmonicStyle,
)
from molpy.potential.improper import (
    ImproperClass2Style,
    ImproperCvffStyle,
    ImproperHarmonicStyle,
)
from molpy.potential.pair import PairBuckStyle, PairLJClass2Style, PairMorseStyle


@pytest.fixture
def ff():
    f = ForceField(name="test")
    astyle = f.def_style(AtomStyle("full"))
    at1 = astyle.def_type(name="C", mass=12.011)
    at2 = astyle.def_type(name="H", mass=1.008)
    at3 = astyle.def_type(name="O", mass=15.999)
    at4 = astyle.def_type(name="N", mass=14.007)
    f._atom_types = [at1, at2, at3, at4]
    return f


class TestRegistry:
    @pytest.mark.parametrize(
        "kind,name",
        [
            ("bond", "morse"),
            ("bond", "class2"),
            ("angle", "class2"),
            ("angle", "class2/bb"),
            ("angle", "class2/ba"),
            ("dihedral", "charmm"),
            ("dihedral", "multi/harmonic"),
            ("dihedral", "class2"),
            ("improper", "harmonic"),
            ("improper", "cvff"),
            ("improper", "class2"),
            ("pair", "buck"),
            ("pair", "morse"),
            ("pair", "lj/class2"),
        ],
    )
    def test_kernel_registry_has_entry(self, kind, name):
        registry = ForceField._kernel_registry.get(kind, {})
        assert name in registry, f"{kind}:{name} not registered. Keys: {list(registry)}"


class TestBondStyles:
    def test_bond_morse_def_type(self, ff):
        at1, at2, *_ = ff._atom_types
        style = ff.def_style(BondMorseStyle())
        bt = style.def_type(at1, at2, D=100.0, alpha=2.0, r0=1.5)
        assert bt.name == "C-H"
        assert bt.get("D") == 100.0
        assert bt.get("alpha") == 2.0
        assert bt.get("r0") == 1.5
        assert bt in list(style.types.bucket(BondType))

    def test_bond_class2_def_type(self, ff):
        at1, _, _, _ = ff._atom_types
        style = ff.def_style(BondClass2Style())
        bt = style.def_type(at1, at1, r0=1.5, k2=300.0, k3=-400.0, k4=500.0)
        assert bt.get("k2") == 300.0


class TestAngleStyles:
    def test_angle_class2_def_type(self, ff):
        at1, at2, at3, _ = ff._atom_types
        style = ff.def_style(AngleClass2Style())
        ang = style.def_type(at1, at2, at3, theta0=109.5, k2=50.0, k3=-10.0, k4=1.0)
        assert ang.name == "C-H-O"
        assert ang.get("theta0") == 109.5


class TestDihedralStyles:
    def test_dihedral_charmm(self, ff):
        at1, at2, at3, at4 = ff._atom_types
        style = ff.def_style(DihedralCharmmStyle())
        dt = style.def_type(at1, at2, at3, at4, k=1.0, n=3, d=0.0, w=0.5)
        assert dt.get("n") == 3

    def test_dihedral_multi_harmonic(self, ff):
        at1, at2, at3, at4 = ff._atom_types
        style = ff.def_style(DihedralMultiHarmonicStyle())
        dt = style.def_type(at1, at2, at3, at4, a1=1.0, a2=2.0, a3=3.0, a4=4.0, a5=5.0)
        assert dt.get("a5") == 5.0


class TestImproperStyles:
    def test_improper_harmonic(self, ff):
        at1, at2, at3, at4 = ff._atom_types
        style = ff.def_style(ImproperHarmonicStyle())
        it = style.def_type(at1, at2, at3, at4, k=10.0, chi0=0.0)
        assert it.get("k") == 10.0

    def test_improper_cvff(self, ff):
        at1, at2, at3, at4 = ff._atom_types
        style = ff.def_style(ImproperCvffStyle())
        it = style.def_type(at1, at2, at3, at4, k=1.0, d=-1, n=2)
        assert it.get("d") == -1


class TestPairStyles:
    def test_pair_buck(self, ff):
        at1, at2, _, _ = ff._atom_types
        style = ff.def_style(PairBuckStyle())
        pt = style.def_type(at1, at2, A=1000.0, rho=0.3, C=50.0)
        assert pt.get("A") == 1000.0

    def test_pair_morse(self, ff):
        at1, at2, _, _ = ff._atom_types
        style = ff.def_style(PairMorseStyle())
        pt = style.def_type(at1, at2, D0=10.0, alpha=2.0, r0=3.5)
        assert pt.get("D0") == 10.0

    def test_pair_lj_class2(self, ff):
        at1, at2, _, _ = ff._atom_types
        style = ff.def_style(PairLJClass2Style())
        pt = style.def_type(at1, at2, epsilon=0.1, sigma=3.0)
        assert pt.get("epsilon") == 0.1
