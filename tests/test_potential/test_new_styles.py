"""Specialized force-field styles: registration + type storage via molrs.

molrs natively owns the force-field model. molpy adds only thin specialized
``Style`` subclasses (gap fillers for kernels molrs has no named class for); each
pins a style name and otherwise flows types/params through molrs. These tests
verify the specialized styles register on a ``ForceField`` and store typed
parameters through the native API.
"""

from __future__ import annotations

import pytest

from molpy.core.forcefield import (
    AngleClass2Style,
    AngleType,
    BondClass2Style,
    BondMorseStyle,
    BondType,
    DihedralCharmmStyle,
    DihedralClass2Style,
    DihedralMultiHarmonicStyle,
    DihedralType,
    ForceField,
    ImproperClass2Style,
    ImproperCvffStyle,
    ImproperHarmonicStyle,
    ImproperType,
    PairBuckStyle,
    PairLJClass2Style,
    PairMorseStyle,
    PairType,
)


@pytest.fixture
def ff():
    f = ForceField(name="test", units="real")
    astyle = f.def_atomstyle("full")
    astyle.def_type("C", mass=12.011)
    astyle.def_type("H", mass=1.008)
    astyle.def_type("O", mass=15.999)
    astyle.def_type("N", mass=14.007)
    return f


def _atom_types(ff):
    return ff.get_styles("atom")[0]


class TestStyleNames:
    @pytest.mark.parametrize(
        "factory,category,name",
        [
            (BondMorseStyle, "bond", "morse"),
            (BondClass2Style, "bond", "class2"),
            (AngleClass2Style, "angle", "class2"),
            (DihedralCharmmStyle, "dihedral", "charmm"),
            (DihedralMultiHarmonicStyle, "dihedral", "multi/harmonic"),
            (DihedralClass2Style, "dihedral", "class2"),
            (ImproperHarmonicStyle, "improper", "harmonic"),
            (ImproperCvffStyle, "improper", "cvff"),
            (ImproperClass2Style, "improper", "class2"),
            (PairBuckStyle, "pair", "buck"),
            (PairMorseStyle, "pair", "morse"),
            (PairLJClass2Style, "pair", "lj/class2"),
        ],
    )
    def test_style_identity(self, factory, category, name):
        style = factory()
        assert style.category == category
        assert style.name == name


class TestBondStyles:
    def test_bond_morse_def_type(self, ff):
        astyle = _atom_types(ff)
        c, h = astyle.get_type_by_name("C"), astyle.get_type_by_name("H")
        style = ff.def_style(BondMorseStyle())
        bt = style.def_type(c, h, D=100.0, alpha=2.0, r0=1.5)
        assert bt.name == "C-H"
        assert bt.get("D") == 100.0
        assert bt.get("alpha") == 2.0
        assert bt.get("r0") == 1.5
        assert bt.name in {t.name for t in style.get_types(BondType)}

    def test_bond_class2_def_type(self, ff):
        astyle = _atom_types(ff)
        c = astyle.get_type_by_name("C")
        style = ff.def_style(BondClass2Style())
        bt = style.def_type(c, c, r0=1.5, k2=300.0, k3=-400.0, k4=500.0)
        assert bt.get("k2") == 300.0


class TestAngleStyles:
    def test_angle_class2_def_type(self, ff):
        astyle = _atom_types(ff)
        c, h, o = (astyle.get_type_by_name(n) for n in ("C", "H", "O"))
        style = ff.def_style(AngleClass2Style())
        ang = style.def_type(c, h, o, theta0=109.5, k2=50.0, k3=-10.0, k4=1.0)
        assert ang.name == "C-H-O"
        assert ang.get("theta0") == 109.5
        assert ang.name in {t.name for t in style.get_types(AngleType)}


class TestDihedralStyles:
    def test_dihedral_charmm(self, ff):
        astyle = _atom_types(ff)
        c, h, o, n = (astyle.get_type_by_name(x) for x in ("C", "H", "O", "N"))
        style = ff.def_style(DihedralCharmmStyle())
        dt = style.def_type(c, h, o, n, k=1.0, n=3, d=0.0, w=0.5)
        assert dt.get("n") == 3
        assert dt.name in {t.name for t in style.get_types(DihedralType)}

    def test_dihedral_multi_harmonic(self, ff):
        astyle = _atom_types(ff)
        c, h, o, n = (astyle.get_type_by_name(x) for x in ("C", "H", "O", "N"))
        style = ff.def_style(DihedralMultiHarmonicStyle())
        dt = style.def_type(c, h, o, n, a1=1.0, a2=2.0, a3=3.0, a4=4.0, a5=5.0)
        assert dt.get("a5") == 5.0


class TestImproperStyles:
    def test_improper_harmonic(self, ff):
        astyle = _atom_types(ff)
        c, h, o, n = (astyle.get_type_by_name(x) for x in ("C", "H", "O", "N"))
        style = ff.def_style(ImproperHarmonicStyle())
        it = style.def_type(c, h, o, n, k=10.0, chi0=0.0)
        assert it.get("k") == 10.0
        assert it.name in {t.name for t in style.get_types(ImproperType)}

    def test_improper_cvff(self, ff):
        astyle = _atom_types(ff)
        c, h, o, n = (astyle.get_type_by_name(x) for x in ("C", "H", "O", "N"))
        style = ff.def_style(ImproperCvffStyle())
        it = style.def_type(c, h, o, n, k=1.0, d=-1, n=2)
        assert it.get("d") == -1


class TestPairStyles:
    def test_pair_buck(self, ff):
        astyle = _atom_types(ff)
        c, h = astyle.get_type_by_name("C"), astyle.get_type_by_name("H")
        style = ff.def_style(PairBuckStyle())
        pt = style.def_type(c, h, A=1000.0, rho=0.3, C=50.0)
        assert pt.get("A") == 1000.0
        assert pt.name in {t.name for t in style.get_types(PairType)}

    def test_pair_morse(self, ff):
        astyle = _atom_types(ff)
        c, h = astyle.get_type_by_name("C"), astyle.get_type_by_name("H")
        style = ff.def_style(PairMorseStyle())
        pt = style.def_type(c, h, D0=10.0, alpha=2.0, r0=3.5)
        assert pt.get("D0") == 10.0

    def test_pair_lj_class2(self, ff):
        astyle = _atom_types(ff)
        c, h = astyle.get_type_by_name("C"), astyle.get_type_by_name("H")
        style = ff.def_style(PairLJClass2Style())
        pt = style.def_type(c, h, epsilon=0.1, sigma=3.0)
        assert pt.get("epsilon") == 0.1
