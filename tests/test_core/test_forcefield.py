import pytest

from molpy.core.forcefield import (
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BondStyle,
    BondType,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairStyle,
    PairType,
    Style,
    Type,
)


class TestType:
    def test_init(self):
        t = Type("label", [1, 2], p1="value")
        assert t.param == [1, 2]
        assert t["p1"] == "value"

    def test_eq(self):
        t1 = Type(label="type1")
        t2 = Type(label="type1")
        t3 = Type(label="type2")
        assert t1 == t2
        assert t1 != t3


class TestStyle:
    def test_init(self):
        s = Style("style1", [1, 2], p1="value")
        assert s.name == "style1"
        assert s.oparam == (1, 2)
        assert s["p1"] == "value"

    def test_repr(self):
        s = Style("style1")
        assert repr(s) == "<Style: style1>"

    def test_n_types(self):
        s = Style("style1")
        assert s.n_types == 0
        s.types.add(Type("type1"))
        assert s.n_types == 1

    def test_get_by(self):
        s = Style("style1")
        t = Type("type1", p1="value")
        s.types.add(t)
        assert s.get_by(lambda x: x["p1"] == "value") == t
        assert s.get_by(lambda x: x["p1"] == "nonexistent") is None

    def test_merge(self):
        s1 = Style("style1")
        s2 = Style("style2")
        t1 = Type("type1")
        t2 = Type("type2")
        s2.types.add(t2)
        s1.merge(s2)
        assert t2 in s1.types


class TestAtomType:
    def test_init(self):
        at = AtomType("atom1", p1="value")
        assert at.label == "atom1"
        assert at["p1"] == "value"


class TestBondType:
    def test_init(self):
        bt = BondType(AtomType("atom1"), AtomType("atom2"))
        assert bt.label == "atom1-atom2"
        assert bt.itype.label == "atom1"
        assert bt.jtype.label == "atom2"

    def test_atomtypes(self):
        bt = BondType(AtomType("atom1"), AtomType("atom2"))
        assert bt.atomtypes == [bt.itype, bt.jtype]


class TestAngleType:
    def test_init(self):
        at = AngleType(AtomType("atom1"), AtomType("atom2"), AtomType("atom3"))
        assert at.label == "atom1-atom2-atom3"

    def test_atomtypes(self):
        at = AngleType(AtomType("atom1"), AtomType("atom2"), AtomType("atom3"))
        assert at.atomtypes == [at.itype, at.jtype, at.ktype]


class TestDihedralType:
    def test_init(self):
        dt = DihedralType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert dt.label == "atom1-atom2-atom3-atom4"

    def test_atomtypes(self):
        dt = DihedralType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert dt.atomtypes == [dt.itype, dt.jtype, dt.ktype, dt.ltype]


class TestImproperType:
    def test_init(self):
        it = ImproperType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert it.label == f"atom1-atom2-atom3-atom4"

    def test_atomtypes(self):
        it = ImproperType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert it.atomtypes == [it.itype, it.jtype, it.ktype, it.ltype]


class TestForceField:
    def test_init(self):
        ff = ForceField("forcefield1")
        assert ff.name == "forcefield1"
        assert ff.atomstyles == []

    def test_def_atomstyle(self):
        ff = ForceField()
        atomstyle = ff.def_atomstyle("style1")
        assert atomstyle.name == "style1"
        assert ff.get_atomstyle("style1") == atomstyle

    def test_merge(self):
        ff1 = ForceField("ff1")
        ff2 = ForceField("ff2")
        ff1.def_atomstyle("style1")
        ff2.def_atomstyle("style2")
        ff1.merge(ff2)
        assert ff1.get_atomstyle("style2") is not None
