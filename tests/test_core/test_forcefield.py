import pytest

from molpy.core.forcefield import (AngleStyle, AngleType, AtomStyle, AtomType,
                                   BondStyle, BondType, DihedralStyle,
                                   DihedralType, ForceField, ImproperStyle,
                                   ImproperType, PairStyle, PairType, Style,
                                   Type)


class TestType:
    def test_init(self):
        t = Type(1, 2, param="value")
        assert t.params == [1, 2]
        assert t["param"] == "value"

    def test_eq(self):
        t1 = Type("type1")
        t2 = Type("type1")
        t3 = Type("type2")
        assert t1 == t2
        assert t1 != t3


class TestStyle:
    def test_init(self):
        s = Style("style1", 1, 2, param="value")
        assert s.name == "style1"
        assert s.params == (1, 2)
        assert s["param"] == "value"

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
        t = Type("type1", param="value")
        s.types.add(t)
        assert s.get_by(lambda x: x["param"] == "value") == t
        assert s.get_by(lambda x: x["param"] == "nonexistent") is None

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
        at = AtomType("atom1", param="value")
        assert at.name == "atom1"
        assert at["param"] == "value"


class TestBondType:
    def test_init(self):
        bt = BondType(AtomType("atom1"), AtomType("atom2"))
        assert bt.name == "atom1-atom2"
        assert bt.itomtype.name == "atom1"
        assert bt.jtomtype.name == "atom2"

    def test_atomtypes(self):
        bt = BondType(AtomType("atom1"), AtomType("atom2"))
        assert bt.atomtypes == [bt.itomtype, bt.jtomtype]


class TestAngleType:
    def test_init(self):
        at = AngleType(AtomType("atom1"), AtomType("atom2"), AtomType("atom3"))
        assert at.name == "atom1-atom2-atom3"

    def test_atomtypes(self):
        at = AngleType(AtomType("atom1"), AtomType("atom2"), AtomType("atom3"))
        assert at.atomtypes == [at.itomtype, at.jtomtype, at.ktomtype]


class TestDihedralType:
    def test_init(self):
        dt = DihedralType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert dt.name == "atom1-atom2-atom3-atom4"


    def test_atomtypes(self):
        dt = DihedralType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert dt.atomtypes == [dt.itomtype, dt.jtomtype, dt.ktomtype, dt.ltomtype]


class TestImproperType:
    def test_init(self):
        it = ImproperType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert it.name == f"atom1-atom2-atom3-atom4"

    def test_atomtypes(self):
        it = ImproperType(
            AtomType("atom1"), AtomType("atom2"), AtomType("atom3"), AtomType("atom4")
        )
        assert it.atomtypes == [it.itomtype, it.jtomtype, it.ktomtype, it.ltomtype]


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
