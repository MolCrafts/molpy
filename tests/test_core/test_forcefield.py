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
        t = Type("type", parms=[1, 2], p1="value")
        assert t.parms == [1, 2]
        assert t["p1"] == "value"

    def test_eq(self):
        t1 = Type(name="type1")
        t2 = Type(name="type1")
        t3 = Type(name="type2")
        assert t1 == t2
        assert t1 != t3


class TestStyle:
    def test_init(self):
        s = Style("style1", [1, 2], p1="value")
        assert s.name == "style1"
        assert s.parms == [1, 2]
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
        assert at.name == "atom1"
        assert at["p1"] == "value"


class TestBondType:
    def test_init(self):
        bt = BondType(AtomType("atom1"), AtomType("atom2"))
        assert bt.name == "atom1-atom2"
        assert bt.itype.name == "atom1"
        assert bt.jtype.name == "atom2"

    def test_atomtypes(self):
        bt = BondType(AtomType("atom1"), AtomType("atom2"))
        assert bt.atomtypes == [bt.itype, bt.jtype]


class TestAngleType:
    def test_init(self):
        at = AngleType(AtomType("atom1"), AtomType("atom2"), AtomType("atom3"))
        assert at.name == "atom1-atom2-atom3"

    def test_atomtypes(self):
        at = AngleType(AtomType("atom1"), AtomType("atom2"), AtomType("atom3"))
        assert at.atomtypes == [at.itype, at.jtype, at.ktype]


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
        assert dt.atomtypes == [dt.itype, dt.jtype, dt.ktype, dt.ltype]


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

    def test_get_styles_and_types(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("a")
        bstyle = ff.def_bondstyle("b")
        angstyle = ff.def_anglestyle("ang")
        dstyle = ff.def_dihedralstyle("d")
        impstyle = ff.def_improperstyle("imp")
        pstyle = ff.def_pairstyle("p")
        assert ff.get_atomstyle("a") == astyle
        assert ff.get_bondstyle("b") == bstyle
        assert ff.get_anglestyle("ang") == angstyle
        assert ff.get_dihedralstyle("d") == dstyle
        assert ff.get_improperstyle("imp") == impstyle
        assert ff.get_pairstyle("p") == pstyle

    def test_get_types(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("a")
        bstyle = ff.def_bondstyle("b")
        angstyle = ff.def_anglestyle("ang")
        dstyle = ff.def_dihedralstyle("d")
        impstyle = ff.def_improperstyle("imp")
        pstyle = ff.def_pairstyle("p")
        at = astyle.def_type("A")
        bt = bstyle.def_type(at, at)
        angt = angstyle.def_type(at, at, at)
        dt = dstyle.def_type(at, at, at, at)
        impt = impstyle.def_type(at, at, at, at)
        pt = pstyle.def_type(at, at)
        assert at in ff.get_atomtypes()
        assert bt in ff.get_bondtypes()
        assert angt in ff.get_angletypes()
        assert dt in ff.get_dihedraltypes()
        assert impt in ff.get_impropertypes()
        assert pt in pstyle.get_types()

    def test_contains_and_getitem(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("a")
        bstyle = ff.def_bondstyle("b")
        assert "a" in ff
        assert "b" in ff
        assert ff["a"] == astyle
        assert ff["b"] == bstyle
        with pytest.raises(KeyError):
            _ = ff["notfound"]

    def test_len(self):
        ff = ForceField()
        assert len(ff) == 0
        ff.def_atomstyle("a")
        ff.def_bondstyle("b")
        ff.def_anglestyle("ang")
        ff.def_dihedralstyle("d")
        ff.def_improperstyle("imp")
        ff.def_pairstyle("p")
        assert len(ff) == 6
