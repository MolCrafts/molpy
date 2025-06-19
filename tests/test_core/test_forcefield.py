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
    DictWithList,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairStyle,
    PairType,
    Style,
    StyleContainer,
    Type,
    TypeContainer,
)
from molpy.core.atomistic import Atom, Bond, Angle


class TestDictWithList:
    def test_init(self):
        d = DictWithList([1, 2, 3], {"key": "value"})
        assert d.parms == [1, 2, 3]
        assert d["key"] == "value"
        
    def test_getitem_int(self):
        d = DictWithList([1, 2, 3], {"key": "value"})
        assert d[0] == 1
        assert d[1] == 2
        assert d[2] == 3
        
    def test_getitem_str(self):
        d = DictWithList([1, 2, 3], {"key": "value"})
        assert d["key"] == "value"


class TestType:
    def test_init(self):
        t = Type("type1", parms=[1, 2], p1="value")
        assert t.name == "type1"
        assert t.parms == [1, 2]
        assert t["p1"] == "value"

    def test_eq_with_type(self):
        t1 = Type(name="type1")
        t2 = Type(name="type1")
        t3 = Type(name="type2")
        assert t1 == t2
        assert t1 != t3
        
    def test_eq_with_string(self):
        t = Type(name="type1")
        assert t == "type1"
        assert t != "type2"
        
    def test_eq_with_other_object(self):
        t = Type(name="type1")
        assert t != 42
        assert t != None
        assert t != []

    def test_hash(self):
        t1 = Type(name="type1")
        t2 = Type(name="type1")
        assert hash(t1) == hash(t2)
        
    def test_repr(self):
        t = Type(name="type1")
        assert repr(t) == "<Type: type1>"
        
    def test_str(self):
        t = Type(name="type1")
        assert str(t) == "type1"
        
    def test_name_property(self):
        t = Type(name="type1")
        assert t.name == "type1"
        t.name = "new_name"
        assert t.name == "new_name"
        
    def test_match_not_implemented(self):
        t = Type(name="type1")
        with pytest.raises(NotImplementedError):
            t.match(None)


class TestTypeContainer:
    def test_init(self):
        tc = TypeContainer()
        assert len(tc) == 0
        
    def test_add(self):
        tc = TypeContainer()
        t = Type("type1")
        tc.add(t)
        assert len(tc) == 1
        assert t in tc
        
    def test_iter(self):
        tc = TypeContainer()
        t1 = Type("type1")
        t2 = Type("type2")
        tc.add(t1)
        tc.add(t2)
        types = list(tc)
        assert t1 in types
        assert t2 in types
        
    def test_get(self):
        tc = TypeContainer()
        t = Type("type1")
        tc.add(t)
        assert tc.get("type1") == t
        assert tc.get("nonexistent") is None
        assert tc.get("nonexistent", "default") == "default"
        
    def test_get_all_by(self):
        tc = TypeContainer()
        t1 = Type("type1", p1="value1")
        t2 = Type("type2", p1="value2")
        t3 = Type("type3", p1="value1")
        tc.add(t1)
        tc.add(t2)
        tc.add(t3)
        results = tc.get_all_by(lambda t: t["p1"] == "value1")
        assert len(results) == 2
        assert t1 in results
        assert t3 in results
        
    def test_update(self):
        tc1 = TypeContainer()
        tc2 = TypeContainer()
        t1 = Type("type1")
        t2 = Type("type2")
        tc1.add(t1)
        tc2.add(t2)
        tc1.update(tc2)
        assert len(tc1) == 2
        assert tc1.get("type1") == t1
        assert tc1.get("type2") == t2


class TestStyle:
    def test_init(self):
        s = Style("style1", [1, 2], p1="value")
        assert s.name == "style1"
        assert s.parms == [1, 2]
        assert s["p1"] == "value"
        assert isinstance(s.types, TypeContainer)

    def test_repr(self):
        s = Style("style1")
        assert repr(s) == "<Style: style1>"

    def test_n_types(self):
        s = Style("style1")
        assert s.n_types == 0
        s.types.add(Type("type1"))
        assert s.n_types == 1

    def test_eq(self):
        s1 = Style("style1")
        s2 = Style("style1")
        s3 = Style("style2")
        assert s1 == s2
        assert s1 != s3
        assert s1 != "not a style"
        assert s1 != None

    def test_get_types(self):
        s = Style("style1")
        t1 = Type("type1")
        t2 = Type("type2")
        s.types.add(t1)
        s.types.add(t2)
        types = s.get_types()
        assert len(types) == 2
        assert t1 in types
        assert t2 in types

    def test_get_by(self):
        s = Style("style1")
        t = Type("type1", p1="value")
        s.types.add(t)
        assert s.get_by(lambda x: x["p1"] == "value") == t
        assert s.get_by(lambda x: x["p1"] == "nonexistent") is None

    def test_get(self):
        s = Style("style1")
        t = Type("type1")
        s.types.add(t)
        assert s.get("type1") == t
        assert s.get("nonexistent") is None

    def test_get_all_by(self):
        s = Style("style1")
        t1 = Type("type1", p1="value")
        t2 = Type("type2", p1="different")
        s.types.add(t1)
        s.types.add(t2)
        results = s.get_all_by(lambda x: x["p1"] == "value")
        assert len(results) == 1
        assert t1 in results

    def test_merge(self):
        s1 = Style("style1")
        s2 = Style("style2", data="value")
        t1 = Type("type1")
        t2 = Type("type2")
        s1.types.add(t1)
        s2.types.add(t2)
        s2["data"] = "value"
        
        result = s1.merge(s2)
        assert result == s1  # Returns self
        assert s1.get("type2") == t2
        assert s1["data"] == "value"


class TestStyleContainer:
    def test_init(self):
        sc = StyleContainer()
        assert len(list(sc)) == 0
        
    def test_add(self):
        sc = StyleContainer()
        s = Style("style1")
        sc.add(s)
        assert s in sc
        
    def test_iter(self):
        sc = StyleContainer()
        s1 = Style("style1")
        s2 = Style("style2")
        sc.add(s1)
        sc.add(s2)
        styles = list(sc)
        assert s1 in styles
        assert s2 in styles
        
    def test_get(self):
        sc = StyleContainer()
        s = Style("style1")
        sc.add(s)
        assert sc.get("style1") == s
        assert sc.get("nonexistent") is None
        assert sc.get("nonexistent", "default") == "default"


class TestAtomType:
    def test_init(self):
        at = AtomType("atom1", p1="value")
        assert at.name == "atom1"
        assert at["p1"] == "value"

    def test_match_valid_atom(self):
        at = AtomType("C")
        # Mock atom-like object
        atom = {"type": "C"}
        assert at.match(atom) == True
        
    def test_match_invalid_atom(self):
        at = AtomType("C")
        # Mock atom-like object with different type
        atom = {"type": "N"}
        assert at.match(atom) == False
        
    def test_match_invalid_entity(self):
        at = AtomType("C")
        assert at.match("not an atom") == False
        assert at.match(None) == False
        assert at.match({}) == False  # No 'type' key

    def test_apply(self):
        at = AtomType("C", mass=12.01)
        atom = {"type": "H"}
        at.apply(atom)
        assert atom["mass"] == 12.01
        assert atom["type"] == "H"  # Original type preserved


class TestBondType:
    def test_init(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        bt = BondType(at1, at2)
        assert bt.name == "atom1-atom2"
        assert bt.itype == at1
        assert bt.jtype == at2

    def test_init_with_name(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        bt = BondType(at1, at2, name="custom_bond")
        assert bt.name == "custom_bond"

    def test_atomtypes(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        bt = BondType(at1, at2)
        assert bt.atomtypes == [at1, at2]

    def test_match_valid_bond(self):
        at1 = AtomType("C")
        at2 = AtomType("H")
        bt = BondType(at1, at2)
        
        # Mock bond-like object
        class MockBond:
            def __init__(self, itype, jtype):
                self.itom = {"type": itype}
                self.jtom = {"type": jtype}
        
        bond = MockBond("C", "H")
        assert bt.match(bond) == True
        
        # Test reverse order
        bond_reverse = MockBond("H", "C")
        assert bt.match(bond_reverse) == True
        
    def test_match_invalid_bond(self):
        at1 = AtomType("C")
        at2 = AtomType("H")
        bt = BondType(at1, at2)
        
        class MockBond:
            def __init__(self, itype, jtype):
                self.itom = {"type": itype}
                self.jtom = {"type": jtype}
        
        bond = MockBond("C", "N")
        assert bt.match(bond) == False
        
    def test_match_invalid_entity(self):
        at1 = AtomType("C")
        at2 = AtomType("H")
        bt = BondType(at1, at2)
        
        assert bt.match("not a bond") == False
        assert bt.match(None) == False


class TestAngleType:
    def test_init(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at = AngleType(at1, at2, at3)
        assert at.name == "atom1-atom2-atom3"

    def test_init_with_name(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at = AngleType(at1, at2, at3, name="custom_angle")
        assert at.name == "custom_angle"

    def test_atomtypes(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at = AngleType(at1, at2, at3)
        assert at.atomtypes == [at1, at2, at3]

    def test_match_valid_angle(self):
        at1 = AtomType("C")
        at2 = AtomType("O")
        at3 = AtomType("H")
        angle_type = AngleType(at1, at2, at3)
        
        class MockAngle:
            def __init__(self, itype, jtype, ktype):
                self.itom = {"type": itype}
                self.jtom = {"type": jtype}
                self.ktom = {"type": ktype}
        
        angle = MockAngle("C", "O", "H")
        assert angle_type.match(angle) == True
        
        # Test reverse order (i and k can be swapped)
        angle_reverse = MockAngle("H", "O", "C")
        assert angle_type.match(angle_reverse) == True


class TestDihedralType:
    def test_init(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at4 = AtomType("atom4")
        dt = DihedralType(at1, at2, at3, at4)
        assert dt.name == "atom1-atom2-atom3-atom4"

    def test_atomtypes(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at4 = AtomType("atom4")
        dt = DihedralType(at1, at2, at3, at4)
        assert dt.atomtypes == [at1, at2, at3, at4]

    def test_match_valid_dihedral(self):
        at1 = AtomType("C")
        at2 = AtomType("C")
        at3 = AtomType("O")
        at4 = AtomType("H")
        dt = DihedralType(at1, at2, at3, at4)
        
        class MockDihedral:
            def __init__(self, itype, jtype, ktype, ltype):
                self.itom = {"type": itype}
                self.jtom = {"type": jtype}
                self.ktom = {"type": ktype}
                self.ltom = {"type": ltype}
        
        dihedral = MockDihedral("C", "C", "O", "H")
        assert dt.match(dihedral) == True
        
        # Test reverse order
        dihedral_reverse = MockDihedral("H", "O", "C", "C")
        assert dt.match(dihedral_reverse) == True


class TestImproperType:
    def test_init(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at4 = AtomType("atom4")
        it = ImproperType(at1, at2, at3, at4)
        assert it.name == f"atom1-atom2-atom3-atom4"

    def test_atomtypes(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        at3 = AtomType("atom3")
        at4 = AtomType("atom4")
        it = ImproperType(at1, at2, at3, at4)
        assert it.atomtypes == [at1, at2, at3, at4]

    def test_match_valid_improper(self):
        at1 = AtomType("C")
        at2 = AtomType("C")
        at3 = AtomType("O")
        at4 = AtomType("H")
        it = ImproperType(at1, at2, at3, at4)
        
        class MockImproper:
            def __init__(self, itype, jtype, ktype, ltype):
                self.itom = {"type": itype}
                self.jtom = {"type": jtype}
                self.ktom = {"type": ktype}
                self.ltom = {"type": ltype}
        
        improper = MockImproper("C", "C", "O", "H")
        assert it.match(improper) == True


class TestPairType:
    def test_init(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        pt = PairType(at1, at2)
        assert pt.name == "atom1-atom2"

    def test_atomtypes(self):
        at1 = AtomType("atom1")
        at2 = AtomType("atom2")
        pt = PairType(at1, at2)
        assert pt.atomtypes == [at1, at2]

    def test_match_valid_pair(self):
        at1 = AtomType("C")
        at2 = AtomType("H")
        pt = PairType(at1, at2)
        
        class MockPair:
            def __init__(self, itype, jtype):
                self.itom = {"type": itype}
                self.jtom = {"type": jtype}
        
        pair = MockPair("C", "H")
        assert pt.match(pair) == True
        
        # Test reverse order
        pair_reverse = MockPair("H", "C")
        assert pt.match(pair_reverse) == True


class TestAtomStyle:
    def test_init(self):
        ast = AtomStyle("lj", [])
        assert ast.name == "lj"
        assert isinstance(ast.classes, dict)

    def test_def_type(self):
        ast = AtomStyle("lj", [])
        at = ast.def_type("C", parms=[12.01])
        assert at.name == "C"
        assert at in ast.types
        assert ast.get("C") == at

    def test_def_type_with_class(self):
        ast = AtomStyle("lj", [])
        at = ast.def_type("C", class_="carbon", parms=[12.01])
        assert at.name == "C"
        assert "C" in ast.classes["carbon"]

    def test_get_class(self):
        ast = AtomStyle("lj", [])
        ast.def_type("C1", class_="carbon")
        ast.def_type("C2", class_="carbon")
        ast.def_type("N", class_="nitrogen")
        
        carbon_types = ast.get_class("carbon")
        assert "C1" in carbon_types
        assert "C2" in carbon_types
        assert len(carbon_types) == 2


class TestForceField:
    def test_init(self):
        ff = ForceField("test_ff")
        assert ff.name == "test_ff"
        assert ff.unit == "real"
        assert ff.atomstyles == []
        assert ff.bondstyles == []
        assert ff.anglestyles == []
        assert ff.dihedralstyles == []
        assert ff.improperstyles == []
        assert ff.pairstyles == []

    def test_init_with_unit(self):
        ff = ForceField("test_ff", unit="metal")
        assert ff.unit == "metal"

    def test_repr(self):
        ff = ForceField("test_ff")
        assert repr(ff) == "<ForceField: test_ff>"

    def test_str_empty(self):
        ff = ForceField("test_ff")
        assert str(ff) == "<ForceField: test_ff>"

    def test_str_with_styles(self):
        ff = ForceField("test_ff")
        ff.def_atomstyle("lj")
        ff.def_bondstyle("harmonic")
        str_repr = str(ff)
        assert "n_atomstyles: 1" in str_repr
        assert "n_bondstyles: 1" in str_repr

    def test_properties(self):
        ff = ForceField("test_ff")
        assert ff.n_atomstyles == 0
        assert ff.n_bondstyles == 0
        assert ff.n_anglestyles == 0
        assert ff.n_dihedralstyles == 0
        assert ff.n_improperstyles == 0
        assert ff.n_pairstyles == 0
        assert ff.n_atomtypes == 0
        assert ff.n_bondtypes == 0
        assert ff.n_angletypes == 0
        assert ff.n_dihedraltypes == 0
        assert ff.n_impropertypes == 0
        assert ff.n_pairtypes == 0

    def test_def_and_get_styles(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("lj")
        bstyle = ff.def_bondstyle("harmonic")
        angstyle = ff.def_anglestyle("harmonic")
        dstyle = ff.def_dihedralstyle("harmonic")
        impstyle = ff.def_improperstyle("harmonic")
        pstyle = ff.def_pairstyle("lj")
        
        assert ff.get_atomstyle("lj") == astyle
        assert ff.get_bondstyle("harmonic") == bstyle
        assert ff.get_anglestyle("harmonic") == angstyle
        assert ff.get_dihedralstyle("harmonic") == dstyle
        assert ff.get_improperstyle("harmonic") == impstyle
        assert ff.get_pairstyle("lj") == pstyle

    def test_def_style_reuse(self):
        ff = ForceField()
        astyle1 = ff.def_atomstyle("lj")
        astyle2 = ff.def_atomstyle("lj")  # Should return the same style
        assert astyle1 is astyle2

    def test_get_types(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("lj")
        bstyle = ff.def_bondstyle("harmonic")
        angstyle = ff.def_anglestyle("harmonic")
        dstyle = ff.def_dihedralstyle("harmonic")
        impstyle = ff.def_improperstyle("harmonic")
        pstyle = ff.def_pairstyle("lj")
        
        at = astyle.def_type("C")
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
        assert pt in ff.get_pairtypes()

    def test_contains_and_getitem(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("lj")
        bstyle = ff.def_bondstyle("harmonic")
        
        assert "lj" in ff
        assert "harmonic" in ff
        assert "nonexistent" not in ff
        
        assert ff["lj"] == astyle
        assert ff["harmonic"] == bstyle
        
        with pytest.raises(KeyError):
            _ = ff["nonexistent"]

    def test_len(self):
        ff = ForceField()
        assert len(ff) == 0
        
        ff.def_atomstyle("lj")
        assert len(ff) == 1
        
        ff.def_bondstyle("harmonic")
        assert len(ff) == 2
        
        ff.def_anglestyle("harmonic")
        ff.def_dihedralstyle("harmonic")
        ff.def_improperstyle("harmonic")
        ff.def_pairstyle("lj")
        assert len(ff) == 6

    def test_merge(self):
        ff1 = ForceField("ff1")
        ff2 = ForceField("ff2")
        
        ff1.def_atomstyle("lj")
        ff2.def_atomstyle("gauss")
        ff2.def_bondstyle("harmonic")
        
        result = ff1.merge(ff2)
        assert result is ff1  # Returns self
        assert ff1.get_atomstyle("lj") is not None
        assert ff1.get_atomstyle("gauss") is not None
        assert ff1.get_bondstyle("harmonic") is not None

    def test_merge_no_overwrite(self):
        ff1 = ForceField("ff1")
        ff2 = ForceField("ff2")
        
        astyle1 = ff1.def_atomstyle("lj")
        astyle2 = ff2.def_atomstyle("lj")  # Same name, different instance
        
        ff1.merge(ff2)
        assert ff1.get_atomstyle("lj") is astyle1  # Should keep original

    def test_from_forcefields(self):
        ff1 = ForceField("ff1")
        ff2 = ForceField("ff2")
        
        ff1.def_atomstyle("lj")
        ff2.def_bondstyle("harmonic")
        
        combined = ForceField.from_forcefields("combined", ff1, ff2)
        assert combined.name == "combined"
        assert combined.get_atomstyle("lj") is not None
        assert combined.get_bondstyle("harmonic") is not None

    def test_merge_alias(self):
        ff1 = ForceField("ff1")
        ff2 = ForceField("ff2")
        
        ff2.def_atomstyle("lj")
        
        result = ff1.merge_(ff2)
        assert result is ff1
        assert ff1.get_atomstyle("lj") is not None
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
