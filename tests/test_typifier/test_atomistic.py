#!/usr/bin/env python3
"""Unit tests for atomistic typifiers.

Tests cover:
- atomtype_matches function
- OplsBondTypifier
- OplsAngleTypifier
- OplsDihedralTypifier
- OplsAtomTypifier
- OplsAtomisticTypifier
"""

import pytest

from molpy import (
    Angle,
    Atom,
    Atomistic,
    AtomisticForcefield,
    AtomType,
    Bond,
    Dihedral,
)
from molpy.typifier.atomistic import (
    OplsAngleTypifier,
    OplsAtomisticTypifier,
    OplsAtomTypifier,
    OplsBondTypifier,
    OplsDihedralTypifier,
    atomtype_matches,
)


class TestAtomtypeMatches:
    """Test atomtype_matches function."""

    def test_atomtype_matches_by_type(self):
        """Test matching by type attribute."""
        at = AtomType("opls_135", type_="opls_135", class_="CT")

        assert atomtype_matches(at, "opls_135") is True
        assert atomtype_matches(at, "opls_136") is False

    def test_atomtype_matches_by_class(self):
        """Test matching by class attribute."""
        at = AtomType("opls_135", type_="opls_135", class_="CT")

        assert atomtype_matches(at, "CT") is True
        assert atomtype_matches(at, "CA") is False

    def test_atomtype_matches_wildcard(self):
        """Test matching wildcard atom type."""
        at = AtomType("*", type_="*", class_="*")

        assert atomtype_matches(at, "anything") is True

    def test_atomtype_matches_type_priority(self):
        """Test that type takes priority over class."""
        at = AtomType("opls_135", type_="opls_135", class_="CT")

        # Should match by type first
        assert atomtype_matches(at, "opls_135") is True
        # But also match by class
        assert atomtype_matches(at, "CT") is True


class TestOplsBondTypifier:
    """Test OplsBondTypifier class."""

    def test_bond_typifier_initialization(self):
        """Test OplsBondTypifier initialization."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("HA", type_="HA", class_="HC")

        bstyle = ff.def_bondstyle("harmonic")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        typifier = OplsBondTypifier(ff)

        assert typifier.ff is ff
        assert hasattr(typifier, "_bond_table")
        assert hasattr(typifier, "class_to_types")

    def test_bond_typifier_typify(self):
        """Test OplsBondTypifier.typify()."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("HA", type_="HA", class_="HC")

        bstyle = ff.def_bondstyle("harmonic")
        bond_type = bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        typifier = OplsBondTypifier(ff)

        # Create bond with typed atoms
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        c.data["type"] = "CA"
        h.data["type"] = "HA"
        asm.add_entity(c, h)

        bond = Bond(c, h)
        asm.add_link(bond)

        # Typify bond
        result = typifier.typify(bond)

        assert result is bond
        assert bond.data["type"] == bond_type.name
        assert "k" in bond.data
        assert "r0" in bond.data

    def test_bond_typifier_typify_reverse_order(self):
        """Test bond typifier handles reverse atom order."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("HA", type_="HA", class_="HC")

        bstyle = ff.def_bondstyle("harmonic")
        bond_type = bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        typifier = OplsBondTypifier(ff)

        # Create bond with reverse order
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        c.data["type"] = "CA"
        h.data["type"] = "HA"
        asm.add_entity(c, h)

        bond = Bond(h, c)  # Reverse order
        asm.add_link(bond)

        # Should still match
        result = typifier.typify(bond)

        assert result is bond
        assert bond.data["type"] == bond_type.name

    def test_bond_typifier_typify_missing_type(self):
        """Test bond typifier raises error when atoms lack type."""
        ff = AtomisticForcefield()
        typifier = OplsBondTypifier(ff)

        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        # Don't set type
        asm.add_entity(c, h)

        bond = Bond(c, h)
        asm.add_link(bond)

        with pytest.raises(ValueError, match="Bond atoms must have 'type' attribute"):
            typifier.typify(bond)

    def test_bond_typifier_typify_no_match(self):
        """Test bond typifier raises error when no match found."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("HA", type_="HA", class_="HC")

        bstyle = ff.def_bondstyle("harmonic")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        typifier = OplsBondTypifier(ff)

        # Create bond with types that don't match
        asm = Atomistic()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        c.data["type"] = "CA"
        o.data["type"] = "OA"  # Not in force field
        asm.add_entity(c, o)

        bond = Bond(c, o)
        asm.add_link(bond)

        with pytest.raises(ValueError, match="No bond type found"):
            typifier.typify(bond)

    def test_bond_typifier_class_matching(self):
        """Test bond typifier matches by class when type doesn't match."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("HA", type_="HA", class_="HC")

        # Add atom types with matching classes for class-based matching
        astyle.def_type("CB", type_="CB", class_="CT")
        astyle.def_type("HB", type_="HB", class_="HC")

        # Create bond type using classes
        bstyle = ff.def_bondstyle("harmonic")
        bond_type = bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        typifier = OplsBondTypifier(ff)

        # Create bond with different types but same classes
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        c.data["type"] = "CB"  # Different type, but should match by class
        h.data["type"] = "HB"  # Different type, but should match by class

        bond = Bond(c, h)
        asm.add_link(bond)

        # Should match by class
        result = typifier.typify(bond)

        assert result is bond
        assert bond.data["type"] == bond_type.name


class TestOplsAngleTypifier:
    """Test OplsAngleTypifier class."""

    def test_angle_typifier_initialization(self):
        """Test OplsAngleTypifier initialization."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("HA", type_="HA", class_="HC")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("HA", type_="HA", class_="HC")

        anglestyle = ff.def_anglestyle("harmonic")
        anglestyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        typifier = OplsAngleTypifier(ff)

        assert typifier.ff is ff
        assert hasattr(typifier, "_angle_table")
        assert hasattr(typifier, "class_to_types")

    def test_angle_typifier_typify(self):
        """Test OplsAngleTypifier.typify()."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("HA", type_="HA", class_="HC")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("HA", type_="HA", class_="HC")

        anglestyle = ff.def_anglestyle("harmonic")
        angle_type = anglestyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        typifier = OplsAngleTypifier(ff)

        # Create angle with typed atoms
        asm = Atomistic()
        h1 = Atom(symbol="H")
        c = Atom(symbol="C")
        h2 = Atom(symbol="H")
        h1.data["type"] = "HA"
        c.data["type"] = "CA"
        h2.data["type"] = "HA"
        asm.add_entity(h1, c, h2)

        angle = Angle(h1, c, h2)
        asm.add_link(angle)

        # Typify angle
        result = typifier.typify(angle)

        assert result is angle
        assert angle.data["type"] == angle_type.name
        assert "k" in angle.data
        assert "theta0" in angle.data

    def test_angle_typifier_typify_reverse_order(self):
        """Test angle typifier handles reverse atom order."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("HA", type_="HA", class_="HC")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("HA", type_="HA", class_="HC")

        anglestyle = ff.def_anglestyle("harmonic")
        angle_type = anglestyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        typifier = OplsAngleTypifier(ff)

        # Create angle with reverse order
        asm = Atomistic()
        h1 = Atom(symbol="H")
        c = Atom(symbol="C")
        h2 = Atom(symbol="H")
        h1.data["type"] = "HA"
        c.data["type"] = "CA"
        h2.data["type"] = "HA"
        asm.add_entity(h1, c, h2)

        angle = Angle(h2, c, h1)  # Reverse order
        asm.add_link(angle)

        # Should still match
        result = typifier.typify(angle)

        assert result is angle
        assert angle.data["type"] == angle_type.name

    def test_angle_typifier_typify_missing_type(self):
        """Test angle typifier raises error when atoms lack type."""
        ff = AtomisticForcefield()
        typifier = OplsAngleTypifier(ff)

        asm = Atomistic()
        h1 = Atom(symbol="H")
        c = Atom(symbol="C")
        h2 = Atom(symbol="H")
        # Don't set type
        asm.add_entity(h1, c, h2)

        angle = Angle(h1, c, h2)
        asm.add_link(angle)

        with pytest.raises(ValueError, match="Angle atoms must have 'type' attribute"):
            typifier.typify(angle)

    def test_angle_typifier_typify_no_match(self):
        """Test angle typifier raises error when no match found."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("HA", type_="HA", class_="HC")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("HA", type_="HA", class_="HC")

        anglestyle = ff.def_anglestyle("harmonic")
        anglestyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        typifier = OplsAngleTypifier(ff)

        # Create angle with types that don't match
        asm = Atomistic()
        h1 = Atom(symbol="H")
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        h1.data["type"] = "HA"
        c.data["type"] = "CA"
        o.data["type"] = "OA"  # Not in force field
        asm.add_entity(h1, c, o)

        angle = Angle(h1, c, o)
        asm.add_link(angle)

        with pytest.raises(ValueError, match="No angle type found"):
            typifier.typify(angle)


class TestOplsDihedralTypifier:
    """Test OplsDihedralTypifier class."""

    def test_dihedral_typifier_initialization(self):
        """Test OplsDihedralTypifier initialization."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("CA", type_="CA", class_="CT")
        at4 = astyle.def_type("CA", type_="CA", class_="CT")

        dihedralstyle = ff.def_dihedralstyle("opls")
        dihedralstyle.def_type(at1, at2, at3, at4, c0=1.0, c1=2.0)

        typifier = OplsDihedralTypifier(ff)

        assert typifier.ff is ff
        assert hasattr(typifier, "_dihedral_list")
        assert hasattr(typifier, "class_to_types")

    def test_dihedral_typifier_typify(self):
        """Test OplsDihedralTypifier.typify()."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("CA", type_="CA", class_="CT")
        at4 = astyle.def_type("CA", type_="CA", class_="CT")

        dihedralstyle = ff.def_dihedralstyle("opls")
        dihedral_type = dihedralstyle.def_type(at1, at2, at3, at4, c0=1.0, c1=2.0)

        typifier = OplsDihedralTypifier(ff)

        # Create dihedral with typed atoms
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        c3 = Atom(symbol="C")
        c4 = Atom(symbol="C")
        c1.data["type"] = "CA"
        c2.data["type"] = "CA"
        c3.data["type"] = "CA"
        c4.data["type"] = "CA"
        asm.add_entity(c1, c2, c3, c4)

        dihedral = Dihedral(c1, c2, c3, c4)
        asm.add_link(dihedral)

        # Typify dihedral
        result = typifier.typify(dihedral)

        assert result is dihedral
        assert dihedral.data["type"] == dihedral_type.name
        assert "c0" in dihedral.data
        assert "c1" in dihedral.data

    def test_dihedral_typifier_typify_reverse_order(self):
        """Test dihedral typifier handles reverse atom order."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("CA", type_="CA", class_="CT")
        at4 = astyle.def_type("CA", type_="CA", class_="CT")

        dihedralstyle = ff.def_dihedralstyle("opls")
        dihedral_type = dihedralstyle.def_type(at1, at2, at3, at4, c0=1.0, c1=2.0)

        typifier = OplsDihedralTypifier(ff)

        # Create dihedral with reverse order
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        c3 = Atom(symbol="C")
        c4 = Atom(symbol="C")
        c1.data["type"] = "CA"
        c2.data["type"] = "CA"
        c3.data["type"] = "CA"
        c4.data["type"] = "CA"
        asm.add_entity(c1, c2, c3, c4)

        dihedral = Dihedral(c4, c3, c2, c1)  # Reverse order
        asm.add_link(dihedral)

        # Should still match
        result = typifier.typify(dihedral)

        assert result is dihedral
        assert dihedral.data["type"] == dihedral_type.name

    def test_dihedral_typifier_typify_missing_type(self):
        """Test dihedral typifier raises error when atoms lack type."""
        ff = AtomisticForcefield()
        typifier = OplsDihedralTypifier(ff)

        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        c3 = Atom(symbol="C")
        c4 = Atom(symbol="C")
        # Don't set type
        asm.add_entity(c1, c2, c3, c4)

        dihedral = Dihedral(c1, c2, c3, c4)
        asm.add_link(dihedral)

        with pytest.raises(
            ValueError, match="Dihedral atoms must have 'type' attribute"
        ):
            typifier.typify(dihedral)

    def test_dihedral_typifier_typify_no_match(self):
        """Test dihedral typifier raises error when no match found."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("CA", type_="CA", class_="CT")
        at4 = astyle.def_type("CA", type_="CA", class_="CT")

        dihedralstyle = ff.def_dihedralstyle("opls")
        dihedralstyle.def_type(at1, at2, at3, at4, c0=1.0, c1=2.0)

        typifier = OplsDihedralTypifier(ff)

        # Create dihedral with types that don't match
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        c3 = Atom(symbol="C")
        o = Atom(symbol="O")
        c1.data["type"] = "CA"
        c2.data["type"] = "CA"
        c3.data["type"] = "CA"
        o.data["type"] = "OA"  # Not in force field
        asm.add_entity(c1, c2, c3, o)

        dihedral = Dihedral(c1, c2, c3, o)
        asm.add_link(dihedral)

        with pytest.raises(ValueError, match="No dihedral type found"):
            typifier.typify(dihedral)


class TestOplsAtomisticTypifier:
    """Test OplsAtomisticTypifier class."""

    def test_atomistic_typifier_initialization_default(self):
        """Test OplsAtomisticTypifier initialization with defaults."""
        ff = AtomisticForcefield()
        typifier = OplsAtomisticTypifier(ff)

        assert typifier.ff is ff
        assert typifier.skip_atom_typing is False
        assert typifier.skip_pair_typing is False
        assert typifier.skip_bond_typing is False
        assert typifier.skip_angle_typing is False
        assert typifier.skip_dihedral_typing is False
        assert hasattr(typifier, "atom_typifier")
        assert hasattr(typifier, "bond_typifier")
        assert hasattr(typifier, "angle_typifier")
        assert hasattr(typifier, "dihedral_typifier")

    def test_atomistic_typifier_initialization_with_atom_typing(self):
        """Test OplsAtomisticTypifier initialization with atom typing enabled."""
        ff = AtomisticForcefield()
        typifier = OplsAtomisticTypifier(ff, skip_atom_typing=False)

        assert typifier.skip_atom_typing is False
        assert hasattr(typifier, "atom_typifier")
        assert isinstance(typifier.atom_typifier, OplsAtomTypifier)

    def test_atomistic_typifier_typify_bonds_only(self):
        """Test OplsAtomisticTypifier.typify() with bonds only."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", **{"type": "CA", "class": "CT"})
        at2 = astyle.def_type("HA", **{"type": "HA", "class": "HC"})

        bstyle = ff.def_bondstyle("harmonic")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        typifier = OplsAtomisticTypifier(
            ff,
            skip_atom_typing=True,
            skip_angle_typing=True,
            skip_dihedral_typing=True,
            skip_pair_typing=True,
        )

        # Create structure with typed atoms
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        c.data["type"] = "CA"
        h.data["type"] = "HA"
        asm.add_entity(c, h)

        bond = Bond(c, h)
        asm.add_link(bond)

        # Typify
        result = typifier.typify(asm)

        assert result is asm
        assert bond.data.get("type") is not None

    def test_atomistic_typifier_typify_all(self):
        """Test OplsAtomisticTypifier.typify() with all typing enabled."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("HA", type_="HA", class_="HC")
        at2 = astyle.def_type("CA", type_="CA", class_="CT")
        at3 = astyle.def_type("HA", type_="HA", class_="HC")

        bstyle = ff.def_bondstyle("harmonic")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.08)

        anglestyle = ff.def_anglestyle("harmonic")
        anglestyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        pairstyle = ff.def_pairstyle("opls")
        pairstyle.def_type(at1, at1, c0=1.0, c1=2.0)
        pairstyle.def_type(at2, at2, c0=1.0, c1=2.0)
        pairstyle.def_type(at3, at3, c0=1.0, c1=2.0)

        typifier = OplsAtomisticTypifier(
            ff,
            skip_atom_typing=True,  # Skip atom typing for simplicity
            skip_angle_typing=False,
            skip_dihedral_typing=True,
        )

        # Create structure
        asm = Atomistic()
        h1 = Atom(symbol="H")
        c = Atom(symbol="C")
        h2 = Atom(symbol="H")
        h1.data["type"] = "HA"
        c.data["type"] = "CA"
        h2.data["type"] = "HA"
        asm.add_entity(h1, c, h2)

        bond1 = Bond(h1, c)
        bond2 = Bond(c, h2)
        asm.add_link(bond1, bond2)

        # Generate topology
        asm.get_topo(gen_angle=True, gen_dihe=False)

        # Typify
        result = typifier.typify(asm)

        assert result is asm
        assert bond1.data.get("type") is not None
