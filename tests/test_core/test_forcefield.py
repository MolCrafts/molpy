"""
Unit tests for forcefield core functionality.

Tests the ForceField, AtomisticForcefield, and related classes.
"""

import pytest

from molpy import (
    AngleStyle,
    AngleType,
    AtomisticForcefield,
    AtomStyle,
    AtomType,
    BondStyle,
    BondType,
    DihedralStyle,
    ForceField,
    ImproperStyle,
    PairStyle,
    Parameters,
    Style,
    Type,
    TypeBucket,
)


class TestParameters:
    """Test Parameters class functionality."""

    def test_parameters_creation(self):
        """Test creating parameters with args and kwargs."""
        params = Parameters(1.0, 2.0, k=3.0, theta=4.0)
        assert params.args == [1.0, 2.0]
        assert params.kwargs == {"k": 3.0, "theta": 4.0}

    def test_parameters_repr(self):
        """Test parameters string representation."""
        params = Parameters(1.0, k=2.0)
        assert "Parameters" in repr(params)
        assert "args" in repr(params)
        assert "kwargs" in repr(params)


class TestType:
    """Test Type base class functionality."""

    def test_type_creation(self):
        """Test creating a type with name and parameters."""
        t = Type("test_type", sigma=0.3, epsilon=0.5)
        assert t.name == "test_type"
        assert t.params.kwargs["sigma"] == 0.3
        assert t.params.kwargs["epsilon"] == 0.5

    def test_type_equality(self):
        """Test type equality based on class and name."""
        t1 = Type("type1")
        t2 = Type("type1")
        t3 = Type("type2")
        assert t1 == t2
        assert t1 != t3

    def test_type_hashing(self):
        """Test that types can be hashed and used in sets."""
        t1 = Type("type1")
        t2 = Type("type1")
        t3 = Type("type2")
        type_set = {t1, t2, t3}
        assert len(type_set) == 2

    def test_type_repr(self):
        """Test type string representation."""
        t = Type("test_type")
        assert "Type" in repr(t)
        assert "test_type" in repr(t)


class TestAtomType:
    """Test AtomType class functionality."""

    def test_atomtype_creation(self):
        """Test creating an atom type."""
        at = AtomType("CA", element="C", mass=12.011)
        assert at.name == "CA"
        assert at.params.kwargs["element"] == "C"
        assert at.params.kwargs["mass"] == 12.011

    def test_atomtype_is_type(self):
        """Test that AtomType is a subclass of Type."""
        at = AtomType("CA")
        assert isinstance(at, Type)

    def test_atomtype_equality(self):
        """Test atom type equality."""
        at1 = AtomType("CA", mass=12.011)
        at2 = AtomType("CA", mass=12.011)
        at3 = AtomType("CB", mass=12.011)
        assert at1 == at2
        assert at1 != at3


class TestBondType:
    """Test BondType class functionality."""

    def test_bondtype_creation(self):
        """Test creating a bond type."""
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bt = BondType("CA-CB", at1, at2, k=1000.0, r0=1.5)
        assert bt.name == "CA-CB"
        assert bt.itom == at1
        assert bt.jtom == at2
        assert bt.params.kwargs["k"] == 1000.0
        assert bt.params.kwargs["r0"] == 1.5


class TestAngleType:
    """Test AngleType class functionality."""

    def test_angletype_creation(self):
        """Test creating an angle type."""
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        at3 = AtomType("CC")
        angle = AngleType("CA-CB-CC", at1, at2, at3, k=500.0, theta0=120.0)
        assert angle.name == "CA-CB-CC"
        assert angle.itom == at1
        assert angle.jtom == at2
        assert angle.ktom == at3
        assert angle.params.kwargs["k"] == 500.0
        assert angle.params.kwargs["theta0"] == 120.0


class TestTypeBucket:
    """Test TypeBucket class functionality."""

    def test_typebucket_add_and_bucket(self):
        """Test adding types and retrieving them by class."""
        tb = TypeBucket[Type]()
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bt1 = BondType("CA-CB", at1, at2)

        tb.add(at1)
        tb.add(at2)
        tb.add(bt1)

        atom_types = tb.bucket(AtomType)
        bond_types = tb.bucket(BondType)
        all_types = tb.bucket(Type)

        # bucket() now returns Entities
        assert len(atom_types) == 2
        assert len(bond_types) == 1
        assert len(all_types) == 3

        # Verify they contain the right instances
        assert at1 in atom_types
        assert at2 in atom_types
        assert bt1 in bond_types

    def test_typebucket_remove(self):
        """Test removing types from bucket."""
        tb = TypeBucket[Type]()
        at = AtomType("CA")
        tb.add(at)
        assert len(tb.bucket(AtomType)) == 1
        tb.remove(at)
        assert len(tb.bucket(AtomType)) == 0

    def test_typebucket_classes(self):
        """Test getting all type classes in bucket."""
        tb = TypeBucket[Type]()
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        tb.add(at1)
        tb.add(BondType("CA-CB", at1, at2))
        classes = list(tb.classes())
        assert AtomType in classes
        assert BondType in classes


class TestStyle:
    """Test Style base class functionality."""

    def test_style_creation(self):
        """Test creating a style."""
        style = Style("harmonic", k_unit="kcal/mol")
        assert style.name == "harmonic"
        assert style.params.kwargs["k_unit"] == "kcal/mol"

    def test_style_equality(self):
        """Test style equality."""
        s1 = Style("harmonic")
        s2 = Style("harmonic")
        s3 = Style("morse")
        assert s1 == s2
        assert s1 != s3

    def test_style_merge(self):
        """Test merging styles."""
        s1 = Style("harmonic", k=1.0)
        s2 = Style("harmonic", r0=1.5)
        s1.types.add(AtomType("CA"))
        s2.types.add(AtomType("CB"))

        s1.merge(s2)
        assert s1.params.kwargs["k"] == 1.0
        assert s1.params.kwargs["r0"] == 1.5
        assert len(s1.types.bucket(AtomType)) == 2


class TestAtomStyle:
    """Test AtomStyle class functionality."""

    def test_atomstyle_def_type(self):
        """Test defining atom types in atom style."""
        astyle = AtomStyle("full")
        at = astyle.def_type("CA", element="C", mass=12.011)
        assert isinstance(at, AtomType)
        assert at.name == "CA"
        assert len(astyle.types.bucket(AtomType)) == 1


class TestBondStyle:
    """Test BondStyle class functionality."""

    def test_bondstyle_def_type(self):
        """Test defining bond types in bond style."""
        bstyle = BondStyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bt = bstyle.def_type(at1, at2, name="CA-CB", k=1000.0, r0=1.5)
        assert isinstance(bt, BondType)
        assert bt.name == "CA-CB"
        assert len(bstyle.types.bucket(BondType)) == 1

    def test_bondstyle_def_type_auto_name(self):
        """Test automatic naming of bond types."""
        bstyle = BondStyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bt = bstyle.def_type(at1, at2, k=1000.0, r0=1.5)
        assert bt.name == "CA-CB"


class TestAngleStyle:
    """Test AngleStyle class functionality."""

    def test_anglestyle_def_type(self):
        """Test defining angle types in angle style."""
        astyle = AngleStyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        at3 = AtomType("CC")
        angle = astyle.def_type(at1, at2, at3, name="CA-CB-CC", k=500.0, theta0=120.0)
        assert isinstance(angle, AngleType)
        assert angle.name == "CA-CB-CC"

    def test_anglestyle_def_type_auto_name(self):
        """Test automatic naming of angle types."""
        astyle = AngleStyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        at3 = AtomType("CC")
        angle = astyle.def_type(at1, at2, at3, k=500.0)
        assert angle.name == "CA-CB-CC"


class TestForceField:
    """Test ForceField base class functionality."""

    def test_forcefield_creation(self):
        """Test creating a force field."""
        ff = ForceField(name="TestFF", units="real")
        assert ff.name == "TestFF"
        assert ff.units == "real"

    def test_forcefield_def_style(self):
        """Test defining styles in force field."""
        ff = ForceField()
        style = ff.def_style(Style, "test_style", param1=1.0)
        assert isinstance(style, Style)
        assert style.name == "test_style"

    def test_forcefield_def_style_idempotent(self):
        """Test that defining the same style twice returns the same instance."""
        ff = ForceField()
        style1 = ff.def_style(Style, "test_style", param1=1.0)
        style2 = ff.def_style(Style, "test_style", param2=2.0)
        assert style1 is style2

    def test_forcefield_get_styles(self):
        """Test getting styles from force field."""
        ff = ForceField()
        ff.def_style(AtomStyle, "full")
        ff.def_style(BondStyle, "harmonic")

        atom_styles = ff.get_styles(AtomStyle)
        bond_styles = ff.get_styles(BondStyle)
        all_styles = ff.get_styles(Style)

        assert len(atom_styles) == 1
        assert len(bond_styles) == 1
        assert len(all_styles) == 2

    def test_forcefield_get_types(self):
        """Test getting types from force field."""
        ff = ForceField()
        astyle = ff.def_style(AtomStyle, "full")
        at1 = astyle.def_type("CA", mass=12.011)
        at2 = astyle.def_type("CB", mass=12.011)

        atom_types = ff.get_types(AtomType)
        assert len(atom_types) == 2
        assert at1 in atom_types
        assert at2 in atom_types

    def test_forcefield_merge(self):
        """Test merging force fields."""
        ff1 = ForceField(name="FF1")
        ff2 = ForceField(name="FF2")

        astyle1 = ff1.def_style(AtomStyle, "full")
        astyle1.def_type("CA", mass=12.011)

        astyle2 = ff2.def_style(AtomStyle, "full")
        astyle2.def_type("CB", mass=12.011)

        ff1.merge(ff2)
        atom_types = ff1.get_types(AtomType)
        assert len(atom_types) == 2


class TestAtomisticForcefield:
    """Test AtomisticForcefield class functionality."""

    def test_atomistic_ff_creation(self):
        """Test creating an atomistic force field."""
        ff = AtomisticForcefield(name="OPLS-AA", units="real")
        assert ff.name == "OPLS-AA"
        assert ff.units == "real"

    def test_def_atomstyle(self):
        """Test defining atom style."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        assert isinstance(astyle, AtomStyle)
        assert astyle.name == "full"

    def test_def_bondstyle(self):
        """Test defining bond style."""
        ff = AtomisticForcefield()
        bstyle = ff.def_bondstyle("harmonic")
        assert isinstance(bstyle, BondStyle)
        assert bstyle.name == "harmonic"

    def test_def_anglestyle(self):
        """Test defining angle style."""
        ff = AtomisticForcefield()
        astyle = ff.def_anglestyle("harmonic")
        assert isinstance(astyle, AngleStyle)
        assert astyle.name == "harmonic"

    def test_def_dihedralstyle(self):
        """Test defining dihedral style."""
        ff = AtomisticForcefield()
        dstyle = ff.def_dihedralstyle("opls")
        assert isinstance(dstyle, DihedralStyle)
        assert dstyle.name == "opls"

    def test_def_improperstyle(self):
        """Test defining improper style."""
        ff = AtomisticForcefield()
        istyle = ff.def_improperstyle("cvff")
        assert isinstance(istyle, ImproperStyle)
        assert istyle.name == "cvff"

    def test_def_pairstyle(self):
        """Test defining pair style."""
        ff = AtomisticForcefield()
        pstyle = ff.def_pairstyle("lj/cut")
        assert isinstance(pstyle, PairStyle)
        assert pstyle.name == "lj/cut"

    def test_get_atomtypes(self):
        """Test getting atom types from atomistic force field."""
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", mass=12.011)
        at2 = astyle.def_type("CB", mass=12.011)

        atom_types = ff.get_atomtypes()
        assert len(atom_types) == 2
        assert at1 in atom_types
        assert at2 in atom_types

    def test_get_bondtypes(self):
        """Test getting bond types from atomistic force field."""
        ff = AtomisticForcefield()
        bstyle = ff.def_bondstyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bt = bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        bond_types = ff.get_bondtypes()
        assert len(bond_types) == 1
        assert bt in bond_types

    def test_get_angletypes(self):
        """Test getting angle types from atomistic force field."""
        ff = AtomisticForcefield()
        astyle = ff.def_anglestyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        at3 = AtomType("CC")
        angle = astyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        angle_types = ff.get_angletypes()
        assert len(angle_types) == 1
        assert angle in angle_types

    def test_complete_workflow(self):
        """Test a complete workflow of building a force field."""
        ff = AtomisticForcefield(name="TestFF", units="real")

        # Define atom style and types
        astyle = ff.def_atomstyle("full")
        ca = astyle.def_type("CA", element="C", mass=12.011, sigma=0.355, epsilon=0.293)
        ha = astyle.def_type("HA", element="H", mass=1.008, sigma=0.242, epsilon=0.126)

        # Define bond style and types
        bstyle = ff.def_bondstyle("harmonic")
        ca_ha_bond = bstyle.def_type(ca, ha, k=1000.0, r0=1.08)

        # Define angle style and types
        anglestyle = ff.def_anglestyle("harmonic")
        ha_ca_ha_angle = anglestyle.def_type(ha, ca, ha, k=500.0, theta0=120.0)

        # Verify everything
        assert len(ff.get_atomtypes()) == 2
        assert len(ff.get_bondtypes()) == 1
        assert len(ff.get_angletypes()) == 1
        assert ca in ff.get_atomtypes()
        assert ha in ff.get_atomtypes()
        assert ca_ha_bond in ff.get_bondtypes()
        assert ha_ca_ha_angle in ff.get_angletypes()


class TestStyleToPotential:
    """Test converting Style to Potential instances."""

    def test_bondstyle_to_potential(self):
        """Test converting BondStyle to Potential."""
        ff = AtomisticForcefield(name="TestFF")
        bstyle = ff.def_bondstyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        # Convert to potential
        potential = bstyle.to_potential()

        # Verify potential is created
        assert potential is not None
        assert hasattr(potential, "calc_energy")
        assert hasattr(potential, "calc_forces")
        assert hasattr(potential, "k")
        assert hasattr(potential, "r0")

        # Verify parameters

        assert len(potential.k) == 1
        assert len(potential.r0) == 1
        assert potential.k[0] == 1000.0
        assert potential.r0[0] == 1.5

    def test_bondstyle_to_potential_multiple_types(self):
        """Test converting BondStyle with multiple types to Potential."""
        ff = AtomisticForcefield(name="TestFF")
        bstyle = ff.def_bondstyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        at3 = AtomType("CC")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.5)
        bstyle.def_type(at2, at3, k=800.0, r0=1.4)

        # Convert to potential
        potential = bstyle.to_potential()

        # Verify parameters (order may vary, so check both values are present)

        assert len(potential.k) == 2
        assert len(potential.r0) == 2
        assert set(potential.k.flatten()) == {1000.0, 800.0}
        assert set(potential.r0.flatten()) == {1.5, 1.4}

    def test_bondstyle_to_potential_missing_parameters(self):
        """Test that missing parameters raise ValueError."""
        ff = AtomisticForcefield(name="TestFF")
        bstyle = ff.def_bondstyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        # Missing r0 parameter
        bstyle.def_type(at1, at2, k=1000.0)

        # Should raise ValueError
        with pytest.raises(ValueError, match="missing required parameters"):
            bstyle.to_potential()

    def test_anglestyle_to_potential(self):
        """Test converting AngleStyle to Potential."""
        ff = AtomisticForcefield(name="TestFF")
        astyle = ff.def_anglestyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        at3 = AtomType("CC")
        astyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        # Convert to potential
        potential = astyle.to_potential()

        # Verify potential is created
        assert potential is not None
        assert hasattr(potential, "calc_energy")
        assert hasattr(potential, "calc_forces")
        assert hasattr(potential, "k")
        assert hasattr(potential, "theta0")

        # Verify parameters

        assert len(potential.k) == 1
        assert len(potential.theta0) == 1
        assert potential.k[0] == 500.0
        assert potential.theta0[0] == 120.0

    def test_pairstyle_to_potential(self):
        """Test converting PairStyle to Potential."""
        ff = AtomisticForcefield(name="TestFF")
        pstyle = ff.def_pairstyle("lj126/cut")
        at1 = AtomType("CA")
        pstyle.def_type(at1, epsilon=0.293, sigma=0.355)

        # Convert to potential
        potential = pstyle.to_potential()

        # Verify potential is created
        assert potential is not None
        assert hasattr(potential, "calc_energy")
        assert hasattr(potential, "calc_forces")
        assert hasattr(potential, "epsilon")
        assert hasattr(potential, "sigma")

        # Verify parameters

        assert len(potential.epsilon) == 1
        assert len(potential.sigma) == 1
        assert potential.epsilon[0] == 0.293
        assert potential.sigma[0] == 0.355

    def test_pairstyle_to_potential_multiple_types(self):
        """Test converting PairStyle with multiple types to Potential."""
        ff = AtomisticForcefield(name="TestFF")
        pstyle = ff.def_pairstyle("lj126/cut")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        pstyle.def_type(at1, epsilon=0.293, sigma=0.355)
        pstyle.def_type(at2, epsilon=0.126, sigma=0.242)

        # Convert to potential
        potential = pstyle.to_potential()

        # Verify parameters (order may vary, so check both values are present)

        assert len(potential.epsilon) == 2
        assert len(potential.sigma) == 2
        assert set(potential.epsilon.flatten()) == {0.293, 0.126}
        assert set(potential.sigma.flatten()) == {0.355, 0.242}

    def test_pairstyle_to_potential_missing_parameters(self):
        """Test that missing parameters raise ValueError."""
        ff = AtomisticForcefield(name="TestFF")
        pstyle = ff.def_pairstyle("lj126/cut")
        at1 = AtomType("CA")
        # Missing sigma parameter
        pstyle.def_type(at1, epsilon=0.293)

        # Should raise ValueError
        with pytest.raises(ValueError, match="missing required parameters"):
            pstyle.to_potential()


class TestForceFieldToPotentials:
    """Test converting ForceField to Potentials collection."""

    def test_forcefield_to_potentials(self):
        """Test converting ForceField to Potentials."""
        ff = AtomisticForcefield(name="TestFF")

        # Define bond style and types
        bstyle = ff.def_bondstyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        # Define angle style and types
        astyle = ff.def_anglestyle("harmonic")
        at3 = AtomType("CC")
        astyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        # Define pair style and types
        pstyle = ff.def_pairstyle("lj126/cut")
        pstyle.def_type(at1, epsilon=0.293, sigma=0.355)

        # Convert to potentials
        potentials = ff.to_potentials()

        # Verify potentials collection
        assert len(potentials) == 3
        assert all(hasattr(p, "calc_energy") for p in potentials)
        assert all(hasattr(p, "calc_forces") for p in potentials)

    def test_forcefield_to_potentials_empty_styles(self):
        """Test converting ForceField with no types to Potentials."""
        ff = AtomisticForcefield(name="TestFF")
        ff.def_bondstyle("harmonic")
        ff.def_anglestyle("harmonic")

        # Convert to potentials (should return empty collection)
        potentials = ff.to_potentials()

        # Should return empty collection since no types are defined
        assert len(potentials) == 0

    def test_forcefield_to_potentials_partial_types(self):
        """Test converting ForceField with some styles missing types."""
        ff = AtomisticForcefield(name="TestFF")

        # Define bond style with types
        bstyle = ff.def_bondstyle("harmonic")
        at1 = AtomType("CA")
        at2 = AtomType("CB")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        # Define angle style without types
        ff.def_anglestyle("harmonic")

        # Convert to potentials
        potentials = ff.to_potentials()

        # Should only have bond potential
        assert len(potentials) == 1
