"""
Unit tests for forcefield core functionality.

molpy's old parallel ForceField hierarchy is DELETED; ``molpy.core.forcefield``
now re-exports molrs's native classes. These tests exercise that molrs-backed
surface through molpy's public re-exports.

Key facts about the molrs-backed API (verified against the installed wheel):

* Types are NEVER constructed standalone. They are always created through a
  style: ``atomtype = atomstyle.def_type("CA", mass=12.0)`` and bond/angle/etc.
  types take previously-defined AtomType handles as endpoints.
* ``Parameters`` is keyword-only: ``Parameters({...})``; ``.args`` is always
  ``[]`` and ``.kwargs`` is the supplied mapping.
* ``style.types`` / ``ff.styles`` are plain lists; query via
  ``style.get_types(cls)`` / ``ff.get_styles(cls)`` / ``ff.get_types(cls)``.
* ``ff.to_potentials()`` with no frame is a *deferred* ``Potentials``
  (``len()==0``, not iterable). To evaluate you must pass a typed ``Frame``.
"""

import numpy as np
import pytest

import molrs

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
)


class TestParameters:
    """Test Parameters class functionality (molrs, keyword-only)."""

    def test_parameters_creation(self):
        """Parameters takes a mapping; args is always empty, kwargs is it."""
        params = Parameters({"k": 3.0, "theta": 4.0})
        assert params.args == []
        assert params.kwargs == {"k": 3.0, "theta": 4.0}

    def test_parameters_getitem(self):
        """Parameters supports keyword item access."""
        params = Parameters({"k": 2.0})
        assert params["k"] == 2.0
        assert "k" in params

    def test_parameters_repr(self):
        """Test parameters string representation."""
        params = Parameters({"k": 2.0})
        assert "Parameters" in repr(params)
        assert "kwargs" in repr(params)


class TestType:
    """Test the Type surface via styles (standalone construction is gone)."""

    def test_type_creation(self):
        """An atom type carries name and keyword params."""
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        t = astyle.def_type("test_type", sigma=0.3, epsilon=0.5)
        assert t.name == "test_type"
        assert t.params.kwargs["sigma"] == 0.3
        assert t.params.kwargs["epsilon"] == 0.5
        assert isinstance(t, Type)

    def test_type_equality(self):
        """Types compare equal by category + name."""
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        t1 = astyle.def_type("type1")
        t1_again = astyle.get_type_by_name("type1", AtomType)
        t3 = astyle.def_type("type2")
        assert t1 == t1_again
        assert t1 != t3

    def test_type_hashing(self):
        """Types can be hashed and used in sets; equal types collapse."""
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        t1 = astyle.def_type("type1")
        t1_again = astyle.get_type_by_name("type1", AtomType)
        t3 = astyle.def_type("type2")
        type_set = {t1, t1_again, t3}
        assert len(type_set) == 2

    def test_type_param_access(self):
        """Types support dict-style param access ``t['k']``."""
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        t = astyle.def_type("CA", mass=12.011)
        assert t["mass"] == 12.011


class TestAtomType:
    """Test AtomType created through an atom style."""

    def test_atomtype_creation(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at = astyle.def_type("CA", element="C", mass=12.011)
        assert at.name == "CA"
        assert at.params.kwargs["element"] == "C"
        assert at.params.kwargs["mass"] == 12.011

    def test_atomtype_is_type(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at = astyle.def_type("CA")
        assert isinstance(at, Type)
        assert isinstance(at, AtomType)

    def test_atomtype_equality(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", mass=12.011)
        at1_again = astyle.get_type_by_name("CA", AtomType)
        at3 = astyle.def_type("CB", mass=12.011)
        assert at1 == at1_again
        assert at1 != at3


class TestBondType:
    """Test BondType created through a bond style with AtomType endpoints."""

    def test_bondtype_creation(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        bt = bstyle.def_type(at1, at2, name="CA-CB", k=1000.0, r0=1.5)
        assert bt.name == "CA-CB"
        assert bt.itom.name == at1.name
        assert bt.jtom.name == at2.name
        assert bt.params.kwargs["k"] == 1000.0
        assert bt.params.kwargs["r0"] == 1.5

    def test_bondtype_auto_name(self):
        """Endpoint atom-type names are joined to form the bond type name."""
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        bt = bstyle.def_type(at1, at2, k=1000.0, r0=1.5)
        assert bt.name == "CA-CB"


class TestAngleType:
    """Test AngleType created through an angle style."""

    def test_angletype_creation(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        at3 = astyle.def_type("CC")
        anglestyle = ff.def_anglestyle("harmonic")
        angle = anglestyle.def_type(
            at1, at2, at3, name="CA-CB-CC", k=500.0, theta0=120.0
        )
        assert angle.name == "CA-CB-CC"
        assert angle.itom.name == at1.name
        assert angle.jtom.name == at2.name
        assert angle.ktom.name == at3.name
        assert angle.params.kwargs["k"] == 500.0
        assert angle.params.kwargs["theta0"] == 120.0

    def test_angletype_auto_name(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        at3 = astyle.def_type("CC")
        anglestyle = ff.def_anglestyle("harmonic")
        angle = anglestyle.def_type(at1, at2, at3, k=500.0)
        assert angle.name == "CA-CB-CC"


class TestStyle:
    """Test the Style surface through a ForceField."""

    def test_style_creation(self):
        """A bound style exposes its name and category."""
        ff = ForceField()
        style = ff.def_bondstyle("harmonic")
        assert style.name == "harmonic"
        assert style.category == "bond"

    def test_style_equality(self):
        """Styles compare equal by category + name."""
        ff = ForceField()
        s1 = ff.def_bondstyle("harmonic")
        s1_again = ff.get_style("bond", "harmonic")
        s3 = ff.def_bondstyle("morse")
        assert s1 == s1_again
        assert s1 != s3

    def test_style_def_type_chaining(self):
        """def_type adds to the style and is queryable via get_types."""
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        astyle.def_type("CA")
        astyle.def_type("CB")
        assert len(astyle.get_types(AtomType)) == 2


class TestAtomStyle:
    """Test AtomStyle.def_type."""

    def test_atomstyle_def_type(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at = astyle.def_type("CA", element="C", mass=12.011)
        assert isinstance(at, AtomType)
        assert at.name == "CA"
        assert len(astyle.get_types(AtomType)) == 1


class TestBondStyle:
    """Test BondStyle.def_type."""

    def test_bondstyle_def_type(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        bt = bstyle.def_type(at1, at2, name="CA-CB", k=1000.0, r0=1.5)
        assert isinstance(bt, BondType)
        assert bt.name == "CA-CB"
        assert len(bstyle.get_types(BondType)) == 1

    def test_bondstyle_def_type_auto_name(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        bt = bstyle.def_type(at1, at2, k=1000.0, r0=1.5)
        assert bt.name == "CA-CB"


class TestAngleStyle:
    """Test AngleStyle.def_type."""

    def test_anglestyle_def_type(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        at3 = astyle.def_type("CC")
        anglestyle = ff.def_anglestyle("harmonic")
        angle = anglestyle.def_type(
            at1, at2, at3, name="CA-CB-CC", k=500.0, theta0=120.0
        )
        assert isinstance(angle, AngleType)
        assert angle.name == "CA-CB-CC"

    def test_anglestyle_def_type_auto_name(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        at3 = astyle.def_type("CC")
        anglestyle = ff.def_anglestyle("harmonic")
        angle = anglestyle.def_type(at1, at2, at3, k=500.0)
        assert angle.name == "CA-CB-CC"


class TestForceField:
    """Test ForceField base class functionality."""

    def test_forcefield_creation(self):
        ff = ForceField(name="TestFF", units="real")
        assert ff.name == "TestFF"
        assert ff.units == "real"

    def test_forcefield_def_style(self):
        """def_style registers an unbound style and returns a bound handle."""
        ff = ForceField()
        style = ff.def_style(molrs.BondHarmonicStyle())
        assert isinstance(style, Style)
        assert style.name == "harmonic"
        assert style.category == "bond"

    def test_forcefield_def_style_idempotent(self):
        """Re-registering the same (category, name) returns an equal handle."""
        ff = ForceField()
        style1 = ff.def_style(molrs.BondHarmonicStyle())
        style2 = ff.def_style(molrs.BondHarmonicStyle())
        assert style1 == style2
        assert len(ff.get_styles(Style)) == 1

    def test_forcefield_get_styles(self):
        ff = ForceField()
        ff.def_atomstyle("full")
        ff.def_bondstyle("harmonic")

        atom_styles = ff.get_styles(AtomStyle)
        bond_styles = ff.get_styles(BondStyle)
        all_styles = ff.get_styles(Style)

        assert len(atom_styles) == 1
        assert len(bond_styles) == 1
        assert len(all_styles) == 2

    def test_forcefield_get_types(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", mass=12.011)
        at2 = astyle.def_type("CB", mass=12.011)

        atom_types = ff.get_types(AtomType)
        assert len(atom_types) == 2
        assert at1 in atom_types
        assert at2 in atom_types

    def test_forcefield_merge(self):
        ff1 = ForceField(name="FF1")
        ff2 = ForceField(name="FF2")

        astyle1 = ff1.def_atomstyle("full")
        astyle1.def_type("CA", mass=12.011)

        astyle2 = ff2.def_atomstyle("full")
        astyle2.def_type("CB", mass=12.011)

        ff1.merge(ff2)
        atom_types = ff1.get_types(AtomType)
        assert len(atom_types) == 2

    def test_forcefield_get_style(self):
        """get_style(category, name) returns the matching style or None."""
        ff = ForceField()
        ff.def_atomstyle("full")
        ff.def_bondstyle("harmonic")

        atom_style = ff.get_style("atom", "full")
        assert atom_style is not None
        assert isinstance(atom_style, AtomStyle)
        assert atom_style.name == "full"

        bond_style = ff.get_style("bond", "harmonic")
        assert bond_style is not None
        assert isinstance(bond_style, BondStyle)
        assert bond_style.name == "harmonic"

        assert ff.get_style("atom", "nonexistent") is None


class TestAtomisticForcefield:
    """``AtomisticForcefield`` is an alias of the native ``ForceField``."""

    def test_atomistic_ff_is_forcefield(self):
        assert AtomisticForcefield is ForceField

    def test_atomistic_ff_creation(self):
        ff = AtomisticForcefield(name="OPLS-AA", units="real")
        assert ff.name == "OPLS-AA"
        assert ff.units == "real"

    def test_def_atomstyle(self):
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        assert isinstance(astyle, AtomStyle)
        assert astyle.name == "full"

    def test_def_bondstyle(self):
        ff = AtomisticForcefield()
        bstyle = ff.def_bondstyle("harmonic")
        assert isinstance(bstyle, BondStyle)
        assert bstyle.name == "harmonic"

    def test_def_anglestyle(self):
        ff = AtomisticForcefield()
        astyle = ff.def_anglestyle("harmonic")
        assert isinstance(astyle, AngleStyle)
        assert astyle.name == "harmonic"

    def test_def_dihedralstyle(self):
        ff = AtomisticForcefield()
        dstyle = ff.def_dihedralstyle("opls")
        assert isinstance(dstyle, DihedralStyle)
        assert dstyle.name == "opls"

    def test_def_improperstyle(self):
        ff = AtomisticForcefield()
        istyle = ff.def_improperstyle("cvff")
        assert isinstance(istyle, ImproperStyle)
        assert istyle.name == "cvff"

    def test_def_pairstyle(self):
        ff = AtomisticForcefield()
        pstyle = ff.def_pairstyle("lj/cut")
        assert isinstance(pstyle, PairStyle)
        assert pstyle.name == "lj/cut"

    def test_get_atomtypes(self):
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", mass=12.011)
        at2 = astyle.def_type("CB", mass=12.011)

        atom_types = ff.get_types(AtomType)
        assert len(atom_types) == 2
        assert at1 in atom_types
        assert at2 in atom_types

    def test_get_bondtypes(self):
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        bt = bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        bond_types = ff.get_types(BondType)
        assert len(bond_types) == 1
        assert bt in bond_types

    def test_get_angletypes(self):
        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        at3 = astyle.def_type("CC")
        anglestyle = ff.def_anglestyle("harmonic")
        angle = anglestyle.def_type(at1, at2, at3, k=500.0, theta0=120.0)

        angle_types = ff.get_types(AngleType)
        assert len(angle_types) == 1
        assert angle in angle_types

    def test_complete_workflow(self):
        """Build a full force field through styles and verify type queries."""
        ff = AtomisticForcefield(name="TestFF", units="real")

        astyle = ff.def_atomstyle("full")
        ca = astyle.def_type("CA", element="C", mass=12.011, sigma=0.355, epsilon=0.293)
        ha = astyle.def_type("HA", element="H", mass=1.008, sigma=0.242, epsilon=0.126)

        bstyle = ff.def_bondstyle("harmonic")
        ca_ha_bond = bstyle.def_type(ca, ha, k=1000.0, r0=1.08)

        anglestyle = ff.def_anglestyle("harmonic")
        ha_ca_ha_angle = anglestyle.def_type(ha, ca, ha, k=500.0, theta0=120.0)

        assert len(ff.get_types(AtomType)) == 2
        assert len(ff.get_types(BondType)) == 1
        assert len(ff.get_types(AngleType)) == 1
        assert ca in ff.get_types(AtomType)
        assert ha in ff.get_types(AtomType)
        assert ca_ha_bond in ff.get_types(BondType)
        assert ha_ca_ha_angle in ff.get_types(AngleType)


def _build_bonded_frame(bond_type_name: str) -> molrs.Frame:
    """Build a minimal two-atom, one-bond frame for potential evaluation."""
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, 2.0]))
    atoms.insert("y", np.array([0.0, 0.0]))
    atoms.insert("z", np.array([0.0, 0.0]))
    frame["atoms"] = atoms

    bonds = molrs.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint32))
    bonds.insert("atomj", np.array([1], dtype=np.uint32))
    bonds.insert("type", np.array([bond_type_name], dtype=str))
    frame["bonds"] = bonds
    return frame


class TestForceFieldToPotentials:
    """Test ``ForceField.to_potentials`` (deferred + frame-evaluated)."""

    def test_to_potentials_no_frame_is_deferred(self):
        """Without a frame, to_potentials returns a deferred Potentials.

        ``len() == 0`` and iteration raises (it is not iterable).
        """
        ff = AtomisticForcefield(name="TestFF")
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CT")
        at2 = astyle.def_type("CT2")
        bstyle = ff.def_bondstyle("harmonic")
        bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        potentials = ff.to_potentials()
        assert len(potentials) == 0
        with pytest.raises(TypeError):
            list(potentials)

    def test_to_potentials_evaluates_harmonic_bond(self):
        """A frame-bound harmonic bond yields the closed-form energy.

        Two atoms 2.0 apart, k=300, r0=1.5 -> 0.5*k*(r-r0)^2 = 37.5.
        """
        ff = molrs.ForceField("bond-only")
        ff.def_bondtype("harmonic", "CT", "CT", {"k": 300.0, "r0": 1.5})

        frame = _build_bonded_frame("CT-CT")
        pots = ff.to_potentials(frame)
        assert len(pots) == 1

        energy = pots.calc_energy(molrs.extract_coords(frame))
        assert energy == pytest.approx(37.5)

    def test_to_potentials_raises_on_missing_params(self):
        """A defined bond type missing a required param raises at eval time.

        molrs raises ``ValueError`` (message names the missing param) when the
        frame-bound potential is built from a type lacking ``r0``.
        """
        ff = AtomisticForcefield(name="TestFF")
        astyle = ff.def_atomstyle("full")
        ca = astyle.def_type("CA")
        cb = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        # Omit the required r0 parameter.
        bstyle.def_type(ca, cb, k=1000.0)

        frame = _build_bonded_frame("CA-CB")
        with pytest.raises(ValueError, match="r0"):
            ff.to_potentials(frame)


class TestStyleGetTypes:
    """Test ``Style.get_types`` / ``Style.get_type_by_name``."""

    def test_get_types(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", mass=12.011)
        at2 = astyle.def_type("CB", mass=12.011)

        atom_types = astyle.get_types(AtomType)
        assert len(atom_types) == 2
        assert at1 in atom_types
        assert at2 in atom_types

    def test_get_types_empty(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        assert len(astyle.get_types(AtomType)) == 0

    def test_get_types_bond(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA")
        at2 = astyle.def_type("CB")
        bstyle = ff.def_bondstyle("harmonic")
        bt = bstyle.def_type(at1, at2, k=1000.0, r0=1.5)

        bond_types = bstyle.get_types(BondType)
        assert len(bond_types) == 1
        assert bt in bond_types

    def test_get_type_by_name(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", mass=12.011)
        at2 = astyle.def_type("CB", mass=12.011)

        result = astyle.get_type_by_name("CA", AtomType)
        assert result is not None
        assert result == at1
        assert result.name == "CA"

        result = astyle.get_type_by_name("CB", AtomType)
        assert result is not None
        assert result == at2

    def test_get_type_by_name_not_found(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        astyle.def_type("CA", mass=12.011)
        assert astyle.get_type_by_name("nonexistent", AtomType) is None

    def test_get_type_by_name_default_class(self):
        ff = ForceField()
        astyle = ff.def_atomstyle("full")
        at = astyle.def_type("CA", mass=12.011)
        result = astyle.get_type_by_name("CA")
        assert result is not None
        assert result == at
