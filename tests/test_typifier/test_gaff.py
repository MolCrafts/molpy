"""Tests for GAFF (Generalized Amber Force Field) typifier."""

import pytest

from molpy import Angle, Atom, Atomistic, Bond
from molpy.io.forcefield.xml import XMLForceFieldReader


@pytest.fixture(scope="module")
def gaff_ff():
    """Load GAFF force field from XML."""
    reader = XMLForceFieldReader("src/molpy/data/forcefield/gaff.xml")
    return reader.read()


@pytest.fixture(scope="module")
def gaff_atom_typifier(gaff_ff):
    """Create a GaffAtomTypifier."""
    from molpy.typifier.gaff import GaffAtomTypifier

    return GaffAtomTypifier(gaff_ff, strict=False)


def _build_ethanol():
    """Build ethanol (CCO) with explicit hydrogens.

    Structure: H3C-CH2-OH
    Atom indices: C0, C1, O2, H3, H4, H5, H6, H7, H8
    """
    asm = Atomistic()

    c0 = Atom(element="C")  # methyl carbon
    c1 = Atom(element="C")  # methylene carbon
    o2 = Atom(element="O")  # hydroxyl oxygen

    h3 = Atom(element="H")  # on C0
    h4 = Atom(element="H")  # on C0
    h5 = Atom(element="H")  # on C0
    h6 = Atom(element="H")  # on C1
    h7 = Atom(element="H")  # on C1
    h8 = Atom(element="H")  # on O2 (hydroxyl)

    asm.add_entity(c0, c1, o2, h3, h4, h5, h6, h7, h8)

    # C-C bond
    asm.add_link(Bond(c0, c1))
    # C-O bond
    asm.add_link(Bond(c1, o2))
    # C0-H bonds
    asm.add_link(Bond(c0, h3))
    asm.add_link(Bond(c0, h4))
    asm.add_link(Bond(c0, h5))
    # C1-H bonds
    asm.add_link(Bond(c1, h6))
    asm.add_link(Bond(c1, h7))
    # O-H bond
    asm.add_link(Bond(o2, h8))

    return asm, {
        "c0": c0,
        "c1": c1,
        "o2": o2,
        "h3": h3,
        "h4": h4,
        "h5": h5,
        "h6": h6,
        "h7": h7,
        "h8": h8,
    }


def _build_benzene():
    """Build benzene (c1ccccc1) with explicit hydrogens.

    6 aromatic carbons + 6 hydrogens.
    """
    asm = Atomistic()

    carbons = []
    hydrogens = []
    for i in range(6):
        c = Atom(element="C", is_aromatic=True)
        h = Atom(element="H")
        carbons.append(c)
        hydrogens.append(h)
        asm.add_entity(c, h)

    # Ring bonds (aromatic)
    for i in range(6):
        bond = Bond(carbons[i], carbons[(i + 1) % 6])
        bond.data["order"] = 1.5
        bond.data["kind"] = ":"
        asm.add_link(bond)

    # C-H bonds
    for i in range(6):
        asm.add_link(Bond(carbons[i], hydrogens[i]))

    return asm, {"carbons": carbons, "hydrogens": hydrogens}


class TestGaffAtomTypifier:
    """Test GAFF atom type assignment."""

    def test_init(self, gaff_atom_typifier):
        """Test GaffAtomTypifier initializes with patterns."""
        assert len(gaff_atom_typifier.pattern_dict) > 100

    def test_ethanol_carbon_types(self, gaff_atom_typifier):
        """Test that ethanol carbons are typed as c3."""
        asm, _ = _build_ethanol()
        typed = gaff_atom_typifier.typify(asm)
        atoms = list(typed.atoms)

        # Both carbons should be sp3 -> c3
        assert atoms[0].get("type") == "c3"
        assert atoms[1].get("type") == "c3"

    def test_ethanol_oxygen_type(self, gaff_atom_typifier):
        """Test that ethanol oxygen is typed as oh."""
        asm, _ = _build_ethanol()
        typed = gaff_atom_typifier.typify(asm)
        atoms = list(typed.atoms)

        assert atoms[2].get("type") == "oh"

    def test_ethanol_hydroxyl_hydrogen(self, gaff_atom_typifier):
        """Test that hydroxyl hydrogen is typed as ho."""
        asm, _ = _build_ethanol()
        typed = gaff_atom_typifier.typify(asm)
        atoms = list(typed.atoms)

        assert atoms[8].get("type") == "ho"

    def test_ethanol_aliphatic_hydrogens(self, gaff_atom_typifier):
        """Test that aliphatic hydrogens are typed as hc or h1."""
        asm, _ = _build_ethanol()
        typed = gaff_atom_typifier.typify(asm)
        atoms = list(typed.atoms)

        # Methyl H (on C not bonded to electronegative group)
        h3_type = atoms[3].get("type")
        assert h3_type == "hc", f"Expected hc, got {h3_type}"

        # H on C bonded to O -> h1 (one electron-withdrawing group)
        h6_type = atoms[6].get("type")
        assert h6_type == "h1", f"Expected h1, got {h6_type}"

    def test_benzene_carbon_types(self, gaff_atom_typifier):
        """Test that benzene carbons are typed as ca."""
        asm, _ = _build_benzene()
        typed = gaff_atom_typifier.typify(asm)

        carbons = [a for a in typed.atoms if a.get("element") == "C"]
        assert len(carbons) == 6
        for a in carbons:
            assert a.get("type") == "ca", f"Expected ca, got {a.get('type')}"

    def test_benzene_hydrogen_types(self, gaff_atom_typifier):
        """Test that benzene hydrogens are typed as ha."""
        asm, _ = _build_benzene()
        typed = gaff_atom_typifier.typify(asm)

        hydrogens = [a for a in typed.atoms if a.get("element") == "H"]
        assert len(hydrogens) == 6
        for a in hydrogens:
            assert a.get("type") == "ha", f"Expected ha, got {a.get('type')}"


class TestGaffAtomisticTypifier:
    """Test GAFF full typing pipeline."""

    def test_init(self, gaff_ff):
        """Test GaffAtomisticTypifier initializes."""
        from molpy.typifier.gaff import GaffAtomisticTypifier

        typifier = GaffAtomisticTypifier(gaff_ff, strict_typing=False)
        assert typifier is not None

    def test_ethanol_bond_typing(self, gaff_ff):
        """Test bond parameter assignment for ethanol."""
        from molpy.typifier.gaff import GaffAtomisticTypifier

        typifier = GaffAtomisticTypifier(
            gaff_ff,
            skip_pair_typing=True,
            skip_angle_typing=True,
            skip_dihedral_typing=True,
            strict_typing=False,
        )

        asm, _ = _build_ethanol()
        typed = typifier.typify(asm)

        # Check that bonds have types assigned
        typed_bonds = [b for b in typed.bonds if b.data.get("type") is not None]
        assert len(typed_bonds) > 0, "Expected some bonds to be typed"

    def test_ethanol_full_pipeline(self, gaff_ff):
        """Test full GAFF typing pipeline on ethanol."""
        from molpy.typifier.gaff import GaffAtomisticTypifier

        typifier = GaffAtomisticTypifier(gaff_ff, strict_typing=False)

        asm, _ = _build_ethanol()
        asm = asm.get_topo(gen_angle=True, gen_dihe=True)
        typed = typifier.typify(asm)

        # All atoms should have types
        for atom in typed.atoms:
            assert atom.get("type") is not None, f"Atom {atom.get('element')} not typed"


class TestGaffXmlLoading:
    """Test GAFF XML loading and structure."""

    def test_load_gaff_xml(self, gaff_ff):
        """Test that GAFF XML loads correctly."""
        assert gaff_ff.name == "GAFF"

    def test_atom_types_count(self, gaff_ff):
        """Test that expected number of atom types are loaded."""
        from molpy.core.forcefield import AtomType

        atom_types = gaff_ff.get_types(AtomType)
        assert len(atom_types) > 50  # GAFF has ~80 unique types

    def test_bond_types_count(self, gaff_ff):
        """Test that bond types are loaded."""
        from molpy.core.forcefield import BondType

        bond_types = gaff_ff.get_types(BondType)
        assert len(bond_types) > 700

    def test_angle_types_count(self, gaff_ff):
        """Test that angle types are loaded."""
        from molpy.core.forcefield import AngleType

        angle_types = gaff_ff.get_types(AngleType)
        assert len(angle_types) > 3000

    def test_dihedral_types_count(self, gaff_ff):
        """Test that periodic dihedral types are loaded."""
        from molpy.core.forcefield import DihedralType

        dihedral_types = gaff_ff.get_types(DihedralType)
        assert len(dihedral_types) > 500

    def test_improper_types_count(self, gaff_ff):
        """Test that improper types are loaded."""
        from molpy.core.forcefield import ImproperType

        improper_types = gaff_ff.get_types(ImproperType)
        assert len(improper_types) > 30

    def test_pair_types_count(self, gaff_ff):
        """Test that VDW pair types are loaded."""
        from molpy.core.forcefield import PairType

        pair_types = gaff_ff.get_types(PairType)
        assert len(pair_types) > 60
