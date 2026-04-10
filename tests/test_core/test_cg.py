"""
Test CoarseGrain class API for creating and managing coarse-grained structures.

Tests the CoarseGrain system including:
- Bead and CGBond entity/link classes
- Factory methods (def_*) and add methods (add_*)
- Batch operations
- Spatial transformations
- System composition
- Bidirectional conversion with Atomistic
"""

import numpy as np
import pytest

from molpy import Atomistic, Bead, CGBond, CoarseGrain


class TestBeadEntity:
    """Test Bead entity class."""

    def test_create_bead_with_attributes(self):
        """Test creating a bead with attributes."""
        bead = Bead(type="PEO", x=1.0, y=2.0, z=3.0)

        assert isinstance(bead, Bead)
        assert bead.get("type") == "PEO"
        assert bead.get("x") == 1.0
        assert bead.get("y") == 2.0
        assert bead.get("z") == 3.0

    def test_bead_repr_with_type(self):
        """Test bead string representation shows type."""
        bead = Bead(type="PEO")
        repr_str = repr(bead)

        assert "Bead" in repr_str
        assert "PEO" in repr_str

    def test_bead_repr_with_name(self):
        """Test bead string representation shows name if no type."""
        bead = Bead(name="BeadA")
        repr_str = repr(bead)

        assert "Bead" in repr_str
        assert "BeadA" in repr_str

    def test_bead_with_atomistic_mapping(self):
        """Test bead can store atomistic structure as member."""
        atomistic = Atomistic()
        atomistic.def_atom(symbol="C", x=0, y=0, z=0)

        bead = Bead(atomistic=atomistic, type="CH3")

        assert bead.atomistic is atomistic


class TestCGBondLink:
    """Test CGBond link class."""

    def test_create_cgbond(self):
        """Test creating a CGBond between two beads."""
        bead1 = Bead(type="A")
        bead2 = Bead(type="B")

        bond = CGBond(bead1, bead2, type="harmonic")

        assert isinstance(bond, CGBond)
        assert bond.ibead is bead1
        assert bond.jbead is bead2
        assert bond.get("type") == "harmonic"

    def test_cgbond_endpoint_properties(self):
        """Test CGBond ibead and jbead properties."""
        bead1 = Bead(type="A")
        bead2 = Bead(type="B")
        bond = CGBond(bead1, bead2)

        assert bond.ibead is bead1
        assert bond.jbead is bead2

    def test_cgbond_repr(self):
        """Test CGBond string representation."""
        bead1 = Bead(type="A")
        bead2 = Bead(type="B")
        bond = CGBond(bead1, bead2)

        repr_str = repr(bond)
        assert "CGBond" in repr_str

    def test_cgbond_requires_beads(self):
        """Test CGBond validates bead types."""
        bead = Bead(type="A")
        not_a_bead = object()

        with pytest.raises(AssertionError):
            CGBond(bead, not_a_bead)


class TestCoarseGrainFactoryMethods:
    """Test def_* factory methods that create and add entities."""

    def test_def_bead_creates_and_adds(self):
        """Test def_bead creates a Bead and adds it to the structure."""
        cg = CoarseGrain()
        bead = cg.def_bead(type="PEO", x=0, y=0, z=0)

        assert isinstance(bead, Bead)
        assert bead.get("type") == "PEO"
        assert len(cg.beads) == 1
        assert next(iter(cg.beads)) is bead

    def test_def_cgbond_creates_and_adds(self):
        """Test def_cgbond creates a CGBond between two beads."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")

        bond = cg.def_cgbond(b1, b2, type="harmonic")

        assert isinstance(bond, CGBond)
        assert bond.ibead is b1
        assert bond.jbead is b2
        assert bond.get("type") == "harmonic"
        assert len(cg.cgbonds) == 1


class TestCoarseGrainAddMethods:
    """Test add_* methods that add existing entity objects."""

    def test_add_bead_adds_existing(self):
        """Test add_bead adds an already created Bead object."""
        cg = CoarseGrain()
        bead = Bead(type="PEO", x=0, y=0, z=0)

        result = cg.add_bead(bead)

        assert result is bead
        assert len(cg.beads) == 1
        assert next(iter(cg.beads)) is bead

    def test_add_cgbond_adds_existing(self):
        """Test add_cgbond adds an already created CGBond object."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")
        bond = CGBond(b1, b2, type="harmonic")

        result = cg.add_cgbond(bond)

        assert result is bond
        assert len(cg.cgbonds) == 1
        assert next(iter(cg.cgbonds)) is bond


class TestCoarseGrainBatchMethods:
    """Test batch operations for creating and adding multiple entities."""

    def test_def_beads_batch_create(self):
        """Test def_beads creates multiple beads at once."""
        cg = CoarseGrain()

        beads = cg.def_beads(
            [
                {"type": "PEO", "x": 0, "y": 0, "z": 0},
                {"type": "PMA", "x": 5, "y": 0, "z": 0},
                {"type": "PEO", "x": 10, "y": 0, "z": 0},
            ]
        )

        assert len(beads) == 3
        assert all(isinstance(b, Bead) for b in beads)
        assert len(cg.beads) == 3
        assert np.array_equal(cg.beads["type"], ["PEO", "PMA", "PEO"])

    def test_def_cgbonds_batch_create(self):
        """Test def_cgbonds creates multiple bonds at once."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")
        b3 = cg.def_bead(type="C")

        bonds = cg.def_cgbonds(
            [
                (b1, b2, {"type": "harmonic"}),
                (b2, b3, {"type": "harmonic"}),
            ]
        )

        assert len(bonds) == 2
        assert all(isinstance(b, CGBond) for b in bonds)
        assert len(cg.cgbonds) == 2

    def test_add_beads_batch_add(self):
        """Test add_beads adds multiple existing Bead objects."""
        cg = CoarseGrain()
        beads = [
            Bead(type="A", x=0, y=0, z=0),
            Bead(type="B", x=5, y=0, z=0),
        ]

        result = cg.add_beads(beads)

        assert result == beads
        assert len(cg.beads) == 2

    def test_add_cgbonds_batch_add(self):
        """Test add_cgbonds adds multiple existing CGBond objects."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")
        b3 = cg.def_bead(type="C")

        bonds = [
            CGBond(b1, b2),
            CGBond(b2, b3),
        ]

        result = cg.add_cgbonds(bonds)

        assert result == bonds
        assert len(cg.cgbonds) == 2


class TestCoarseGrainSpatialOperations:
    """Test spatial transformation operations."""

    def test_move_translates_beads(self):
        """Test move translates all bead positions."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)
        cg.def_bead(type="B", x=1, y=0, z=0)

        result = cg.move([5, 10, 15])

        assert result is cg  # Returns self for chaining
        positions_x = cg.beads["x"]
        assert np.allclose(positions_x, [5, 6])

    def test_spatial_operations_chain(self):
        """Test spatial operations can be chained."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)

        # Chain multiple operations
        result = cg.move([1, 0, 0]).scale(2.0).move([0, 5, 0])

        assert result is cg
        # Verify position changed
        bead = list(cg.beads)[0]
        assert bead.get("x") != 0 or bead.get("y") != 0


class TestCoarseGrainSystemComposition:
    """Test system composition operations."""

    def test_iadd_merges_in_place(self):
        """Test += operator merges structures in place."""
        cg1 = CoarseGrain()
        cg1.def_bead(type="A", x=0, y=0, z=0)

        cg2 = CoarseGrain()
        cg2.def_bead(type="B", x=10, y=0, z=0)

        result = cg1.__iadd__(cg2)

        assert result is cg1
        assert len(cg1.beads) == 2

    def test_add_creates_new_structure(self):
        """Test + operator creates new merged structure."""
        cg1 = CoarseGrain()
        cg1.def_bead(type="A", x=0, y=0, z=0)

        cg2 = CoarseGrain()
        cg2.def_bead(type="B", x=10, y=0, z=0)

        cg3 = cg1 + cg2

        assert cg3 is not cg1
        assert cg3 is not cg2
        assert len(cg3.beads) == 2
        assert len(cg1.beads) == 1  # Original unchanged
        assert len(cg2.beads) == 1  # Original unchanged

    def test_replicate_creates_copies(self):
        """Test replicate creates multiple copies."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)

        result = cg.replicate(5, lambda mol, i: mol.move([i * 5, 0, 0]))

        assert len(result.beads) == 5
        assert len(cg.beads) == 1  # Original unchanged

    def test_len_returns_bead_count(self):
        """Test len() returns number of beads."""
        cg = CoarseGrain()
        assert len(cg) == 0

        cg.def_bead(type="A")
        assert len(cg) == 1

        cg.def_bead(type="B")
        assert len(cg) == 2

    def test_repr_shows_structure_summary(self):
        """Test repr shows bead and bond counts."""
        cg = CoarseGrain()
        cg.def_bead(type="PEO")
        cg.def_bead(type="PMA")

        repr_str = repr(cg)

        assert "CoarseGrain" in repr_str
        assert "2 beads" in repr_str


class TestToAtomisticConversion:
    """Test CoarseGrain to Atomistic conversion."""

    def test_convert_unmapped_beads_to_atoms(self):
        """Test beads without atomistic mapping create single atoms."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)
        cg.def_bead(type="B", x=5, y=0, z=0)

        atomistic = cg.to_atomistic()

        assert len(atomistic.atoms) == 2
        atoms = list(atomistic.atoms)
        assert atoms[0].get("element") == "A"
        assert atoms[1].get("element") == "B"

    def test_convert_mapped_beads_to_atomistic(self):
        """Test beads with atomistic mapping expand to full structure."""
        # Create atomistic structure for bead
        atom_struct = Atomistic()
        atom_struct.def_atom(symbol="C", x=0, y=0, z=0)
        atom_struct.def_atom(symbol="H", x=1, y=0, z=0)

        cg = CoarseGrain()
        bead = cg.def_bead(atomistic=atom_struct, type="CH2", x=0, y=0, z=0)

        atomistic = cg.to_atomistic()

        assert len(atomistic.atoms) == 2

    def test_convert_cgbonds_to_atomistic_bonds(self):
        """Test CGBonds are converted to atomistic bonds."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A", x=0, y=0, z=0)
        b2 = cg.def_bead(type="B", x=5, y=0, z=0)
        cg.def_cgbond(b1, b2)

        atomistic = cg.to_atomistic()

        assert len(atomistic.bonds) == 1


class TestFromAtomisticConversion:
    """Test Atomistic to CoarseGrain conversion."""

    def test_convert_atomistic_with_mask(self):
        """Test creating CG from atomistic with bead mask."""
        # Create simple atomistic structure
        atomistic = Atomistic()
        atomistic.def_atom(symbol="C", x=0, y=0, z=0)
        atomistic.def_atom(symbol="H", x=1, y=0, z=0)
        atomistic.def_atom(symbol="H", x=0, y=1, z=0)

        # All atoms in one bead
        mask = np.array([0, 0, 0])

        cg = CoarseGrain.from_atomistic(atomistic, mask)

        assert len(cg.beads) == 1
        bead = list(cg.beads)[0]
        assert bead.atomistic is not None

    def test_convert_creates_multiple_beads(self):
        """Test conversion creates multiple beads from mask."""
        atomistic = Atomistic()
        atomistic.def_atom(symbol="C", x=0, y=0, z=0)
        atomistic.def_atom(symbol="H", x=1, y=0, z=0)
        atomistic.def_atom(symbol="O", x=5, y=0, z=0)
        atomistic.def_atom(symbol="H", x=6, y=0, z=0)

        # Two beads: [0,1] and [2,3]
        mask = np.array([0, 0, 1, 1])

        cg = CoarseGrain.from_atomistic(atomistic, mask)

        assert len(cg.beads) == 2

    def test_bead_position_is_center_of_mass(self):
        """Test bead position is calculated as center of atoms."""
        atomistic = Atomistic()
        atomistic.def_atom(symbol="C", x=0, y=0, z=0)
        atomistic.def_atom(symbol="H", x=2, y=0, z=0)

        mask = np.array([0, 0])

        cg = CoarseGrain.from_atomistic(atomistic, mask)

        bead = list(cg.beads)[0]
        # Center should be at (1, 0, 0)
        assert np.isclose(bead.get("x"), 1.0)
        assert np.isclose(bead.get("y"), 0.0)
        assert np.isclose(bead.get("z"), 0.0)

    def test_infer_cgbonds_from_atomistic_bonds(self):
        """Test CGBonds are inferred from atomistic bonds crossing beads."""
        atomistic = Atomistic()
        a1 = atomistic.def_atom(symbol="C", x=0, y=0, z=0)
        a2 = atomistic.def_atom(symbol="H", x=1, y=0, z=0)
        a3 = atomistic.def_atom(symbol="O", x=5, y=0, z=0)
        a4 = atomistic.def_atom(symbol="H", x=6, y=0, z=0)

        # Bond within bead 0
        atomistic.def_bond(a1, a2)
        # Bond crossing beads
        atomistic.def_bond(a2, a3)
        # Bond within bead 1
        atomistic.def_bond(a3, a4)

        mask = np.array([0, 0, 1, 1])

        cg = CoarseGrain.from_atomistic(atomistic, mask)

        # Should create one CGBond between the two beads
        assert len(cg.cgbonds) == 1

    def test_mask_length_validation(self):
        """Test error when mask length doesn't match atom count."""
        atomistic = Atomistic()
        atomistic.def_atom(symbol="C", x=0, y=0, z=0)

        mask = np.array([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="must match number of atoms"):
            CoarseGrain.from_atomistic(atomistic, mask)


class TestRoundTripConversion:
    """Test bidirectional conversion between Atomistic and CoarseGrain."""

    def test_round_trip_preserves_structure(self):
        """Test converting atomistic -> CG -> atomistic preserves connectivity."""
        # Create atomistic structure
        atomistic = Atomistic()
        a1 = atomistic.def_atom(symbol="C", x=0, y=0, z=0)
        a2 = atomistic.def_atom(symbol="H", x=1, y=0, z=0)
        a3 = atomistic.def_atom(symbol="O", x=5, y=0, z=0)
        atomistic.def_bond(a1, a2)
        atomistic.def_bond(a2, a3)

        # Convert to CG
        mask = np.array([0, 0, 1])
        cg = CoarseGrain.from_atomistic(atomistic, mask)

        # Convert back to atomistic
        atomistic2 = cg.to_atomistic()

        # Should have same number of atoms (from mapped structures)
        assert len(atomistic2.atoms) == 3
        # Should preserve connectivity
        assert len(atomistic2.bonds) >= 1


class TestGeneralImplementation:
    """Test that implementation is general without hardcoding."""

    def test_arbitrary_bead_types(self):
        """Test system works with arbitrary bead type names."""
        cg = CoarseGrain()
        cg.def_bead(type="CustomType1")
        cg.def_bead(type="AnotherType")
        cg.def_bead(type="X")

        assert len(cg.beads) == 3

    def test_arbitrary_bond_attributes(self):
        """Test CGBonds support arbitrary attributes."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")

        bond = cg.def_cgbond(
            b1, b2, custom_attr="value", strength=100.0, another_field=[1, 2, 3]
        )

        assert bond.get("custom_attr") == "value"
        assert bond.get("strength") == 100.0
        assert bond.get("another_field") == [1, 2, 3]

    def test_arbitrary_bead_attributes(self):
        """Test beads support arbitrary attributes."""
        cg = CoarseGrain()
        bead = cg.def_bead(
            type="Custom", mass=50.0, charge=-1.5, metadata={"key": "value"}
        )

        assert bead.get("mass") == 50.0
        assert bead.get("charge") == -1.5
        assert bead.get("metadata") == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
