"""Tests for BigSMILES to Monomer conversion."""

import pytest

from molpy import Atomistic
from molpy.core.wrappers.monomer import Monomer
from molpy.parser.smiles import (
    BondDescriptorIR,
    PolymerSegment,
    PolymerSpec,
    SmilesParser,
    bigsmilesir_to_monomer,
    bigsmilesir_to_polymerspec,
    descriptor_to_port_name,
)


class TestDegenerate:
    """Test BigSmilesIR.degenerate() method."""

    def test_degenerate_removes_descriptors(self):
        """Test that degenerate() returns clean SmilesIR."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}")

        # Original has no descriptors in atoms (they're extracted to chain)
        assert len(ir.atoms) == 2  # C, C only

        # Degenerated should also be clean
        clean_ir = ir.degenerate()
        assert len(clean_ir.atoms) == 2  # C, C only

        # All atoms should be AtomIR
        from molpy.parser.smiles import AtomIR

        assert all(isinstance(a, AtomIR) for a in clean_ir.atoms)

    def test_degenerate_preserves_bonds(self):
        """Test that degenerate() keeps only real atom bonds."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}")

        clean_ir = ir.degenerate()

        # Should have 1 bond between the two carbons
        assert len(clean_ir.bonds) >= 1

        # All bonds should connect AtomIR
        from molpy.parser.smiles import AtomIR

        for bond in clean_ir.bonds:
            assert isinstance(bond.start, AtomIR)
            assert isinstance(bond.end, AtomIR)


class TestBigSmilesToMonomer:
    """Test bigsmilesir_to_monomer() function for single repeat units."""

    def test_simple_monomer_topology(self):
        """Test {[<]CC[>]} creates monomer without coords."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}")
        monomer = bigsmilesir_to_monomer(ir)

        # Check types
        assert isinstance(monomer, Monomer)

        # Check topology
        assert len(monomer.atoms) == 2  # 2 carbons
        assert set(monomer.port_names()) == {"in", "out"}

        # Check port connections
        assert monomer.get_port("in").target == monomer.atoms[0]  # First atom
        assert monomer.get_port("out").target == monomer.atoms[-1]  # Last atom

        # No coordinates! (Atom is dict-like)
        for atom in monomer.atoms:
            assert "xyz" not in atom or atom["xyz"] is None

    def test_atom_class_port_single(self):
        """Test CCCCO[*:1] creates monomer with atom class port."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("CCCCO[*:1]")
        monomer = bigsmilesir_to_monomer(ir)

        # Check structure
        assert len(monomer.atoms) == 5  # 4C + 1O (no * atom)
        symbols = [a["symbol"] for a in monomer.atoms]
        # Note: Order may vary based on parsing, but should have 4C + 1O
        assert symbols.count("C") == 4
        assert symbols.count("O") == 1

        # Check port
        assert monomer.port_names() == ["port_1"]
        port = monomer.get_port("port_1")
        assert port is not None
        assert port.target["symbol"] == "O"  # Port points to O (connected to [*:1])

    def test_atom_class_port_multiple(self):
        """Test CC(C[*:2])O[*:3] creates monomer with multiple ports."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("CC(C[*:2])O[*:3]")
        monomer = bigsmilesir_to_monomer(ir)

        # Check structure
        assert len(monomer.atoms) == 4  # 3C + 1O (no * atoms)

        # Check ports
        assert set(monomer.port_names()) == {"port_2", "port_3"}

        # Port 2 points to C (connected to [*:2])
        port_2 = monomer.get_port("port_2")
        assert port_2 is not None and port_2.target["symbol"] == "C"

        # Port 3 points to O (connected to [*:3])
        port_3 = monomer.get_port("port_3")
        assert port_3 is not None and port_3.target["symbol"] == "O"

    def test_multiple_repeat_units_raises_error(self):
        """Test that multiple repeat units raise ValueError."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>],[<]OCC[>]}")

        with pytest.raises(ValueError, match="multiple repeat units"):
            bigsmilesir_to_monomer(ir)

    def test_block_copolymer_raises_error(self):
        """Test that block copolymer raises ValueError."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}{[<]O[>]}")

        with pytest.raises(ValueError, match="multiple repeat units"):
            bigsmilesir_to_monomer(ir)

    def test_port_connections(self):
        """Test ports connect to correct atoms."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}")
        monomer = bigsmilesir_to_monomer(ir)
        atomistic = monomer

        # Check port names
        assert set(monomer.port_names()) == {"in", "out"}

        # Check port atoms (left -> first, right -> last)
        in_port = monomer.get_port("in")
        out_port = monomer.get_port("out")
        assert in_port is not None and in_port.target is atomistic.atoms[0]
        assert out_port is not None and out_port.target is atomistic.atoms[1]


class TestDescriptorToPortName:
    """Test descriptor_to_port_name() function."""

    def test_basic_descriptors(self):
        """Test basic descriptor symbols."""
        assert descriptor_to_port_name(BondDescriptorIR(symbol="<")) == "in"
        assert descriptor_to_port_name(BondDescriptorIR(symbol=">")) == "out"
        assert descriptor_to_port_name(BondDescriptorIR(symbol="$")) == "branch"

    def test_indexed_descriptors(self):
        """Test indexed descriptors."""
        assert descriptor_to_port_name(BondDescriptorIR(symbol="<", index=1)) == "in_1"
        assert descriptor_to_port_name(BondDescriptorIR(symbol=">", index=2)) == "out_2"
        assert (
            descriptor_to_port_name(BondDescriptorIR(symbol="$", index=3)) == "branch_3"
        )

    def test_none_symbol(self):
        """Test descriptor with None symbol."""
        result = descriptor_to_port_name(BondDescriptorIR(symbol=None))
        assert result == "port"


class TestCoordinateBinding:
    """Test that users can bind coordinates to atoms."""

    def test_coordinate_binding(self):
        """Test user can bind coordinates to atoms."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}")
        monomer = bigsmilesir_to_monomer(ir)

        # User generates coords (mocked here)
        coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]

        # User binds coords (Atom is dict-like)
        for i, atom in enumerate(monomer.atoms):
            atom["xyz"] = coords[i]

        # Verify
        assert monomer.atoms[0]["xyz"] == (0.0, 0.0, 0.0)
        assert monomer.atoms[1]["xyz"] == (1.5, 0.0, 0.0)


class TestPolymerSpec:
    """Test PolymerSpec and bigsmiles_to_polymerspec()."""

    def test_homopolymer(self):
        """Test {[<]CC[>]} -> homopolymer."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        assert isinstance(spec, PolymerSpec)
        assert spec.topology == "homopolymer"
        assert len(spec.segments) == 1
        assert len(spec.segments[0].monomers) == 1
        assert len(spec.all_monomers) == 1

    def test_random_copolymer(self):
        """Test {[<]CC[>],[<]OCC[>]} -> random copolymer."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>],[<]OCC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        assert spec.topology == "random_copolymer"
        assert len(spec.segments) == 1
        assert len(spec.segments[0].monomers) == 2
        assert spec.segments[0].composition_type == "random"
        assert len(spec.all_monomers) == 2

    def test_block_copolymer(self):
        """Test {[<]CC[>]}{[<]OCC[>]} -> block copolymer."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}{[<]OCC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        assert spec.topology == "block_copolymer"
        assert len(spec.segments) == 2
        assert len(spec.all_monomers) == 2

        # First block
        assert len(spec.segments[0].monomers) == 1
        assert len(spec.segments[0].monomers[0].atoms) == 2  # CC

        # Second block
        assert len(spec.segments[1].monomers) == 1
        assert len(spec.segments[1].monomers[0].atoms) == 3  # OCC

    def test_polymer_segment_structure(self):
        """Test PolymerSegment structure."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>],[<]OCC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)
        segment = spec.segments[0]

        assert isinstance(segment, PolymerSegment)
        assert isinstance(segment.monomers, list)
        assert len(segment.monomers) == 2
        assert all(isinstance(m, Monomer) for m in segment.monomers)
        assert segment.composition_type == "random"

    def test_all_monomers_computed(self):
        """Test that all_monomers is correctly computed."""
        parser = SmilesParser()
        ir = parser.parse_bigsmiles("{[<]CC[>]}{[<]OCC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        # 2 blocks, 2 monomers total
        assert len(spec.segments) == 2
        assert len(spec.all_monomers) == 2

        # Check they're the actual monomers from segments
        expected_monomers = []
        for segment in spec.segments:
            expected_monomers.extend(segment.monomers)

        assert spec.all_monomers == expected_monomers
