"""
Tests for PolymerBuilder class.
"""

import pytest

# Skip entire test module if RDKit is not available
pytest.importorskip("rdkit", reason="RDKit is not installed")

from molpy.builder.polymer import PolymerBuilder
from molpy.core.atomistic import Atomistic
from molpy.parser.smiles import parse_bigsmiles, bigsmilesir_to_monomer
from molpy.adapter import RDKitAdapter
from molpy.compute import Generate3D


def create_test_monomer(smiles: str) -> Atomistic:
    """Helper to create a test monomer from SMILES."""
    # Parse and convert to Atomistic
    # Use {[<]smiles[>]} format to ensure ports are created
    ir = parse_bigsmiles(f"{{[<]{smiles}[>]}}")
    # Pass full IR to bigsmilesir_to_monomer
    monomer = bigsmilesir_to_monomer(ir)

    # Add hydrogens and generate 3D
    adapter = RDKitAdapter(internal=monomer)
    generate_3d = Generate3D(
        add_hydrogens=True, embed=True, optimize=False, update_internal=True
    )
    adapter = generate_3d(adapter)
    monomer = adapter.get_internal()
    monomer.get_topo(gen_angle=True, gen_dihe=True)

    # Assign IDs
    for idx, atom in enumerate(monomer.atoms):
        atom["id"] = idx + 1

    return monomer


class TestPolymerBuilder:
    """Tests for PolymerBuilder class."""

    def test_import(self):
        """Test that PolymerBuilder can be imported."""
        from molpy.builder.polymer import PolymerBuilder

        assert PolymerBuilder is not None

    def test_init(self):
        """Test PolymerBuilder initialization."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        # Create minimal library
        library = {}

        # Create minimal connector with required port_map
        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        # port_map: maps (left_port, right_port) to (left_port_name, right_port_name)
        port_map = {(">", "<"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        # Initialize builder
        builder = PolymerBuilder(
            library=library,
            connector=connector,
        )

        assert builder.library == library
        assert builder.connector == connector

    def test_validate_missing_label(self):
        """Test validation catches missing labels."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.builder.polymer.errors import SequenceError
        from molpy.reacter import Reacter, form_single_bond

        # Create library with only one label
        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {(">", "<"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)

        # Try to build with missing label
        with pytest.raises(SequenceError, match="not found in library"):
            builder.build("{[#A][#B]}")

    def test_validate_empty_graph(self):
        """Test validation catches empty graphs."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {(">", "<"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)

        # Try to build empty graph
        with pytest.raises(ValueError, match="empty"):
            builder.build("{}")

    def test_build_linear_chain(self):
        """Test building a simple linear chain."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        # Create library with one monomer type
        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "A"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)

        # Build linear chain: {[#A][#A][#A]}
        result = builder.build("{[#A][#A][#A]}")

        assert result.polymer is not None
        assert result.total_steps == 2  # Two connections for 3 monomers
        assert len(result.connection_history) == 2

    def test_build_with_repeat_operator(self):
        """Test building with repeat operator."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "A"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)

        # Build with repeat: {[#A]|3}
        result = builder.build("{[#A]|3}")

        assert result.polymer is not None
        assert result.total_steps == 2  # Two connections for 3 monomers

    def test_build_branched_structure(self):
        """Test building a branched structure."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "A"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)

        # Build branched: {[#A]([#A])[#A]}
        result = builder.build("{[#A]([#A])[#A]}")

        assert result.polymer is not None
        assert result.total_steps == 2  # Two connections for 3 monomers

    def test_build_cyclic_structure(self):
        """Test building a cyclic structure."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "A"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)

        # Build cyclic: {[#A]1[#A][#A]1}
        result = builder.build("{[#A]1[#A][#A]1}")

        assert result.polymer is not None
        assert result.total_steps == 3  # Three connections for 3 monomers in ring

    def test_connection_history_tracking(self):
        """Test that connection history is properly tracked."""
        from molpy.builder.polymer import PolymerBuilder, ReacterConnector
        from molpy.reacter import Reacter, form_single_bond

        library = {"A": create_test_monomer("CC")}

        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "A"): (">", "<")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        builder = PolymerBuilder(library=library, connector=connector)
        result = builder.build("{[#A][#A]}")

        assert len(result.connection_history) == 1
        metadata = result.connection_history[0]
        assert hasattr(metadata, "port_L")
        assert hasattr(metadata, "port_R")
        assert hasattr(metadata, "reaction_name")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
