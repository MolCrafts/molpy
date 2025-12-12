"""
Tests for PolymerBuilder class.
"""

import pytest

from molpy.builder.polymer import PolymerBuilder
from molpy.core.atomistic import Atomistic
from molpy.parser.smiles import parse_bigsmiles, bigsmilesir_to_monomer
from molpy.external import RDKitAdapter, Generate3D


def create_test_monomer(smiles: str) -> Atomistic:
    """Helper to create a test monomer from SMILES."""
    # Parse and convert to Atomistic
    ir = parse_bigsmiles(f"{{[<]{smiles}[>]}}")
    monomer = bigsmilesir_to_monomer(ir.stochastic_objects[0].monomers[0])
    
    # Add hydrogens and generate 3D
    adapter = RDKitAdapter(internal=monomer)
    generate_3d = Generate3D(add_hydrogens=True, embed=True, optimize=False, update_internal=True)
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
        
        # Create minimal connector
        reacter = Reacter(
            name="test",
            port_selector_left=lambda a, p: None,
            port_selector_right=lambda a, p: None,
            leaving_selector_left=lambda a, p: [],
            leaving_selector_right=lambda a, p: [],
            bond_former=form_single_bond,
        )
        connector = ReacterConnector(default=reacter)
        
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
            port_selector_left=lambda a, p: None,
            port_selector_right=lambda a, p: None,
            leaving_selector_left=lambda a, p: [],
            leaving_selector_right=lambda a, p: [],
            bond_former=form_single_bond,
        )
        connector = ReacterConnector(default=reacter)
        
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
            port_selector_left=lambda a, p: None,
            port_selector_right=lambda a, p: None,
            leaving_selector_left=lambda a, p: [],
            leaving_selector_right=lambda a, p: [],
            bond_former=form_single_bond,
        )
        connector = ReacterConnector(default=reacter)
        
        builder = PolymerBuilder(library=library, connector=connector)
        
        # Try to build empty graph
        with pytest.raises(ValueError, match="empty"):
            builder.build("{}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
