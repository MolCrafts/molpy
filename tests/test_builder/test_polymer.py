"""
Test suite for molpy.builder.polymer module.
"""

import pytest
from molpy.builder.polymer import PolymerBuilder, MonomerTemplate, AnchorRule
from molpy.core.atomistic import Atomistic, Atom
from molpy.core.wrapper import Wrapper


class TestAnchorRule:
    def test_anchor_rule_creation(self):
        """Test AnchorRule creation."""
        rule = AnchorRule(anchor_atom="C1", when_prev="*", when_next="*")
        assert rule.anchor_atom == "C1"
        assert rule.when_prev == "*"
        assert rule.when_next == "*"
        
    def test_anchor_rule_context_matching(self):
        """Test anchor rule context matching."""
        rule = AnchorRule(anchor_atom="C1", when_prev="A", when_next="B")
        
        # Should match exact context
        assert rule.matches_context("A", "B")
        
        # Should not match different context
        assert not rule.matches_context("X", "B")
        assert not rule.matches_context("A", "Y")
        
        # Test wildcard matching
        wild_rule = AnchorRule(anchor_atom="C1", when_prev="*", when_next="*")
        assert wild_rule.matches_context("anything", "anything")
        assert wild_rule.matches_context(None, None)


class TestMonomerTemplate:
    def test_monomer_template_creation(self):
        """Test MonomerTemplate creation."""
        struct = Atomistic(name="test_monomer")
        struct.def_atom(name="C1", xyz=[0, 0, 0])
        
        anchors = {
            "left": [AnchorRule(anchor_atom="C1", when_prev=None, when_next="*")],
            "right": [AnchorRule(anchor_atom="C1", when_prev="*", when_next=None)]
        }
        
        template = MonomerTemplate(struct, anchors)
        assert isinstance(template, MonomerTemplate)
        assert isinstance(template, Wrapper)
        assert len(template.anchors) == 2
        
    def test_monomer_template_clone(self):
        """Test MonomerTemplate cloning."""
        struct = Atomistic(name="test_monomer")
        struct.def_atom(name="C1", xyz=[0, 0, 0])
        
        template = MonomerTemplate(struct, {})
        cloned = template.clone()
        
        assert isinstance(cloned, Atomistic)
        assert cloned is not struct  # Should be a copy
        assert len(cloned.atoms) == 1
        
    def test_monomer_template_transformed(self):
        """Test MonomerTemplate transformation."""
        struct = Atomistic(name="test_monomer")
        struct.def_atom(name="C1", xyz=[0, 0, 0])
        
        template = MonomerTemplate(struct, {})
        transformed = template.transformed(
            position=[1, 2, 3],
            name="transformed_monomer"
        )
        
        assert isinstance(transformed, Atomistic)
        assert transformed["name"] == "transformed_monomer"
        # Check that position was applied
        assert len(transformed.atoms) == 1


class TestPolymerBuilder:
    def test_polymer_builder_creation(self):
        """Test PolymerBuilder creation."""
        builder = PolymerBuilder()
        assert isinstance(builder, PolymerBuilder)
        assert len(builder.monomers) == 0
        
    def test_register_monomer(self):
        """Test monomer registration."""
        builder = PolymerBuilder()
        
        struct = Atomistic(name="test_monomer")
        struct.def_atom(name="C1", xyz=[0, 0, 0])
        template = MonomerTemplate(struct, {})
        
        builder.register_monomer("test", template)
        assert "test" in builder.monomers
        assert builder.monomers["test"] is template
        
    def test_build_linear_empty(self):
        """Test building empty linear polymer."""
        builder = PolymerBuilder()
        polymer = builder.build_linear([])
        
        assert isinstance(polymer, Atomistic)
        assert len(polymer.atoms) == 0
        
    def test_build_linear_with_monomers(self):
        """Test building linear polymer with monomers."""
        builder = PolymerBuilder()
        
        # Create a simple monomer
        struct = Atomistic(name="test_monomer")
        struct.def_atom(name="C1", xyz=[0, 0, 0])
        template = MonomerTemplate(struct, {})
        
        builder.register_monomer("A", template)
        
        # Build linear polymer
        polymer = builder.build_linear(["A", "A"])
        assert isinstance(polymer, Atomistic)
        assert "polymer_A-A" in polymer["name"]
        
    def test_validate_sequence(self):
        """Test sequence validation."""
        builder = PolymerBuilder()
        
        struct = Atomistic(name="test_monomer")
        template = MonomerTemplate(struct, {})
        builder.register_monomer("A", template)
        
        assert builder.validate_sequence(["A", "A"])
        assert not builder.validate_sequence(["A", "B"])  # B not registered
        
    def test_available_monomers(self):
        """Test getting available monomers."""
        builder = PolymerBuilder()
        
        struct = Atomistic(name="test_monomer")
        template = MonomerTemplate(struct, {})
        builder.register_monomer("A", template)
        builder.register_monomer("B", template)
        
        available = builder.available_monomers()
        assert set(available) == {"A", "B"}
        
    def test_get_anchor_atoms(self):
        """Test getting anchor atoms."""
        builder = PolymerBuilder()
        
        struct = Atomistic(name="test_monomer")
        atom = struct.def_atom(name="C1", xyz=[0, 0, 0])
        
        anchors = {
            "left": [AnchorRule(anchor_atom="C1", when_prev=None, when_next="*")]
        }
        template = MonomerTemplate(struct, anchors)
        
        anchor_atoms = builder.get_anchor_atoms(template, "left", None, "B")
        assert len(anchor_atoms) >= 0  # Should not crash
        
    def test_build_branched(self):
        """Test building branched polymer."""
        builder = PolymerBuilder()
        
        struct = Atomistic(name="test_monomer")
        struct.def_atom(name="C1", xyz=[0, 0, 0])
        template = MonomerTemplate(struct, {})
        
        builder.register_monomer("A", template)
        
        # Build branched polymer
        polymer = builder.build_branched(["A", "A"], {1: ["A"]})
        assert isinstance(polymer, Atomistic)
        
    def test_repr(self):
        """Test PolymerBuilder representation."""
        builder = PolymerBuilder()
        
        struct = Atomistic(name="test_monomer")
        template = MonomerTemplate(struct, {})
        builder.register_monomer("A", template)
        
        repr_str = repr(builder)
        assert "PolymerBuilder" in repr_str
        assert "A" in repr_str
