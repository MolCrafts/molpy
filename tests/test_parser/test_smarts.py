"""
Unit tests for SMARTS parser.

Tests cover:
- Simple atoms (symbols, wildcards)
- Atomic numbers
- Logical expressions (AND, OR, NOT, weak AND)
- Neighbor counts, ring sizes, ring counts
- Branches
- Ring closures
- Complex patterns
"""

import pytest

from molpy.parser.smarts import (
    AtomExpressionIR,
    AtomPrimitiveIR,
    SmartsParser,
)


class TestSimpleAtoms:
    """Test parsing of simple atom patterns."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_single_carbon(self, parser):
        """Test parsing single carbon atom."""
        ir = parser.parse_smarts("C")
        assert len(ir.atoms) == 1
        assert len(ir.bonds) == 0

        atom = ir.atoms[0]
        assert isinstance(atom.expression, AtomExpressionIR)
        assert atom.expression.op == "primitive"
        assert len(atom.expression.children) == 1

        prim = atom.expression.children[0]
        assert isinstance(prim, AtomPrimitiveIR)
        assert prim.type == "symbol"
        assert prim.value == "C"

    def test_wildcard(self, parser):
        """Test parsing wildcard atom."""
        ir = parser.parse_smarts("*")
        assert len(ir.atoms) == 1

        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "wildcard"

    def test_multiple_atoms(self, parser):
        """Test parsing chain of atoms."""
        ir = parser.parse_smarts("CCO")
        assert len(ir.atoms) == 3
        assert len(ir.bonds) == 2

        # Check atoms
        symbols = [a.expression.children[0].value for a in ir.atoms]
        assert symbols == ["C", "C", "O"]

        # Check bonds
        assert all(b.bond_type == "-" for b in ir.bonds)


class TestBracketedAtoms:
    """Test parsing of bracketed atom expressions."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_atomic_number(self, parser):
        """Test parsing atomic number [#6]."""
        ir = parser.parse_smarts("[#6]")
        assert len(ir.atoms) == 1

        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "atomic_num"
        assert prim.value == 6

    def test_explicit_symbol(self, parser):
        """Test parsing explicit symbol [C]."""
        ir = parser.parse_smarts("[C]")
        assert len(ir.atoms) == 1

        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "symbol"
        assert prim.value == "C"


class TestLogicalExpressions:
    """Test parsing of logical expressions in atoms."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_or_expression(self, parser):
        """Test parsing OR expression [C,N,O]."""
        ir = parser.parse_smarts("[C,N,O]")
        assert len(ir.atoms) == 1

        expr = ir.atoms[0].expression
        assert expr.op == "or"
        assert len(expr.children) == 3

        # Each child should be a primitive
        for i, expected in enumerate(["C", "N", "O"]):
            child = expr.children[i]
            if isinstance(child, AtomExpressionIR) and child.op == "primitive":
                prim = child.children[0]
            else:
                prim = child
            assert isinstance(prim, AtomPrimitiveIR)
            assert prim.value == expected

    def test_and_expression(self, parser):
        """Test parsing AND expression [C&X4]."""
        ir = parser.parse_smarts("[C&X4]")
        assert len(ir.atoms) == 1

        expr = ir.atoms[0].expression
        assert expr.op == "and"
        assert len(expr.children) == 2

        # First child: C
        child0 = expr.children[0]
        if isinstance(child0, AtomExpressionIR):
            child0 = child0.children[0]
        assert isinstance(child0, AtomPrimitiveIR)
        assert child0.type == "symbol"
        assert child0.value == "C"

        # Second child: X4
        child1 = expr.children[1]
        if isinstance(child1, AtomExpressionIR):
            child1 = child1.children[0]
        assert isinstance(child1, AtomPrimitiveIR)
        assert child1.type == "neighbor_count"
        assert child1.value == 4

    def test_not_expression(self, parser):
        """Test parsing NOT expression [!C]."""
        ir = parser.parse_smarts("[!C]")
        assert len(ir.atoms) == 1

        expr = ir.atoms[0].expression
        # Could be wrapped in primitive
        if expr.op == "primitive":
            expr = expr.children[0]

        assert isinstance(expr, AtomExpressionIR)
        assert expr.op == "not"
        assert len(expr.children) == 1

        child = expr.children[0]
        if isinstance(child, AtomExpressionIR):
            child = child.children[0]
        assert isinstance(child, AtomPrimitiveIR)
        assert child.type == "symbol"
        assert child.value == "C"

    def test_weak_and_expression(self, parser):
        """Test parsing weak AND expression [C;X4]."""
        ir = parser.parse_smarts("[C;X4]")
        assert len(ir.atoms) == 1

        expr = ir.atoms[0].expression
        assert expr.op == "weak_and"
        assert len(expr.children) == 2


class TestBranches:
    """Test parsing of branched structures."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_simple_branch(self, parser):
        """Test parsing C(C)C - propane."""
        ir = parser.parse_smarts("C(C)C")
        assert len(ir.atoms) == 3
        assert len(ir.bonds) == 2

        # Atom 0 should connect to atoms 1 and 2
        bond_pairs = {(id(b.start), id(b.end)) for b in ir.bonds}
        atom_ids = [id(a) for a in ir.atoms]

        # Check connectivity
        assert (atom_ids[0], atom_ids[1]) in bond_pairs or (
            atom_ids[1],
            atom_ids[0],
        ) in bond_pairs
        assert (atom_ids[0], atom_ids[2]) in bond_pairs or (
            atom_ids[2],
            atom_ids[0],
        ) in bond_pairs

    def test_multiple_branches(self, parser):
        """Test parsing C(C)(O)N - branched structure."""
        ir = parser.parse_smarts("C(C)(O)N")
        assert len(ir.atoms) == 4
        assert len(ir.bonds) == 3


class TestRings:
    """Test parsing of ring structures."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_simple_ring(self, parser):
        """Test parsing C1CCCCC1 - cyclohexane."""
        ir = parser.parse_smarts("C1CCCCC1")
        assert len(ir.atoms) == 6
        assert len(ir.bonds) == 6

        # Should form a closed ring
        # Verify ring closure: last bond connects atom 5 to atom 0
        ring_bond = next(b for b in ir.bonds if id(b.start) == id(ir.atoms[5]))
        assert id(ring_bond.end) == id(ir.atoms[0])

    def test_aromatic_ring(self, parser):
        """Test parsing c1ccccc1 - benzene."""
        ir = parser.parse_smarts("c1ccccc1")
        assert len(ir.atoms) == 6
        assert len(ir.bonds) == 6

        # Check all atoms are aromatic (lowercase)
        for atom in ir.atoms:
            prim = atom.expression.children[0]
            # Symbol should be 'c' (lowercase)
            assert prim.value == "c"

    def test_multiple_rings(self, parser):
        """Test parsing C1CC2CCC1C2 - bicyclic structure."""
        ir = parser.parse_smarts("C1CC2CCC1C2")
        assert len(ir.atoms) == 7
        # Should have ring closures for both 1 and 2
        assert len(ir.bonds) == 8

    def test_unclosed_ring_error(self, parser):
        """Test that unclosed rings raise an error."""
        with pytest.raises(ValueError, match="Unclosed rings"):
            parser.parse_smarts("C1CCC")


class TestComplexPatterns:
    """Test parsing of complex SMARTS patterns."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_combined_or_and_pattern(self, parser):
        """Test parsing [C,N]C[O,S] - heteroatom pattern."""
        ir = parser.parse_smarts("[C,N]C[O,S]")
        assert len(ir.atoms) == 3
        assert len(ir.bonds) == 2

        # First atom: [C,N]
        expr0 = ir.atoms[0].expression
        assert expr0.op == "or"

        # Last atom: [O,S]
        expr2 = ir.atoms[2].expression
        assert expr2.op == "or"

    def test_negation_pattern(self, parser):
        """Test parsing [!H]C[!H] - non-hydrogen pattern."""
        ir = parser.parse_smarts("[!H]C[!H]")
        assert len(ir.atoms) == 3

        # First and last atoms should have NOT expressions
        for i in [0, 2]:
            expr = ir.atoms[i].expression
            if expr.op == "primitive":
                expr = expr.children[0]
            assert isinstance(expr, AtomExpressionIR)
            assert expr.op == "not"

    def test_ring_size_pattern(self, parser):
        """Test parsing [r5] - 5-membered ring atom."""
        ir = parser.parse_smarts("[r5]")
        assert len(ir.atoms) == 1

        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "ring_size"
        assert prim.value == 5

    def test_ring_count_pattern(self, parser):
        """Test parsing [R2] - atom in 2 rings."""
        ir = parser.parse_smarts("[R2]")
        assert len(ir.atoms) == 1

        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "ring_count"
        assert prim.value == 2

    def test_neighbor_count_pattern(self, parser):
        """Test parsing [X3] - 3 neighbors."""
        ir = parser.parse_smarts("[X3]")
        assert len(ir.atoms) == 1

        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "neighbor_count"
        assert prim.value == 3

    def test_hydrogen_count_pattern(self, parser):
        """Test parsing [C;H2] - carbon with 2 hydrogens."""
        ir = parser.parse_smarts("[C;H2]")
        assert len(ir.atoms) == 1

        # Should have weak_and expression with C and H2
        expr = ir.atoms[0].expression
        assert expr.op == "weak_and"
        assert len(expr.children) == 2

        # First child should be C symbol
        assert expr.children[0].type == "symbol"
        assert expr.children[0].value == "C"

        # Second child should be hydrogen_count
        assert expr.children[1].type == "hydrogen_count"
        assert expr.children[1].value == 2

    def test_implicit_hydrogen_count_pattern(self, parser):
        """Test parsing [C;h2] - carbon with 2 implicit hydrogens."""
        ir = parser.parse_smarts("[C;h2]")
        assert len(ir.atoms) == 1

        # Should have weak_and expression with C and h2
        expr = ir.atoms[0].expression
        assert expr.op == "weak_and"
        assert len(expr.children) == 2

        # First child should be C symbol
        assert expr.children[0].type == "symbol"
        assert expr.children[0].value == "C"

        # Second child should be implicit_hydrogen_count
        assert expr.children[1].type == "implicit_hydrogen_count"
        assert expr.children[1].value == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_empty_string(self, parser):
        """Test parsing empty string."""
        # This may raise a parse error or return empty IR
        # depending on grammar definition
        try:
            ir = parser.parse_smarts("")
            assert len(ir.atoms) == 0
        except Exception:
            # Parse error is acceptable for empty input
            pass

    def test_single_wildcard(self, parser):
        """Test single wildcard."""
        ir = parser.parse_smarts("*")
        assert len(ir.atoms) == 1
        prim = ir.atoms[0].expression.children[0]
        assert prim.type == "wildcard"

    def test_multiple_wildcards(self, parser):
        """Test chain of wildcards."""
        ir = parser.parse_smarts("***")
        assert len(ir.atoms) == 3
        assert len(ir.bonds) == 2


# ===================================================================
#   IR Structure Tests
# ===================================================================


class TestIRStructure:
    """Test the intermediate representation structure."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_ir_atoms_list(self, parser):
        """Test that IR contains atoms list."""
        ir = parser.parse_smarts("CC")
        assert hasattr(ir, "atoms")
        assert isinstance(ir.atoms, list)
        assert len(ir.atoms) == 2

    def test_ir_bonds_list(self, parser):
        """Test that IR contains bonds list."""
        ir = parser.parse_smarts("CC")
        assert hasattr(ir, "bonds")
        assert isinstance(ir.bonds, list)
        assert len(ir.bonds) == 1

    def test_atom_expression_structure(self, parser):
        """Test atom expression structure."""
        ir = parser.parse_smarts("C")
        atom = ir.atoms[0]

        assert hasattr(atom, "expression")
        assert isinstance(atom.expression, AtomExpressionIR)
        assert hasattr(atom.expression, "op")
        assert hasattr(atom.expression, "children")

    def test_primitive_structure(self, parser):
        """Test primitive atom structure."""
        ir = parser.parse_smarts("C")
        prim = ir.atoms[0].expression.children[0]

        assert isinstance(prim, AtomPrimitiveIR)
        assert hasattr(prim, "type")
        assert hasattr(prim, "value")
        assert prim.type == "symbol"
        assert prim.value == "C"

    def test_bond_structure(self, parser):
        """Test bond structure."""
        ir = parser.parse_smarts("CC")
        bond = ir.bonds[0]

        assert hasattr(bond, "start")
        assert hasattr(bond, "end")
        assert hasattr(bond, "bond_type")
        # Default bond type (implicit single bond)
        assert bond.bond_type in ["-", None, ""]


# ===================================================================
#   Advanced Pattern Tests
# ===================================================================


class TestAdvancedPatterns:
    """Test advanced SMARTS patterns."""

    @pytest.fixture
    def parser(self):
        return SmartsParser()

    def test_combined_and_or(self, parser):
        """Test combined AND/OR expressions."""
        ir = parser.parse_smarts("[C&X4,N&X3]")
        assert len(ir.atoms) == 1
        expr = ir.atoms[0].expression
        assert expr.op == "or"
        # First branch: C&X4
        # Second branch: N&X3

    def test_multiple_constraints(self, parser):
        """Test multiple constraints on atom."""
        ir = parser.parse_smarts("[C;X4;H2;r5]")
        assert len(ir.atoms) == 1
        expr = ir.atoms[0].expression
        assert expr.op == "weak_and"
        # Should have all constraints

    def test_aromatic_detection(self, parser):
        """Test aromatic atom detection."""
        ir = parser.parse_smarts("c1ccccc1")
        assert len(ir.atoms) == 6

        # All atoms should be aromatic lowercase 'c'
        for atom in ir.atoms:
            prim = atom.expression.children[0]
            assert prim.value == "c"
