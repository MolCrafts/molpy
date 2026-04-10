"""Comprehensive CGSmiles syntax tests.

Tests are organized by the original paper:

  CGSmiles: Grunewald, Kroon et al.; JCIM 2025
  DOI: 10.1021/acs.jcim.5c00064

Each test class corresponds to a syntactic feature in the specification.
"""

from molpy.parser.smiles import parse_cgsmiles


# =====================================================================
# §2.1  Basic Node Syntax
# =====================================================================


class TestNodeSyntax:
    """CGSmiles node notation: [#Label].

    Paper §2.1: "Each node is a labeled bead enclosed in brackets."
    """

    def test_single_node(self):
        """{[#A]} — single node."""
        ir = parse_cgsmiles("{[#A]}")
        assert len(ir.base_graph.nodes) == 1
        assert ir.base_graph.nodes[0].label == "A"

    def test_two_nodes(self):
        """{[#A][#B]} — two connected nodes."""
        ir = parse_cgsmiles("{[#A][#B]}")
        assert len(ir.base_graph.nodes) == 2
        assert len(ir.base_graph.bonds) == 1

    def test_chain(self):
        """{[#A][#B][#C][#D]} — linear chain."""
        ir = parse_cgsmiles("{[#A][#B][#C][#D]}")
        assert len(ir.base_graph.nodes) == 4
        assert len(ir.base_graph.bonds) == 3

    def test_label_with_numbers(self):
        """{[#TC5]} — alphanumeric label."""
        ir = parse_cgsmiles("{[#TC5]}")
        assert ir.base_graph.nodes[0].label == "TC5"

    def test_label_with_underscore(self):
        """{[#my_bead]} — underscore in label."""
        ir = parse_cgsmiles("{[#my_bead]}")
        assert ir.base_graph.nodes[0].label == "my_bead"


# =====================================================================
# §2.2  Annotations (key=value)
# =====================================================================


class TestAnnotations:
    """CGSmiles node annotations: [#Label;key=value].

    Paper §2.2: "Nodes can carry arbitrary key-value annotations
    separated by semicolons."
    """

    def test_charge_annotation(self):
        """{[#A;q=1]} — charge annotation."""
        ir = parse_cgsmiles("{[#A;q=1]}")
        node = ir.base_graph.nodes[0]
        assert node.annotations.get("q") == "1"

    def test_weight_annotation(self):
        """{[#A;w=0.5]} — weight annotation."""
        ir = parse_cgsmiles("{[#A;w=0.5]}")
        node = ir.base_graph.nodes[0]
        assert node.annotations.get("w") == "0.5"

    def test_multiple_annotations(self):
        """{[#A;q=1;w=0.5]} — multiple annotations."""
        ir = parse_cgsmiles("{[#A;q=1;w=0.5]}")
        node = ir.base_graph.nodes[0]
        assert node.annotations.get("q") == "1"
        assert node.annotations.get("w") == "0.5"

    def test_custom_annotation(self):
        """{[#A;foo=bar]} — arbitrary custom annotation."""
        ir = parse_cgsmiles("{[#A;foo=bar]}")
        node = ir.base_graph.nodes[0]
        assert node.annotations.get("foo") == "bar"

    def test_mixed_annotations(self):
        """{[#A;q=1;foo=bar;w=2.5]} — predefined + custom."""
        ir = parse_cgsmiles("{[#A;q=1;foo=bar;w=2.5]}")
        node = ir.base_graph.nodes[0]
        assert node.annotations.get("q") == "1"
        assert node.annotations.get("foo") == "bar"
        assert node.annotations.get("w") == "2.5"


# =====================================================================
# §2.3  Bond Types
# =====================================================================


class TestBondTypes:
    """CGSmiles bond types.

    Paper §2.3: ". (non-bonded), - (single), = (double), # (triple)"
    """

    def test_single_bond_implicit(self):
        """{[#A][#B]} — implicit single bond."""
        ir = parse_cgsmiles("{[#A][#B]}")
        assert len(ir.base_graph.bonds) == 1

    def test_single_bond_explicit(self):
        """{[#A]-[#B]} — explicit single bond."""
        ir = parse_cgsmiles("{[#A]-[#B]}")
        assert len(ir.base_graph.bonds) == 1

    def test_double_bond(self):
        """{[#A]=[#B]} — double bond."""
        ir = parse_cgsmiles("{[#A]=[#B]}")
        bond = ir.base_graph.bonds[0]
        assert bond.order == 2

    def test_triple_bond(self):
        """{[#A]#[#B]} — triple bond."""
        ir = parse_cgsmiles("{[#A]#[#B]}")
        bond = ir.base_graph.bonds[0]
        assert bond.order == 3

    def test_non_bonded(self):
        """{[#A].[#B]} — non-bonded (virtual edge)."""
        ir = parse_cgsmiles("{[#A].[#B]}")
        bond = ir.base_graph.bonds[0]
        assert bond.order == 0

    def test_mixed_bond_orders(self):
        """{[#A]=[#B]-[#C]} — mixed bond orders in chain."""
        ir = parse_cgsmiles("{[#A]=[#B]-[#C]}")
        assert len(ir.base_graph.bonds) == 2
        orders = sorted([b.order for b in ir.base_graph.bonds])
        assert orders == [1, 2]


# =====================================================================
# §2.4  Repeat Operator
# =====================================================================


class TestRepeatOperator:
    """CGSmiles repeat operator: |N.

    Paper §2.4: "[#A]|5 generates 5 consecutive A nodes."
    """

    def test_repeat_single_node(self):
        """{[#PEO]|5} — 5 PEO beads."""
        ir = parse_cgsmiles("{[#PEO]|5}")
        assert len(ir.base_graph.nodes) == 5

    def test_repeat_preserves_bonds(self):
        """Repeated nodes should be linearly connected."""
        ir = parse_cgsmiles("{[#A]|3}")
        assert len(ir.base_graph.nodes) == 3
        assert len(ir.base_graph.bonds) == 2

    def test_repeat_with_annotation(self):
        """{[#A;q=1]|3} — repeat with annotations."""
        ir = parse_cgsmiles("{[#A;q=1]|3}")
        for node in ir.base_graph.nodes:
            assert node.annotations.get("q") == "1"

    def test_repeat_large(self):
        """{[#A]|100} — large repeat count."""
        ir = parse_cgsmiles("{[#A]|100}")
        assert len(ir.base_graph.nodes) == 100


# =====================================================================
# §2.5  Ring Closures
# =====================================================================


class TestRingClosures:
    """CGSmiles ring closures: integer markers.

    Paper §2.5: "Ring closures follow SMILES convention."
    """

    def test_triangle(self):
        """{[#A]1[#B][#C]1} — triangle (3 nodes, 3 bonds)."""
        ir = parse_cgsmiles("{[#A]1[#B][#C]1}")
        assert len(ir.base_graph.nodes) == 3
        assert len(ir.base_graph.bonds) == 3

    def test_square(self):
        """{[#A]1[#B][#C][#D]1} — square (4 nodes, 4 bonds)."""
        ir = parse_cgsmiles("{[#A]1[#B][#C][#D]1}")
        assert len(ir.base_graph.nodes) == 4
        assert len(ir.base_graph.bonds) == 4

    def test_double_ring(self):
        """{[#A]1[#B]2[#C]1[#D]2} — two overlapping rings."""
        ir = parse_cgsmiles("{[#A]1[#B]2[#C]1[#D]2}")
        assert len(ir.base_graph.nodes) == 4
        # 2 chain bonds + 2 ring closure bonds
        assert len(ir.base_graph.bonds) >= 4

    def test_ring_with_bond_order(self):
        """{[#A].1[#B][#C]1} — ring with non-bonded closure."""
        ir = parse_cgsmiles("{[#A].1[#B][#C]1}")
        assert len(ir.base_graph.bonds) == 3


# =====================================================================
# §2.6  Branching
# =====================================================================


class TestBranching:
    """CGSmiles branching with parentheses.

    Paper §2.6: "Parentheses denote branches, as in SMILES."
    """

    def test_simple_branch(self):
        """{[#A]([#D])[#B]} — D branches from A."""
        ir = parse_cgsmiles("{[#A]([#D])[#B]}")
        assert len(ir.base_graph.nodes) == 3
        assert len(ir.base_graph.bonds) == 2

    def test_two_branches(self):
        """{[#A]([#B])([#C])[#D]} — B and C branch from A."""
        ir = parse_cgsmiles("{[#A]([#B])([#C])[#D]}")
        assert len(ir.base_graph.nodes) == 4
        assert len(ir.base_graph.bonds) == 3

    def test_branch_chain(self):
        """{[#A]([#B][#C])[#D]} — branch is a chain."""
        ir = parse_cgsmiles("{[#A]([#B][#C])[#D]}")
        assert len(ir.base_graph.nodes) == 4
        assert len(ir.base_graph.bonds) == 3

    def test_branch_with_repeat(self):
        """{[#A]([#B]|3)[#C]} — branch with repeat."""
        ir = parse_cgsmiles("{[#A]([#B]|3)[#C]}")
        assert len(ir.base_graph.nodes) == 5  # A + 3 B + C


# =====================================================================
# §2.7  Fragment Definitions
# =====================================================================


class TestFragmentDefinitions:
    """CGSmiles fragment definitions: {#Name=SMILES_body}.

    Paper §2.7: "Fragment blocks map labels to atomistic structures."
    """

    def test_single_fragment(self):
        """{[#A]}.{#A=CC} — simple fragment."""
        ir = parse_cgsmiles("{[#A]}.{#A=CC}")
        assert len(ir.fragments) == 1
        assert ir.fragments[0].name == "A"

    def test_fragment_with_ports(self):
        """{[#A][#B]}.{#A=[$]CC[$],#B=[$]OO[$]} — with bonding operators."""
        ir = parse_cgsmiles("{[#A][#B]}.{#A=[$]CC[$],#B=[$]OO[$]}")
        assert len(ir.fragments) == 2

    def test_multiple_fragments(self):
        """{[#PS][#PEO]}.{#PS=[$]CC[$](c1ccccc1),#PEO=[$]COC[$]}"""
        ir = parse_cgsmiles("{[#PS][#PEO]}.{#PS=[$]CC[$](c1ccccc1),#PEO=[$]COC[$]}")
        assert len(ir.fragments) == 2
        names = {f.name for f in ir.fragments}
        assert names == {"PS", "PEO"}


# =====================================================================
# §2.8  Multi-Resolution Layering
# =====================================================================


class TestMultiResolution:
    """CGSmiles multi-resolution via chained fragment blocks.

    Paper §2.8: "{CG_graph}.{fragment_mapping}"
    """

    def test_martini_benzene(self):
        """{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]} — 3-bead benzene."""
        ir = parse_cgsmiles("{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]}")
        assert len(ir.base_graph.nodes) == 3
        assert len(ir.fragments) == 1

    def test_polystyrene_5mer(self):
        """{[#PS]|5}.{#PS=[$]CC[$](c1ccccc1)} — 5-unit PS."""
        ir = parse_cgsmiles("{[#PS]|5}.{#PS=[$]CC[$](c1ccccc1)}")
        assert len(ir.base_graph.nodes) == 5
        assert len(ir.fragments) == 1

    def test_cyclohexane_cg(self):
        """{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]} — Martini cyclohexane."""
        ir = parse_cgsmiles("{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}")
        assert len(ir.base_graph.nodes) == 2
        bond = ir.base_graph.bonds[0]
        assert bond.order == 2


# =====================================================================
# Combined Features
# =====================================================================


class TestCombinedFeatures:
    """Complex CGSmiles strings combining multiple features."""

    def test_repeat_with_branch_and_fragment(self):
        """{[#PMA]([#PEG])|3}.{#PMA=[$]CC[$],#PEG=[$]COC[$]}"""
        ir = parse_cgsmiles("{[#PMA]([#PEG])|3}.{#PMA=[$]CC[$],#PEG=[$]COC[$]}")
        # 3 PMA backbone + 3 PEG branches
        assert len(ir.base_graph.nodes) == 6

    def test_annotated_nodes_in_polymer(self):
        """{[#A;q=1][#B;q=-1]|5} — charged alternating polymer."""
        ir = parse_cgsmiles("{[#A;q=1][#B;q=-1]|5}")
        charged = [n for n in ir.base_graph.nodes if n.annotations.get("q")]
        assert len(charged) > 0

    def test_ring_with_branches(self):
        """{[#A]1([#D])[#B][#C]1} — ring with branch."""
        ir = parse_cgsmiles("{[#A]1([#D])[#B][#C]1}")
        assert len(ir.base_graph.nodes) == 4

    def test_empty_graph(self):
        """{} — empty graph."""
        ir = parse_cgsmiles("{}")
        assert len(ir.base_graph.nodes) == 0
