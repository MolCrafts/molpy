"""Tests for CGSmiles parser.

Based on test cases from gruenewald-lab/CGsmiles repository.
"""

import pytest

from molpy.parser.smiles import (
    CGSmilesBondIR,
    CGSmilesFragmentIR,
    CGSmilesGraphIR,
    CGSmilesIR,
    CGSmilesNodeIR,
    parse_cgsmiles,
)


def mk_cgsmiles_ir(node_labels, bond_tuples, annotations=None):
    """Helper to build expected CGSmilesGraphIR.

    Args:
        node_labels: List of node labels (strings)
        bond_tuples: List of (i, j, order) tuples
        annotations: Optional dict mapping node index to annotations dict

    Returns:
        CGSmilesGraphIR
    """
    nodes = []
    for i, label in enumerate(node_labels):
        annots = annotations.get(i, {}) if annotations else {}
        nodes.append(CGSmilesNodeIR(label=label, annotations=annots))

    bonds = [
        CGSmilesBondIR(node_i=nodes[i], node_j=nodes[j], order=order)
        for (i, j, order) in bond_tuples
    ]

    return CGSmilesGraphIR(nodes=nodes, bonds=bonds)


# Test data: (cgsmiles_string, expected_node_labels, expected_edges, expected_bond_orders)
basic_sequences = [
    # Simple linear sequence
    (
        "{[#PEO][#PMA][#PEO]}",
        ["PEO", "PMA", "PEO"],
        [(0, 1), (1, 2)],
        [1, 1],
    ),
    # Linear sequence with bond orders
    (
        "{[#PEO]=[#PMA]-[#PEO]}",
        ["PEO", "PMA", "PEO"],
        [(0, 1), (1, 2)],
        [2, 1],
    ),
    # Linear sequence with no bond
    (
        "{[#PEO].[#PMA][#PEO]}",
        ["PEO", "PMA", "PEO"],
        [(0, 1), (1, 2)],
        [0, 1],
    ),
    # All bond types
    (
        "{[#A]-[#B]=[#C]#[#D]$[#E].[#F]}",
        ["A", "B", "C", "D", "E", "F"],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        [1, 2, 3, 4, 0],
    ),
]

repeat_operator_tests = [
    # Simple linear sequence with expansion
    (
        "{[#PMA]|3}",
        ["PMA", "PMA", "PMA"],
        [(0, 1), (1, 2)],
        [1, 1],
    ),
    # Expansion with bond order
    (
        "{[#PMA]|3=[#PEO]}",
        ["PMA", "PMA", "PMA", "PEO"],
        [(0, 1), (1, 2), (2, 3)],
        [1, 1, 2],
    ),
]

ring_closure_tests = [
    # Simple cycle sequence
    (
        "{[#PMA]1[#PEO][#PMA]1}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (1, 2), (0, 2)],
        [1, 1, 1],
    ),
    # Simple cycle with bond order to next
    (
        "{[#PMA]1=[#PEO][#PMA]1}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (1, 2), (0, 2)],
        [2, 1, 1],
    ),
    # Simple cycle with bond order in cycle
    (
        "{[#PMA]=1[#PEO][#PMA]1}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (1, 2), (0, 2)],
        [1, 1, 2],
    ),
    # Simple cycle with two bond orders
    (
        "{[#PMA].1=[#PEO][#PMA]1}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (1, 2), (0, 2)],
        [2, 1, 0],
    ),
    # Complex cycle with two rings
    (
        "{[#PMA]1[#PEO]2[#PMA]1[#PEO]2}",
        ["PMA", "PEO", "PMA", "PEO"],
        [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)],
        [1, 1, 1, 1, 1],
    ),
]

branch_tests = [
    # Simple branch
    (
        "{[#PMA]([#PEO])[#PMA]}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (0, 2)],
        [1, 1],
    ),
    # Branch with bond order
    (
        "{[#PMA]([#PEO])=[#PMA]}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (0, 2)],
        [1, 2],
    ),
    # Branch with internal bond order
    (
        "{[#PMA](=[#PEO])[#PMA]}",
        ["PMA", "PEO", "PMA"],
        [(0, 1), (0, 2)],
        [2, 1],
    ),
    # Simple branch expansion
    (
        "{[#PMA]([#PEO][#PEO][#OHter])|3}",
        [
            "PMA",
            "PEO",
            "PEO",
            "OHter",
            "PMA",
            "PEO",
            "PEO",
            "OHter",
            "PMA",
            "PEO",
            "PEO",
            "OHter",
        ],
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (4, 8),
            (8, 9),
            (9, 10),
            (10, 11),
        ],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
    # Branch expansion with bond orders
    (
        "{[#PMA]([#PEO][#PEO]=[#OHter])|3}",
        [
            "PMA",
            "PEO",
            "PEO",
            "OHter",
            "PMA",
            "PEO",
            "PEO",
            "OHter",
            "PMA",
            "PEO",
            "PEO",
            "OHter",
        ],
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (4, 8),
            (8, 9),
            (9, 10),
            (10, 11),
        ],
        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
    ),
    # Nested branching
    (
        "{[#PMA][#PMA]([#PEO][#PEO]([#OH])[#PEO])[#PMA]}",
        ["PMA", "PMA", "PEO", "PEO", "OH", "PEO", "PMA"],
        [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (1, 6)],
        [1, 1, 1, 1, 1, 1],
    ),
    # Nested branching with expansion
    (
        "{[#PMA][#PMA]([#PEO][#PEO]([#OH]|2)[#PEO])[#PMA]}",
        ["PMA", "PMA", "PEO", "PEO", "OH", "OH", "PEO", "PMA"],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (1, 7)],
        [1, 1, 1, 1, 1, 1, 1],
    ),
]

annotation_tests = [
    # Single annotation
    (
        "{[#PEO;q=1]}",
        ["PEO"],
        [],
        [],
        {0: {"q": "1"}},
    ),
    # Multiple annotations
    (
        "{[#PEO;q=1;foo=bar]}",
        ["PEO"],
        [],
        [],
        {0: {"q": "1", "foo": "bar"}},
    ),
    # Annotations on multiple nodes
    (
        "{[#PEO;q=1][#PMA;q=-1][#PEO;q=1]}",
        ["PEO", "PMA", "PEO"],
        [(0, 1), (1, 2)],
        [1, 1],
        {0: {"q": "1"}, 1: {"q": "-1"}, 2: {"q": "1"}},
    ),
]

fragment_tests = [
    # Single fragment
    (
        "{[#A]}.{#A=[$]CC[$]}",
        ["A"],
        [],
        [],
        {},
        [("A", "[$]CC[$]")],
    ),
    # Multiple fragments
    (
        "{[#A][#B]}.{#A=[$]CC[$],#B=[$]O[$]}",
        ["A", "B"],
        [(0, 1)],
        [1],
        {},
        [("A", "[$]CC[$]"), ("B", "[$]O[$]")],
    ),
    # Fragment with nested structure
    (
        "{[#PEO]}.{#PEO=[$]COC[$]}",
        ["PEO"],
        [],
        [],
        {},
        [("PEO", "[$]COC[$]")],
    ),
]


class TestCGSmilesParser:
    """Test suite for CGSmiles parser."""

    @pytest.mark.parametrize(
        "cgsmiles, expected_labels, expected_edges, expected_orders", basic_sequences
    )
    def test_basic_sequences(
        self, cgsmiles, expected_labels, expected_edges, expected_orders
    ):
        """Test basic linear sequences with various bond orders."""
        result = parse_cgsmiles(cgsmiles)

        # Check node count
        assert len(result.base_graph.nodes) == len(expected_labels)

        # Check node labels
        actual_labels = [node.label for node in result.base_graph.nodes]
        assert actual_labels == expected_labels

        # Check bond count
        assert len(result.base_graph.bonds) == len(expected_edges)

        # Check bonds and orders
        for bond, (i, j), expected_order in zip(
            result.base_graph.bonds, expected_edges, expected_orders
        ):
            assert result.base_graph.nodes.index(bond.node_i) == i
            assert result.base_graph.nodes.index(bond.node_j) == j
            assert bond.order == expected_order

    @pytest.mark.parametrize(
        "cgsmiles, expected_labels, expected_edges, expected_orders",
        repeat_operator_tests,
    )
    def test_repeat_operators(
        self, cgsmiles, expected_labels, expected_edges, expected_orders
    ):
        """Test repeat operator |INT."""
        result = parse_cgsmiles(cgsmiles)

        assert len(result.base_graph.nodes) == len(expected_labels)
        actual_labels = [node.label for node in result.base_graph.nodes]
        assert actual_labels == expected_labels

        assert len(result.base_graph.bonds) == len(expected_edges)
        for bond, (i, j), expected_order in zip(
            result.base_graph.bonds, expected_edges, expected_orders
        ):
            assert result.base_graph.nodes.index(bond.node_i) == i
            assert result.base_graph.nodes.index(bond.node_j) == j
            assert bond.order == expected_order

    @pytest.mark.parametrize(
        "cgsmiles, expected_labels, expected_edges, expected_orders", ring_closure_tests
    )
    def test_ring_closures(
        self, cgsmiles, expected_labels, expected_edges, expected_orders
    ):
        """Test ring closure syntax."""
        result = parse_cgsmiles(cgsmiles)

        assert len(result.base_graph.nodes) == len(expected_labels)
        actual_labels = [node.label for node in result.base_graph.nodes]
        assert actual_labels == expected_labels

        assert len(result.base_graph.bonds) == len(expected_edges)

        # Check that all expected edges exist (order may vary)
        actual_edges = set()
        bond_orders = {}
        for bond in result.base_graph.bonds:
            i = result.base_graph.nodes.index(bond.node_i)
            j = result.base_graph.nodes.index(bond.node_j)
            edge = tuple(sorted([i, j]))
            actual_edges.add(edge)
            bond_orders[edge] = bond.order

        expected_edges_set = set(tuple(sorted([i, j])) for i, j in expected_edges)
        assert actual_edges == expected_edges_set

        # Check bond orders
        for (i, j), expected_order in zip(expected_edges, expected_orders):
            edge = tuple(sorted([i, j]))
            assert bond_orders[edge] == expected_order

    @pytest.mark.parametrize(
        "cgsmiles, expected_labels, expected_edges, expected_orders", branch_tests
    )
    def test_branches(self, cgsmiles, expected_labels, expected_edges, expected_orders):
        """Test branch syntax and expansion."""
        result = parse_cgsmiles(cgsmiles)

        assert len(result.base_graph.nodes) == len(expected_labels)
        actual_labels = [node.label for node in result.base_graph.nodes]
        assert actual_labels == expected_labels

        assert len(result.base_graph.bonds) == len(expected_edges)

        # For branches, check edges exist (order may vary)
        actual_edges = set()
        bond_orders = {}
        for bond in result.base_graph.bonds:
            i = result.base_graph.nodes.index(bond.node_i)
            j = result.base_graph.nodes.index(bond.node_j)
            edge = tuple(sorted([i, j]))
            actual_edges.add(edge)
            bond_orders[edge] = bond.order

        expected_edges_set = set(tuple(sorted([i, j])) for i, j in expected_edges)
        assert actual_edges == expected_edges_set

    @pytest.mark.parametrize(
        "cgsmiles, expected_labels, expected_edges, expected_orders, expected_annots",
        annotation_tests,
    )
    def test_annotations(
        self,
        cgsmiles,
        expected_labels,
        expected_edges,
        expected_orders,
        expected_annots,
    ):
        """Test node annotations."""
        result = parse_cgsmiles(cgsmiles)

        assert len(result.base_graph.nodes) == len(expected_labels)

        # Check annotations
        for i, node in enumerate(result.base_graph.nodes):
            if i in expected_annots:
                assert node.annotations == expected_annots[i]
            else:
                assert node.annotations == {}

    @pytest.mark.parametrize(
        "cgsmiles, expected_labels, expected_edges, expected_orders, expected_annots, expected_fragments",
        fragment_tests,
    )
    def test_fragments(
        self,
        cgsmiles,
        expected_labels,
        expected_edges,
        expected_orders,
        expected_annots,
        expected_fragments,
    ):
        """Test fragment definitions."""
        result = parse_cgsmiles(cgsmiles)

        # Check base graph
        assert len(result.base_graph.nodes) == len(expected_labels)

        # Check fragments
        assert len(result.fragments) == len(expected_fragments)
        for fragment, (expected_name, expected_body) in zip(
            result.fragments, expected_fragments
        ):
            assert fragment.name == expected_name
            assert fragment.body == expected_body

    def test_unclosed_ring_error(self):
        """Test that unclosed rings raise an error."""
        with pytest.raises(ValueError, match="Unclosed ring"):
            parse_cgsmiles("{[#PMA]1[#PEO]}")

    def test_empty_graph(self):
        """Test parsing empty graph."""
        result = parse_cgsmiles("{}")
        assert len(result.base_graph.nodes) == 0
        assert len(result.base_graph.bonds) == 0

    def test_complex_nested_structure(self):
        """Test complex nested branching with expansion."""
        cgsmiles = "{[#PMA][#PMA]([#PEO][#PQ]([#OH])|3[#PEO])[#PMA]}"
        result = parse_cgsmiles(cgsmiles)

        expected_labels = [
            "PMA",
            "PMA",
            "PEO",
            "PQ",
            "OH",
            "PQ",
            "OH",
            "PQ",
            "OH",
            "PEO",
            "PMA",
        ]
        actual_labels = [node.label for node in result.base_graph.nodes]
        assert actual_labels == expected_labels

        # Should have 10 bonds
        assert len(result.base_graph.bonds) == 10
