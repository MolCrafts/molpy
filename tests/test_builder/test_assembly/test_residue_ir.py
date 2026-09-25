"""Unit tests for residue-topology IR dataclasses."""

import pytest

from molpy.builder.assembly._residue_ir import (
    ResidueBond,
    ResidueTopology,
    ResidueNode,
)


class TestResidueNode:
    def test_identity_is_by_id(self):
        a = ResidueNode(label="EO")
        b = ResidueNode(label="EO")
        assert a != b
        assert hash(a) == a.id

    def test_first_positional_argument_is_the_label(self):
        node = ResidueNode("EO")
        assert node.label == "EO"
        assert isinstance(node.id, int)


class TestResidueBond:
    def test_links_two_nodes(self):
        a = ResidueNode(label="A")
        b = ResidueNode(label="B")
        bond = ResidueBond(node_i=a, node_j=b)
        assert bond.node_i is a
        assert bond.node_j is b

    def test_a_bond_equals_itself(self):
        bond = ResidueBond(node_i=ResidueNode(label="A"), node_j=ResidueNode(label="B"))
        assert bond == bond

    def test_bonds_with_different_endpoints_are_not_equal(self):
        a = ResidueNode(label="A")
        b = ResidueNode(label="B")
        c = ResidueNode(label="C")
        assert ResidueBond(node_i=a, node_j=b) != ResidueBond(node_i=b, node_j=c)

    def test_membership_in_a_bond_list_is_that_bond_only(self):
        a = ResidueNode(label="A")
        b = ResidueNode(label="B")
        c = ResidueNode(label="C")
        ab = ResidueBond(node_i=a, node_j=b)
        bc = ResidueBond(node_i=b, node_j=c)
        topology = ResidueTopology(nodes=[a, b, c], bonds=[ab])
        assert ab in topology.bonds
        assert bc not in topology.bonds


class TestResidueTopology:
    def test_empty_graph(self):
        g = ResidueTopology()
        assert g.nodes == []
        assert g.bonds == []

    def test_same_nodes_different_bonds_are_not_equal(self):
        a = ResidueNode(label="A")
        b = ResidueNode(label="B")
        c = ResidueNode(label="C")
        path_abc = ResidueTopology(
            nodes=[a, b, c],
            bonds=[ResidueBond(node_i=a, node_j=b), ResidueBond(node_i=b, node_j=c)],
        )
        path_bac = ResidueTopology(
            nodes=[a, b, c],
            bonds=[ResidueBond(node_i=b, node_j=a), ResidueBond(node_i=a, node_j=c)],
        )
        assert path_abc != path_bac

    def test_bond_to_a_node_outside_the_graph_is_rejected(self):
        a = ResidueNode(label="A")
        stranger = ResidueNode(label="B")
        with pytest.raises(ValueError):
            ResidueTopology(nodes=[a], bonds=[ResidueBond(node_i=a, node_j=stranger)])

    def test_self_bond_is_rejected(self):
        a = ResidueNode(label="A")
        with pytest.raises(ValueError):
            ResidueTopology(nodes=[a], bonds=[ResidueBond(node_i=a, node_j=a)])

    def test_empty_label_is_rejected(self):
        with pytest.raises(ValueError):
            ResidueTopology(nodes=[ResidueNode(label="")])

    def test_duplicate_node_ids_are_rejected(self):
        a = ResidueNode(label="A")
        with pytest.raises(ValueError):
            ResidueTopology(nodes=[a, a])
