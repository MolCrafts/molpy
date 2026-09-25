"""Unit tests for residue-topology IR dataclasses."""

from molpy.builder.assembly._cgsmiles_ir import (
    CGSmilesBondIR,
    CGSmilesGraphIR,
    CGSmilesNodeIR,
)


class TestCGSmilesNodeIR:
    def test_identity_is_by_id(self):
        a = CGSmilesNodeIR(label="EO")
        b = CGSmilesNodeIR(label="EO")
        assert a != b
        assert hash(a) == a.id


class TestCGSmilesBondIR:
    def test_links_two_nodes(self):
        a = CGSmilesNodeIR(label="A")
        b = CGSmilesNodeIR(label="B")
        bond = CGSmilesBondIR(node_i=a, node_j=b)
        assert bond.node_i is a
        assert bond.node_j is b


class TestCGSmilesGraphIR:
    def test_empty_graph(self):
        g = CGSmilesGraphIR()
        assert g.nodes == []
        assert g.bonds == []
