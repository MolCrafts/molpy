#!/usr/bin/env python3
"""Unit tests for adapter functions.

Tests cover:
- build_mol_graph
- _extract_atom_attributes
- _extract_bond_attributes
- _compute_derived_properties
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.typifier.adapter import build_mol_graph


class TestBuildMolGraph:
    """Test build_mol_graph function."""

    def test_build_mol_graph_simple(self):
        """Test building graph from simple structure."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        graph, vs_to_atomid, atomid_to_vs = build_mol_graph(asm)

        assert graph.vcount() == 2
        assert graph.ecount() == 1
        assert len(vs_to_atomid) == 2
        assert len(atomid_to_vs) == 2

    def test_build_mol_graph_vertex_attributes(self):
        """Test that vertex attributes are set correctly."""
        asm = Atomistic()
        c = Atom(symbol="C", element="C")
        h = Atom(symbol="H", element="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        # Check vertex attributes
        for v in graph.vs:
            assert "element" in v.attributes()
            assert "number" in v.attributes()
            assert "charge" in v.attributes()
            assert "degree" in v.attributes()
            assert "in_ring" in v.attributes()

    def test_build_mol_graph_edge_attributes(self):
        """Test that edge attributes are set correctly."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        bond = Bond(c, h, order=1)
        asm.add_link(bond)

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        # Check edge attributes
        for e in graph.es:
            assert "order" in e.attributes()
            assert "is_aromatic" in e.attributes()
            assert "is_in_ring" in e.attributes()

    def test_build_mol_graph_aromatic(self):
        """Test building graph with aromatic atoms."""
        asm = Atomistic()
        c = Atom(symbol="C", is_aromatic=True)
        asm.add_entity(c)

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        assert graph.vs[0]["is_aromatic"] is True

    def test_build_mol_graph_charge(self):
        """Test building graph with charged atoms."""
        asm = Atomistic()
        n = Atom(symbol="N", charge=1)
        asm.add_entity(n)

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        assert graph.vs[0]["charge"] == 1

    def test_build_mol_graph_bond_order(self):
        """Test building graph with different bond orders."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)
        bond = Bond(c1, c2, order=2)
        asm.add_link(bond)

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        assert graph.es[0]["order"] == 2

    def test_build_mol_graph_aromatic_bond(self):
        """Test building graph with aromatic bonds."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)
        bond = Bond(c1, c2, order=":", aromatic=True)
        asm.add_link(bond)

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        assert graph.es[0]["order"] == ":"
        assert graph.es[0]["is_aromatic"] is True

    def test_build_mol_graph_degree(self):
        """Test that degree is computed correctly."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")
        asm.add_entity(c, h1, h2, h3)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        # Find carbon vertex
        c_vs = None
        for i, v in enumerate(graph.vs):
            if v["element"] == "C":
                c_vs = i
                break

        assert c_vs is not None
        assert graph.vs[c_vs]["degree"] == 3

    def test_build_mol_graph_type_error(self):
        """Test that build_mol_graph raises TypeError for non-Atomistic."""
        with pytest.raises(TypeError, match="Expected Atomistic"):
            build_mol_graph("not an atomistic")

    def test_build_mol_graph_empty(self):
        """Test building graph from empty structure."""
        asm = Atomistic()

        graph, vs_to_atomid, atomid_to_vs = build_mol_graph(asm)

        assert graph.vcount() == 0
        assert graph.ecount() == 0
        assert len(vs_to_atomid) == 0
        assert len(atomid_to_vs) == 0

    def test_build_mol_graph_no_bonds(self):
        """Test building graph with atoms but no bonds."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        # No bonds

        graph, _vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        assert graph.vcount() == 2
        assert graph.ecount() == 0
        assert graph.vs[0]["degree"] == 0
        assert graph.vs[1]["degree"] == 0
