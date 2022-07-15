# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-15
# version: 0.0.1

import networkx as nx
import molpy.algorithms.isomorphvf2 as iso
from molpy.topo import Graph, Topo

import os
import random
import struct

class TestIsomorph:

    def test_with_molpy_topo(self):

        G1 = Graph()
        G2 = Graph()
        G1.from_networkx(nx.path_graph(4))
        G2.from_networkx(nx.path_graph(4))
        GM = iso.GraphMatcher(G1, G2)
        assert GM.is_isomorphic()

class TestIsomorph:

    @classmethod
    def setup_class(cls):
        cls.G1 = Graph()
        cls.G2 = Graph()
        cls.G3 = Graph()
        cls.G4 = Graph()
        cls.G5 = Graph()
        cls.G6 = Graph()
        cls.G1.add_bonds([[1, 2], [1, 3], [1, 5], [2, 3]])
        cls.G2.add_bonds([[10, 20], [20, 30], [10, 30], [10, 50]])
        cls.G3.add_bonds([[1, 2], [1, 3], [1, 5], [2, 5]])
        cls.G4.add_bonds([[1, 2], [1, 3], [1, 5], [2, 4]])
        cls.G5.add_bonds([[1, 2], [1, 3]])
        cls.G6.add_bonds([[10, 20], [20, 30], [10, 30], [10, 50], [20, 50]])

    def test_faster_could_be_isomorphic(self):
        assert iso.faster_could_be_isomorphic(self.G3, self.G2)
        assert not iso.faster_could_be_isomorphic(self.G3, self.G5)
        assert not iso.faster_could_be_isomorphic(self.G1, self.G6)

    def test_is_isomorphic(self):
        assert iso.is_isomorphic(self.G1, self.G2)
        assert not iso.is_isomorphic(self.G1, self.G4)

class TestWikipediaExample:
    # Source: https://en.wikipedia.org/wiki/Graph_isomorphism

    # Nodes 'a', 'b', 'c' and 'd' form a column.
    # Nodes 'g', 'h', 'i' and 'j' form a column.
    g1edges = [
        ["a", "g"],
        ["a", "h"],
        ["a", "i"],
        ["b", "g"],
        ["b", "h"],
        ["b", "j"],
        ["c", "g"],
        ["c", "i"],
        ["c", "j"],
        ["d", "h"],
        ["d", "i"],
        ["d", "j"],
    ]

    # Nodes 1,2,3,4 form the clockwise corners of a large square.
    # Nodes 5,6,7,8 form the clockwise corners of a small square
    g2edges = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 5],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
    ]

    def test_graph(self):
        g1 = Graph()
        g2 = Graph()
        g1.add_bonds(self.g1edges)
        g2.add_bonds(self.g2edges)
        gm = iso.GraphMatcher(g1, g2)
        assert gm.is_isomorphic()
        # Just testing some cases
        assert gm.subgraph_is_monomorphic()

        mapping = sorted(gm.mapping.items())

    # this mapping is only one of the possibilies
    # so this test needs to be reconsidered
    #        isomap = [('a', 1), ('b', 6), ('c', 3), ('d', 8),
    #                  ('g', 2), ('h', 5), ('i', 4), ('j', 7)]
    #        assert_equal(mapping, isomap)

    def test_subgraph(self):
        g1 = Graph()
        g2 = Graph()
        g1.add_bonds(self.g1edges)
        g2.add_bonds(self.g2edges)
        g3 = g2.subgraph([1, 2, 3, 4])
        gm = iso.GraphMatcher(g1, g3)
        assert gm.subgraph_is_isomorphic()

    def test_subgraph_mono(self):
        g1 = Graph()
        g2 = Graph()
        g1.add_bonds(self.g1edges)
        g2.add_bonds([[1, 2], [2, 3], [3, 4]])
        gm = iso.GraphMatcher(g1, g2)
        assert gm.subgraph_is_monomorphic()


class TestVF2GraphDB:
    # https://web.archive.org/web/20090303210205/http://amalfi.dis.unina.it/graph/db/

    @staticmethod
    def create_graph(filename):
        """Creates a Graph instance from the filename."""

        # The file is assumed to be in the format from the VF2 graph database.
        # Each file is composed of 16-bit numbers (unsigned short int).
        # So we will want to read 2 bytes at a time.

        # We can read the number as follows:
        #   number = struct.unpack('<H', file.read(2))
        # This says, expect the data in little-endian encoding
        # as an unsigned short int and unpack 2 bytes from the file.

        fh = open(filename, mode="rb")

        # Grab the number of nodes.
        # Node numeration is 0-based, so the first node has index 0.
        nodes = struct.unpack("<H", fh.read(2))[0]

        graph = Graph()
        for from_node in range(nodes):
            # Get the number of edges.
            edges = struct.unpack("<H", fh.read(2))[0]
            for edge in range(edges):
                # Get the terminal node.
                to_node = struct.unpack("<H", fh.read(2))[0]
                graph.add_bond(from_node, to_node)

        fh.close()
        return graph

    def test_graph(self):
        head, tail = os.path.split(__file__)
        head = head + '/data'
        g1 = self.create_graph(os.path.join(head, "iso_r01_s80.A99"))
        g2 = self.create_graph(os.path.join(head, "iso_r01_s80.B99"))
        gm = iso.GraphMatcher(g1, g2)
        assert gm.is_isomorphic()

    def test_subgraph(self):
        # A is the subgraph
        # B is the full graph
        head, tail = os.path.split(__file__)
        head = head + '/data'
        subgraph = self.create_graph(os.path.join(head, "si2_b06_m200.A99"))
        graph = self.create_graph(os.path.join(head, "si2_b06_m200.B99"))
        gm = iso.GraphMatcher(graph, subgraph)
        assert gm.subgraph_is_isomorphic()
        # Just testing some cases
        assert gm.subgraph_is_monomorphic()