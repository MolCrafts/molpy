# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-26
# version: 0.0.1
# source: networkx/algorithms/traversal/tests/test_dfs.py

import pytest
from molpy.group import Group
from molpy.atom import Atom
import molpy

class TestDFS:
    
    @classmethod
    def setup_class(cls):
        # simple graph
        G = Group('connected')
        atoms = [Atom(f'G{i}') for i in range(5)]
        cls.G0, cls.G1, cls.G2, cls.G3, cls.G4 = atoms
        G.addAtoms(atoms)
        G.addBondByIndex(0, 1)
        G.addBondByIndex(1, 2)
        G.addBondByIndex(1, 3)
        G.addBondByIndex(2, 4)
        G.addBondByIndex(3, 4)
        cls.G = G
        
        # simple graph, disconnected
        D = Group('disconnected')
        atoms = [Atom(f'D{i}') for i in range(4)]
        cls.D0, cls.D1, cls.D2, cls.D3 = atoms
        D.addAtoms(atoms)
        D.addBondByIndex(0, 1)
        D.addBondByIndex(2, 3)
        cls.D = D
        
    def test_dfs_edges(self):
        edges = molpy.dfs_edges(self.G, source=self.G0)
        assert [ (edge[0], edge[1]) for edge in edges] == [(self.G0, self.G1), (self.G1, self.G2), (self.G2, self.G4), (self.G4, self.G3)]
        edges = molpy.dfs_edges(self.D)
        assert  [ (edge[0], edge[1]) for edge in edges] == [(self.D0, self.D1), (self.D2, self.D3)]

    def test_dfs_tree(self):
        exp_atoms = sorted(self.G.atoms)
        exp_bonds = sorted([self.G.getBondByIndex(0, 1), self.G.getBondByIndex(1, 2),
                            self.G.getBondByIndex(2, 4), self.G.getBondByIndex(4, 3)])
        # Search from first node
        T = molpy.dfs_tree(self.G, source=self.G0)
        assert sorted(T.atoms) == exp_atoms
        assert len(T.bonds) == len(exp_bonds)
        assert sorted(T.bonds) == exp_bonds
        # Check source=None
        T = molpy.dfs_tree(self.G, source=None)
        assert sorted(T.atoms) == exp_atoms
        assert sorted(T.bonds) == exp_bonds
        # Check source=None is the default
        T = molpy.dfs_tree(self.G)
        assert sorted(T.atoms) == exp_atoms
        assert sorted(T.bonds) == exp_bonds
        
    def test_preorder_nodes(self):
        assert list(molpy.dfs_preorder_nodes(self.G, source=self.G0)) == [self.G0, self.G1, self.G2, self.G4, self.G3]
        assert list(molpy.dfs_preorder_nodes(self.D)) == [self.D0, self.D1, self.D2, self.D3]

    def test_postorder_nodes(self):
        assert list(molpy.dfs_postorder_nodes(self.G, source=self.G0)) == [self.G3, self.G4, self.G2, self.G1, self.G0]
        assert list(molpy.dfs_postorder_nodes(self.D)) == [self.D1, self.D0, self.D3, self.D2]

    def test_successor(self):
        assert molpy.dfs_successors(self.G, source=self.G0) == {self.G0: [self.G1], self.G1: [self.G2], self.G2: [self.G4], self.G4: [self.G3]}
        assert molpy.dfs_successors(self.D) == {self.D0: [self.D1], self.D2: [self.D3]}

    def test_predecessor(self):
        assert molpy.dfs_predecessors(self.G, source=self.G0) == {self.G1: self.G0, self.G2: self.G1, self.G3: self.G4, self.G4: self.G2}
        print(molpy.dfs_predecessors(self.D))
        assert molpy.dfs_predecessors(self.D) == {self.D3: self.D2, self.D1: self.D0}


    def test_dfs_labeled_edges(self):
        edges = list(molpy.dfs_labeled_edges(self.G, source=self.G0))
        forward = [(u, v) for (u, v, d) in edges if d == "forward"]
        assert forward == [(self.G0, self.G0), (self.G0, self.G1), (self.G1, self.G2), (self.G2, self.G4), (self.G4, self.G3)]

    def test_dfs_labeled_disconnected_edges(self):
        edges = list(molpy.dfs_labeled_edges(self.D))
        forward = [(u, v) for (u, v, d) in edges if d == "forward"]
        assert forward == [(self.D0, self.D0), (self.D0, self.D1), (self.D2, self.D2), (self.D2, self.D3)]

    def test_dfs_tree_isolates(self):
        G = molpy.Group('G')
        G1 = Atom('G1')
        G.addAtom(G1)
        G2 = Atom('G2')
        G.addAtom(G2)
        T = molpy.dfs_tree(G, source=G1)
        assert sorted(T.atoms) == [G1]
        assert sorted(T.bonds) == []
        T = molpy.dfs_tree(G, source=None)
        assert sorted(T.atoms) == sorted([G1, G2])
        assert sorted(T.bonds) == []


class TestDepthLimitedSearch:
    
    @classmethod
    def setup_class(cls):
        # a tree
        G = molpy.Group('G')
        atoms = [Atom(f'G{i}') for i in range(11)]
        cls.G0, cls.G1, cls.G2, cls.G3, cls.G4, cls.G5, cls.G6, cls.G7, cls.G8, cls.G9, cls.G10 = atoms
        G.addAtoms(atoms)
        G.addBondByIndex(0, 1)
        G.addBondByIndex(1, 2)
        G.addBondByIndex(2, 3)
        G.addBondByIndex(3, 4)
        G.addBondByIndex(4, 5)
        G.addBondByIndex(5, 6)
        G.addBondByIndex(2, 7)
        G.addBondByIndex(7, 8)
        G.addBondByIndex(8, 9)
        G.addBondByIndex(9, 10)       
        cls.G = G
        # a disconnected graph
        D = molpy.Group('D')
        atoms = [Atom(f'D{i}') for i in range(4)]
        cls.D0, cls.D1, cls.D2, cls.D3 = atoms
        D.addAtoms(atoms)
        atoms = [Atom(f'D{i}') for i in range(7, 11)]
        cls.D7, cls.D8, cls.D9, cls.D10 = atoms
        D.addAtoms(atoms)
        D.addBondByIndex(0, 1)
        D.addBondByIndex(2, 3)
        D.addBondByIndex(2, 4)
        D.addBondByIndex(4, 5)
        D.addBondByIndex(5, 6) 
        D.addBondByIndex(6, 7) 
        cls.D = D

    def test_dls_preorder_nodes(self):
        assert list(molpy.dfs_preorder_nodes(self.G, source=self.G0, depth_limit=2)) == [self.G0, self.G1, self.G2]
        assert list(molpy.dfs_preorder_nodes(self.D, source=self.D1, depth_limit=2)) == ([self.D1, self.D0])

    def test_dls_postorder_nodes(self):
        assert list(molpy.dfs_postorder_nodes(self.G, source=self.G3, depth_limit=3)) == [
            self.G1,
            self.G7,
            self.G2,
            self.G5,
            self.G4,
            self.G3,
        ]
        assert list(molpy.dfs_postorder_nodes(self.D, source=self.D2, depth_limit=2)) == (
            [self.D3, self.D7, self.D2]
        )

    def test_dls_successor(self):
        result = molpy.dfs_successors(self.G, source=self.G4, depth_limit=3)
        assert {n: set(v) for n, v in result.items()} == {
            self.G2: {self.G1, self.G7},
            self.G3: {self.G2},
            self.G4: {self.G3, self.G5},
            self.G5: {self.G6},
        }
        result = molpy.dfs_successors(self.D, source=self.D7, depth_limit=2)
        assert {n: set(v) for n, v in result.items()} == {self.D8: {self.D9}, self.D2: {self.D3}, self.D7: {self.D8, self.D2}}

    def test_dls_predecessor(self):
        assert molpy.dfs_predecessors(self.G, source=self.G0, depth_limit=3) == {
            self.G1: self.G0,
            self.G2: self.G1,
            self.G3: self.G2,
            self.G7: self.G2,
        }
        assert molpy.dfs_predecessors(self.D, source=self.D2, depth_limit=3) == {
            self.D8: self.D7,
            self.D9: self.D8,
            self.D3: self.D2,
            self.D7: self.D2,
        }

    def test_dls_tree(self):
        T = molpy.dfs_tree(self.G, source=self.G3, depth_limit=1)
        assert sorted(T.bonds) == sorted([self.G.getBondByIndex(3, 2), self.G.getBondByIndex(3, 4)])

    def test_dls_edges(self):
        edges = molpy.dfs_edges(self.G, source=self.G9, depth_limit=4)
        assert list(edges) == [(self.G9, self.G8), (self.G8, self.G7), (self.G7, self.G2), (self.G2, self.G1), (self.G2, self.G3), (self.G9, self.G10)]

    def test_dls_labeled_edges(self):
        edges = list(molpy.dfs_labeled_edges(self.G, source=self.G5, depth_limit=1))
        forward = [(u, v) for (u, v, d) in edges if d == "forward"]
        assert forward == [(self.G5, self.G5), (self.G5, self.G4), (self.G5, self.G6)]

    def test_dls_labeled_disconnected_edges(self):
        edges = list(molpy.dfs_labeled_edges(self.G, source=self.G6, depth_limit=2))
        forward = [(u, v) for (u, v, d) in edges if d == "forward"]
        assert forward == [(self.G6, self.G6), (self.G6, self.G5), (self.G5, self.G4)]