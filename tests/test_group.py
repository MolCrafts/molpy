# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

from molpy.group import Group

class TestGroup:
    
    @pytest.fixture(scope='class')
    def CH4(self):
        CH4 = mp.Group('CH4')
        C = mp.Atom('C')
        Hs = [mp.Atom(f'H{i}') for i in range(4)]
        CH4.add(C)
        for H in Hs:
            CH4.add(H)
        yield CH4
        
    @pytest.fixture(scope='class')
    def C6(self):
        C6 = mp.Group('C6')
        for C in [mp.Atom(f'C{i}') for i in range(6)]:
            C6.add(C)
        covalentMap = np.array([[0, 1, 2, 3, 2, 1],
                                [1, 0, 1, 2, 3, 2],
                                [2, 1, 0, 1, 2, 3],
                                [3, 2, 1, 0, 1, 2],
                                [2, 3, 2, 1, 0, 1],
                                [1, 2, 3, 2, 1, 0]], dtype=int)
        C6.setTopoByCovalentMap(covalentMap)
        C6.reference_covalentMap = covalentMap
        yield C6
        
    @pytest.fixture(scope='class')
    def K3(self):
        
        
        k0, k1, k2 = k3nodes = [mp.Atom(f'k{i}') for i in range(3)]
        
        K3 = Group('K3')
        K3.addAtoms(k3nodes)
        K3.addBond(k0, k1)
        K3.addBond(k0, k2)
        K3.addBond(k1, k2)

        yield K3, k3nodes     
        
    def test_atoms(self, CH4, C6):
        assert len(CH4.atoms) == 5
        assert len(C6.atoms) == 6
            
    def test_setTopoByCovalentMap(self, CH4):
        covalentMap = np.zeros((CH4.natoms, CH4.natoms), dtype=int)
        covalentMap[0, 1:] = covalentMap[1:, 0] = 1
        CH4.setTopoByCovalentMap(covalentMap)
        assert len(CH4['C'].bondedAtoms) == 4
        assert CH4['C'] in CH4['H0'].bondedAtoms
        
    def test_getCovalentMap(self, CH4):
        co = CH4.getCovalentMap()
        print(co)
        assert co[1][0, 0] == 0
        assert all(co[1][0, 1:] == 1)
        assert all(co[1][1:, 0] == 1)
        
    def test_getRingCovalentMap(self, C6):
        assert(C6.reference_covalentMap == C6.getCovalentMap()[1]).all()
        
    def test_setRingTopoByCovalentMap(self, C6):
        atoms = C6.getAtoms()
        assert len(atoms[0].bondedAtoms) == 2
        assert len(atoms[1].bondedAtoms) == 2
        assert len(atoms[2].bondedAtoms) == 2
        assert len(atoms[3].bondedAtoms) == 2
        
    def test_getRingBonds(self, C6):
        bonds = C6.getBonds()
        assert len(bonds) == 6
        
    def test_getBonds(self, CH4):
        bonds = CH4.getBonds()
        assert len(bonds) == 4
        
    def test_getRingAngles(self, C6):
        angles = C6.getAngles()
        assert len(angles) == 6
        
    def test_getAngles(self, CH4):
        angles = CH4.getAngles()
        assert len(angles) == 6
        
    def test_getSubGroup(self, CH4):
        H4 = CH4.getSubGroup('H4', [CH4[f'H{i}'] for i in range(4)])
        assert H4.natoms == 4
        assert H4.nbonds == 0
        
        CH = CH4.getSubGroup('CH', [CH4['C'], CH4['H0']])
        assert CH.natoms == 2
        assert CH.nbonds == 1
    
    def test_getBasisCycles(self, C6):
        
        assert len(C6.getBasisCycles()) == 1
        
    # def test_nbunch_iter(self, K3):
    #     G, k3nodes = K3
    #     k0, k1, k2 = k3nodes
    #     assert G.atomsEqual(k3nodes)
        
    #     assert nodes_equal(G.nbunch_iter(), self.k3nodes)  # all nodes
    #     assert nodes_equal(G.nbunch_iter(0), [0])  # single node
    #     assert nodes_equal(G.nbunch_iter([0, 1]), [0, 1])  # sequence
    #     # sequence with none in graph
    #     assert nodes_equal(G.nbunch_iter([-1]), [])
    #     # string sequence with none in graph
    #     assert nodes_equal(G.nbunch_iter("foo"), [])
    #     # node not in graph doesn't get caught upon creation of iterator
    #     bunch = G.nbunch_iter(-1)
    #     # but gets caught when iterator used
    #     with pytest.raises(nx.NetworkXError, match="is not a node or a sequence"):
    #         list(bunch)
    #     # unhashable doesn't get caught upon creation of iterator
    #     bunch = G.nbunch_iter([0, 1, 2, {}])
    #     # but gets caught when iterator hits the unhashable
    #     with pytest.raises(
    #         nx.NetworkXError, match="in sequence nbunch is not a valid node"
    #     ):
    #         list(bunch)

    # def test_nbunch_iter_node_format_raise(self):
    #     # Tests that a node that would have failed string formatting
    #     # doesn't cause an error when attempting to raise a
    #     # :exc:`nx.NetworkXError`.

    #     # For more information, see pull request #1813.
    #     G = self.Graph()
    #     nbunch = [("x", set())]
    #     with pytest.raises(nx.NetworkXError):
    #         list(G.nbunch_iter(nbunch))

    def test_copy(self, CH4):
        
        CH4new = CH4.copy()
        assert CH4new.natoms == CH4.natoms
        assert CH4new.nbonds == CH4.nbonds
        assert CH4new.uuid != CH4.uuid
        assert CH4new.getAtomByName('C').uuid != CH4.getAtomByName('C').uuid