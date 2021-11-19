# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
import networkx as nx
from molpy import fromNetworkXGraph

from molpy.group import Group

class TestGroup:
        
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
        assert len(C6.atoms) == 12
            
    def test_setTopoByCovalentMap(self, CH4):

        covalentMap = np.zeros((CH4.natoms, CH4.natoms), dtype=int)
        covalentMap[0, 1:] = covalentMap[1:, 0] = 1
        CH4.setTopoByCovalentMap(covalentMap)
        assert len(CH4['C'].bondedAtoms) == 4
        assert CH4['C'] in CH4['H0'].bondedAtoms
        assert CH4.nbonds == 4
        
    def test_getCovalentMap(self, CH4):
        co = CH4.getCovalentMap()
        assert co[0, 0] == 0
        assert all(co[0, 1:] == 1)
        assert all(co[1:, 0] == 1)
        
    def test_getRingCovalentMap(self, C6):
        assert C6.getCovalentMap().shape == (12, 12)
        
    def test_setRingTopoByCovalentMap(self, C6):
        atoms = C6.getAtoms()
        assert len(atoms[0].bondedAtoms) == 3
        assert len(atoms[1].bondedAtoms) == 3
        assert len(atoms[2].bondedAtoms) == 3
        assert len(atoms[3].bondedAtoms) == 3
        
    def test_getRingBonds(self, C6):
        bonds = C6.getBonds()
        assert len(bonds) == 12
        assert len(C6._bondList) == 12

        
    def test_getBonds(self, CH4):
        bonds = CH4.getBonds()
        assert len(bonds) == 4
        assert len(CH4._bondList) == 4

        
    def test_getSubGroup(self, CH4):
        H4 = CH4.getSubGroup('H4', [CH4[f'H{i}'] for i in range(4)])
        assert H4.natoms == 4
        assert H4.nbonds == 0
        
        CH = CH4.getSubGroup('CH', [CH4['C'], CH4['H0']])
        assert CH.natoms == 2
        assert CH.nbonds == 1
    
    def test_getBasisCycles(self, C6):
        
        assert len(C6.getBasisCycles()) == 1

    def test_copy(self, CH4):
        
        CH4new = CH4.copy()
        assert CH4new.atoms[0].nbondedAtoms == CH4.atoms[0].nbondedAtoms
        
        assert CH4new.natoms == CH4.natoms
        assert CH4new.nbonds == CH4.nbonds
        assert CH4new.uuid != CH4.uuid
        assert CH4new.getAtomByName('C').uuid != CH4.getAtomByName('C').uuid
        
    def test_removeAtom(self, CH4):
        ch4 = CH4()
        ch4.removeAtom(ch4.atoms[1])
        assert ch4.natoms == 4
        assert ch4.nbonds == 3
        
class TestGroupTopo:
    
    @classmethod
    def setup_class(cls):
        cls.linear5 = fromNetworkXGraph('linear5', nx.path_graph(5))
        cls.K5 = fromNetworkXGraph('K5', nx.complete_graph(5))
        cls.ring3 = fromNetworkXGraph('ring3', nx.cycle_graph(3))
        cls.ring4 = fromNetworkXGraph('ring4', nx.cycle_graph(4))
        assert cls.K5.natoms == 5
        assert cls.ring3.natoms == 3
        assert cls.ring3.nbonds == 3
    
    def testSerachAngle(self):
        assert len(self.linear5.searchAngles()) == 3
        assert len(self.K5.searchAngles()) == 30
        assert len(self.ring3.searchAngles()) == 3
        assert len(self.ring4.searchAngles()) == 4
        
    def testSearchDihedral(self):
        assert len(self.linear5.searchDihedrals()) == 2
        assert len(self.ring3.searchDihedrals()) == 0
        assert len(self.ring4.searchDihedrals()) == 4
        assert len(self.K5.searchDihedrals()) == 60
        
class TestInterGroup:
    
    @pytest.fixture(scope='class')
    def CH4(self):
        CH4 = mp.Group('CH4')
        CH4.addAtoms([mp.Atom(f'H{i}') for i in range(4)])
        CH4.addAtom(mp.Atom('C'))
        CH4.addBondByName('C', 'H0')  #       H3
        CH4.addBondByName('C', 'H1')  #       |
        CH4.addBondByName('C', 'H2')  #  H0 - C - H2
        CH4.addBondByName('C', 'H3')  #       |
        yield CH4                     #       H1
        
    def test_addition(self, CH4):
        
        ch41 = CH4.copy()
        ch42 = CH4.copy()
        ch41.reacto(ch42, method='addition', atomName='H2', btomName='H0', copy=True)
        assert ch41.nbonds == 5
        
    def test_condensation(self, CH4):
        
        ch41 = CH4.copy()
        ch42 = CH4.copy()
        ch41.reacto(ch42, method='concentration', atomName='H2', btomName='H0', copy=True)
        assert ch41.natoms == 4
        assert ch41.nbonds == 4
                