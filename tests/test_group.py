# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from copy import deepcopy
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
        
    def test_addAtom(self, particle):
        g = mp.Group('test')
        g.addAtom(particle)
        assert np.array_equal(particle.position, g.atoms[-1].position)
        assert g.atoms[-1].uuid == particle.uuid
        
        g.addAtom(particle())
        assert np.array_equal(particle.position, g.atoms[-1].position)
        assert g.atoms[-1].uuid != particle.uuid
        
    def test_radii(self, CH4):
        radii = CH4.getRadii();
        assert radii[0] == mp.element.COVALENT_RADII["C"]
        assert radii[1] == mp.element.COVALENT_RADII["H"]
        radii[2] = 10
        CH4.setRadii(radii)
        radii = CH4.getRadii();
        assert radii[2] == 10;
        print(mp.element.COVALENT_RADII["C"])
        CH4.setRadii_by(mp.element.COVALENT_RADII)
        radii = CH4.getRadii();
        assert radii[2] == mp.element.COVALENT_RADII["H"];


    def test_atoms(self, CH4, C6):
        assert len(CH4.atoms) == 5
        assert len(C6.atoms) == 12
        symbols = ["C0", "H1", "H2", "H3"]
        names = CH4.getNames()
        with pytest.raises(IndexError):
            CH4.setNames(symbols)
        symbols.append("H4")
        CH4.setNames(symbols)
        assert CH4[0].name == symbols[0]
        assert CH4[3].name == symbols[3]
        CH4.setNames(names)
            
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
        
        for bond in CH4.bonds:
            assert bond.atom in CH4.atoms
        
        ch4 = CH4(name=f'ch4')
        assert ch4.natoms == CH4.natoms
        assert ch4.nbonds == CH4.nbonds
        assert ch4.atoms[-1] != CH4.atoms[-1]
        assert np.array_equal(ch4.atoms[-1].position, CH4.atoms[-1].position)
        for bond in ch4.bonds:
            assert bond.atom in ch4.atoms
        
    def test_removeAtom(self, CH4):
        ch4 = CH4()
        assert ch4.natoms == 5
        assert ch4.nbonds == 4
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
    
    def testSerachAngle(self, CH4):
        assert len(self.linear5.searchAngles()) == 3
        assert len(self.K5.searchAngles()) == 30
        assert len(self.ring3.searchAngles()) == 3
        assert len(self.ring4.searchAngles()) == 4
        assert len(CH4.searchAngles()) == 6
        
    def testSearchDihedral(self):
        assert len(self.linear5.searchDihedrals()) == 2
        assert len(self.ring3.searchDihedrals()) == 0
        assert len(self.ring4.searchDihedrals()) == 4
        assert len(self.K5.searchDihedrals()) == 60
        
    def testAddBondByName(self):
        g = mp.Group('test')
        a = mp.Atom('a')
        b = mp.Atom('b')
        g.addAtoms([a, b])
        g.addBondByName('a', 'b')
        for bond in g.bonds:
            assert bond.atom in g.atoms
            
        gg = g()
        for bond in gg.bonds:
            assert bond.atom in gg.atoms        
        
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
        
        ch41 = CH4()
        ch42 = CH4()
        ch41.reacto(ch42, method='addition', atomName='H2', btomName='H0')
        assert ch41.nbonds == 5
        
    def test_condensation(self, CH4):
        
        ch41 = CH4()
        ch42 = CH4()
        ch41.reacto(ch42, method='concentration', atomName='H2', btomName='H0')
        assert ch41.natoms == 4
        assert ch41.nbonds == 4
                
class TestGroupGeometry:
    
    def test_move(self, H2O):
        opos = H2O.positions
        vec = np.array([1, 1, 1])
        H2O.move(vec)
        npos = H2O.positions
        assert np.array_equal(opos+vec, npos)
        for i in range(5):
            opos = H2O.positions
            newH2O = H2O(name=f'{i}')
            assert id(newH2O) != id(H2O)
            newH2O = newH2O.move(vec)
            npos = newH2O.positions
            print('H2O: ', H2O.positions[0])
            print('new: ', npos[0])
            assert np.array_equal(opos+vec, npos)

class TestGroupCopy:
    
    @pytest.fixture(scope='function')
    def g(self):
        
        g = mp.full('g', [f'a{i}' for i in range(5)], position=np.arange(15).reshape((5, 3)), addBondByIndex=[[0, 1], [1, 2], [2, 3], [3, 4]])
        
        
        yield g
    
    def test_group_copy(self, g):
        
        gcopy = deepcopy(g)
        
        assert g.name == gcopy.name
        assert g.natoms == gcopy.natoms
        assert [atom.uuid for atom in g.atoms] != [atom.uuid for atom in gcopy.atoms]
        g.atoms[0].position == (0, 0, 0)
        assert np.array_equal(gcopy.atoms[0].position, np.array([0, 1, 2]))
        
    def test_bond_copy(self, g):
        
        gcopy = deepcopy(g)

        assert g.nbonds == gcopy.nbonds
        assert gcopy.bonds[0].atom == gcopy.atoms[0]
        assert gcopy.bonds[0].btom == gcopy.atoms[1]
        
        assert g._bonds.keys() != gcopy._bonds.keys()
        
    def test_uniformity(self, g):
        
        a = g.atoms[0]
        b = g.atoms[1]
        
        a.key = 1
        b.key = 1
        assert g.bonds[0].atom.key == 1
        assert g.bonds[0].btom.key == 1
        
        gg = g()
        assert gg.atoms[0].key == 1
        assert gg.atoms[0].key == 1
        assert gg.bonds[0].atom.key == 1
        assert gg.bonds[0].btom.key == 1
        assert gg.bonds[0].atom != a or gg.bonds[0].btom != b    