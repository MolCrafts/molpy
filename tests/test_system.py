# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp

class TestSystem:
    
    @pytest.fixture(scope='class')
    def system(self):
        
        system = mp.System('test')
        
        yield system
        
    def test_init_system(self, system, C6):
        assert C6.natoms == 12
        assert C6.nbonds == 12
        assert len(C6._bondList) == 12
        assert len(C6._bonds) == 12
        C6.searchAngles()
        C6.searchDihedrals()
        system.addMolecule(C6)
        assert system.natoms == 12
        assert system.nbonds == 12
        assert system.nangles == 18
        assert system.ndihedrals == 24
        
    def test_promote(self):
        
        system = mp.System('test')
        m = system.promote(mp.Atom('a'))
        assert m.itemType == 'Molecule'
        assert m.natoms == 1
        
        g = mp.Group('g')
        g.addAtom(mp.Atom('a'))
        m = system.promote(g)
        assert m.itemType == 'Molecule'
        assert m.natoms == 1       
        
    def test_addSolvent(self):
        
        system = mp.System('test')
        system.box = mp.Box(3, 'ppp', xlo=0, xhi=10,  ylo=0, yhi=10,  zlo=0, zhi=10)
        for i in range(10):
            system.addMolecule(mp.Atom(i, charge=1))
            
        assert system.charge == 10
        system.addSolvent(mp.Atom('k', charge=-1, position=(1,2,3)), ionicStrength=0)
        assert system.natoms == 20