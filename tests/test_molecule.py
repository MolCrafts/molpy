# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-14
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

from molpy.molecule import Molecule

class TestMolecule:
    
    def test_add_group(self, CH4):
        
        m = mp.Molecule('pe')
        
        nCH4 = 10
        for i in range(nCH4):
            m.addGroup(CH4(name=f'ch4-{i}'))
        assert m.ngroups == nCH4
        assert m.natoms == nCH4 * 5
        assert m.nbonds == nCH4 * 4
        
    def test_uniformity(self, CH4):
        ch4 = mp.Molecule('CH4')
        CH4copy = CH4()
        for bond in CH4copy.bonds:
            assert bond.atom in CH4copy.atoms
        ch4.addGroup(CH4())
        for atom in ch4.atoms:
            atom.key = 1
            
        for atom in ch4.atoms:
            assert atom.key == 1
            
        for bond in ch4.bonds:
            assert bond.atom in ch4.atoms
            assert bond.atom.key == 1
            assert bond.btom.key == 1
        
    def test_addBondByName(self, CH4):
        degreeOfPolymerization = 10
        pe = mp.Molecule('polyethene')
        for unit in range(degreeOfPolymerization):
            ch4 = CH4(name=f'CH4-{unit}')
            pe.addGroup(ch4)
            if unit != 0:
                pe.addBondByName(f'H0@CH4-{unit}', f'H2@CH4-{unit-1}')
                
        pe.searchAngles()
        pe.searchDihedrals()
                
        assert pe.ngroups == degreeOfPolymerization
        assert pe.natoms == 5*degreeOfPolymerization
        assert pe.nbonds == 4*degreeOfPolymerization + degreeOfPolymerization - 1
        print(pe._angleList)
        assert pe.nangles == 6 * degreeOfPolymerization + (degreeOfPolymerization-1)*2
        assert pe.ndihedrals == (3+1+3)*(degreeOfPolymerization-1)
        
class TestMoleculeGeometry:
    
    def test_move(self, H2O):
        m = Molecule('waters')