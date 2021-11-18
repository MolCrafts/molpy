# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-14
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

class TestMolecule:
    
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
        
    def testMolecule(self, CH4):
        CH4.searchAngles()
        degreeOfPolymerization = 10
        pe = mp.Molecule('polyethene')
        for unit in range(degreeOfPolymerization):
            ch4 = CH4.copy(name=f'CH4-{unit}')
            pe.addGroup(ch4)
            if unit != 0:
                pe.addBondByName(f'H0@CH4-{unit}', f'H2@CH4-{unit-1}')
        pe.searchAngles()
        pe.searchDihedrals()
                
        assert pe.natoms == 5*degreeOfPolymerization
        assert pe.nbonds == 4*degreeOfPolymerization + degreeOfPolymerization - 1
        assert pe.nangles == 6 * degreeOfPolymerization + (degreeOfPolymerization-1)*2
        assert pe.ndihedrals == 63