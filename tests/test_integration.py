# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-16
# version: 0.0.1

import molpy as mp
import pytest
import numpy as np
import profile

class TestMoltemplateCase:
    
    @pytest.fixture(scope='class')
    def H2O(self):
        #
        # file "spce_simple.lt"
        #
        #  h1  h2
        #   \ /
        #    o
        o = mp.Atom('o', position=np.array([0.0000000, 0.000000, 0.00000]), element='O', charge= -0.8476)
        h1 = mp.Atom('h1', position=np.array([0.8164904, 0.5773590, 0.00000]), element='H', charge= 0.4238)
        h2 = mp.Atom('h2', position=np.array([-0.8164904, 0.5773590, 0.00000]), element='H', charge=0.4238)
        
        h2o = mp.Group('h2o')
        h2o.addAtoms([o, h1, h2])
        # h2o.addBondByName('o', 'h1')
        # h2o.addBondByName('o', 'h2')
        yield h2o
        
    @pytest.fixture(scope='class')
    def SPCEforcefield(self):
        ff = mp.ForceField('SPCE')
        h2oT = mp.Template('h2oT')
        ff.defTemplate(h2oT)
        o = mp.Atom('o')
        h1 = mp.Atom('h1')
        h2 = mp.Atom('h2')
        h2oT.addAtoms([o, h1, h2])
        h2oT.addBondByName('o', 'h1', typeName='OH')
        h2oT.addBondByName('o', 'h2', typeName='OH')
        h2oT.addAngleByName('h1', 'o', 'h2', typeName='HOH')
        
        ff.defBondType('OH', style='harmonic', k='1000.0', r0='1.0')
        ff.defAngleType('HOH', style='harmonic', k='1000.0', theta0='109.47')
        yield ff
        
    def test_H2O_case(self, H2O, SPCEforcefield):
        ff = SPCEforcefield
        assert H2O.natoms == 3
        assert H2O.nbonds == 0
        template = ff.matchTemplate(H2O, criterion='medium')
        assert template.name == 'h2oT'
        ff.patch(template, H2O)
        assert H2O.nbonds == 2
        assert H2O.nangles == 1
        
    @pytest.fixture(scope='class')
    def Monomer(self):
        g = mp.Group('M')
        ca = mp.Atom('ca', atomType='CA', charge=0.0)
        ca.position = np.array([0.0000, 1.0000, 0.0000])
        r = mp.Atom('r', atomType='R', charge=0.0)
        r.position = np.array([0.0000, 4.4000, 0.0000])
        g.addAtoms([ca, r])
        g.addBond(ca, r, name='Sidechain')
        
        yield g
    
    
    def test_PP_case(self, Monomer):
        pp = mp.Molecule('PP')
        degreeOfPolymerization = 10
        for i in range(degreeOfPolymerization):
            pp.addGroup(Monomer(name=f'mol{i}'))
            if i != 0:
                pp.addBondByName(f'ca@mol{i-1}', f'ca@mol{i}', bondType='Backbone')
            
        assert pp.natoms == 2 * degreeOfPolymerization
        assert pp.nbonds == degreeOfPolymerization + (degreeOfPolymerization - 1)