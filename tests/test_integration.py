# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-16
# version: 0.0.1

import molpy as mp
import pytest
import numpy as np
import profile

from molpy import element
from molpy.cell import Cell
from molpy.system import System

class TestMoltemplateH2Ocase:
    """Those tests is modeled using the moltemplate style: setup basic unit first with atoms and bonds, def forcefield second with atom and bond properties, match all the info at least.
    """
    
        
    @pytest.fixture(scope='class', name='ff')
    def SPCEforcefield(self):
        ff = mp.ForceField('SPCE')
        
        ff.defAtomType('O', mass=15.9994, charge=-0.8476, element='O')
        ff.defAtomType('H', mass=1.008, charge=0.4238, element='H')
        
        ff.defBondType('OH', style='harmonic', k='1000.0', r0='1.0')
        
        ff.defAngleType('HOH', style='harmonic', k='1000.0', theta0='109.47')
        yield ff
        
    @pytest.fixture(scope='class')
    def H2O(self, ff):
        #
        # file "spce_simple.lt"
        #
        #  h1  h2
        #   \ /
        #    o
        o = mp.Atom('o', atomType=ff.atomType['O'], position=np.array([0.0000000, 0.000000, 0.00000]))
        h1 = mp.Atom('h1', atomType=ff.atomType['H'], position=np.array([0.8164904, 0.5773590, 0.00000]))
        h2 = mp.Atom('h2', atomType=ff.atomType['H'], position=np.array([-0.8164904, 0.5773590, 0.00000]))
        
        h2o = mp.Group('h2o')
        h2o.addAtoms([o, h1, h2])
        h2o.addBondByName('o', 'h1', bondType=ff.bondTypes['OH'])
        h2o.addBondByName('o', 'h2', bondType=ff.bondTypes['HOH'])
        yield h2o

    @pytest.fixture(scope='system')
    def system(self, ff, H2O):
        system = System('SPCE H2O')
        system.setUnitType('SI')
        system.setAtomStyle('full')
        system.setBondStyle('harmonic')
        system.setAngleStyle()
        system.setDihedralStyle()
        system.setPairStyle()   
        # kspace_style pppm 0.0001 # long-range electrostatics sum method
        # pair_modify mix arithmetic
        
        system.cell = Cell(3, 'ppp', xlo=0, xhi=35, ylo=0, yhi=35, zlo=0, zhi=35)
        system.forcefield = ff
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    system.addMolecule(H2O(f'h2o{i+j+k+1}').move(3.10*i, 3.10*j, 3.10*k))
                    
        system.complete()
        yield system
        
    def test_H2O_case(self, system):

        assert system.natoms == 3000
        assert system.nbonds == 2000
        assert system.nangles == 1000
        assert system.natomTypes == 2
        assert system.nbondTypes == 1
        assert system.nangleTypes == 1
        
        
class TestMoltemplatePolymercase:
        
    @pytest.fixture(scope='class')
    def ff(self):
        ff = mp.ForceField('polymer')
        ff.defAtomType('CA', mass=13.0, charge=0.0)
        ff.defAtomType('R', mass=50.0, charge=0.0)
        
        ff.defBondType('SideChain', k=15.0, r0=3.4)
        ff.defBondType('Backbond', k=15.0, r0=3.7)
        
        ff.defAngleType('Backbone',  k=30.0, theta0=114, itom='CA', jtom='CA', ktom='CA')
        ff.defAngleType('Sidechain', k=30.0, theta0=132, itom='CA', jtom='CA', ktom='R')
        
        ff.defDihedralType('CCCC', K=-0.5, n=1, d=-180, w=0.0, itom='CA', jtom='CA', ktom='CA', ltom='CA')
        ff.defDihedralType('RCCR', K=-1.5, n=1, d=-180, w=0.0, itom='R', jtom='CA', ktom='CA', ltom='CA')
        

    @pytest.fixture(scope='class')
    def Monomer(self, ff):
        
        g = mp.Group('Monomer')
        
        ca = mp.Atom('ca')
        ca.atomType=ff.atomTypes['CA']
        ca.charge=0.0
        ca.position = np.array([0.0000, 1.0000, 0.0000])
        
        r = mp.Atom('r')
        r.atomType=ff.atomTypes['R']
        r.charge=0.0
        r.position = np.array([0.0000, 4.4000, 0.0000])
        
        g.addAtoms([ca, r])
        g.addBond(ca, r, bondType=ff.bondTypes['OH'])
        
        yield g
        
    @pytest.fixture(scope='class')
    def Polymer(self, Monomer, ff):
        
        p = mp.Molecule('Polymer')
        
        for i in range(7):
            p.addGroup(Monomer(name=f'mon{i+1}').rot(180*i, 1, 0, 0).move(3.2*i, 0, 0))
            if i!=0:
                p.addBondByName(f'ca@mon{i-1}', f'ca@mon{i}')
                
        # add topology among groups
                
        yield p
        
    @pytest.fixture(scope='class')
    def system(self, ff, Polymer):
        system = mp.System('Polymer')
        for i in range(1):
            system.addMolecule(Polymer(name=f'poly{i+1}'))
    
    
    def test_PP_case(self, Monomer):
        pp = mp.Molecule('PP')
        degreeOfPolymerization = 10
        for i in range(degreeOfPolymerization):
            pp.addGroup(Monomer(name=f'mol{i}'))
            if i != 0:
                pp.addBondByName(f'ca@mol{i-1}', f'ca@mol{i}', bondType='Backbone')
            
        assert pp.natoms == 2 * degreeOfPolymerization
        assert pp.nbonds == degreeOfPolymerization + (degreeOfPolymerization - 1)