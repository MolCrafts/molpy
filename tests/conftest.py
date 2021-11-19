import molpy as mp

atom = mp.Atom("test")
group = mp.Group("test")

import pytest
import numpy as np


@pytest.fixture()
def C6():
    C6 = mp.Group("C6")

    for i in range(6):
        C = mp.Atom(f"C{i}")
        C6.addAtom(C)
    covalentMap = np.array(
        [
            [0, 1, 2, 3, 2, 1],
            [1, 0, 1, 2, 3, 2],
            [2, 1, 0, 1, 2, 3],
            [3, 2, 1, 0, 1, 2],
            [2, 3, 2, 1, 0, 1],
            [1, 2, 3, 2, 1, 0],
        ],
        dtype=int,
    )
    C6.setTopoByCovalentMap(covalentMap)
    for i in range(6):
        H = mp.Atom(f'H{i}')
        C6.addAtom(H)
        C6.addBondByIndex(i, -1)

    assert C6.nbonds == 12
    # assert len(C6._bondList) == len(C6._bonds)
    # C6.reference_covalentMap = covalentMap
    C6.setPositions(
        np.array(
            [
                [-0.921, 2.855, 0.075],
                [0.474, 2.855, 0.075],
                [1.172, 4.063, 0.075],
                [0.474, 5.271, 0.074],
                [-0.921, 5.271, 0.073],
                [-1.619, 4.063, 0.074],
                [-1.471, 1.903, 0.075],
                [1.024, 1.903, 0.076],
                [2.271, 4.063, 0.075],
                [1.024, 6.223, 0.074],
                [-1.471, 6.223, 0.072],
                [-2.718, 4.063, 0.074],
            ]
        )
    )
    CType = mp.AtomType('C')
    HType = mp.AtomType('H')
    assert CType is not None
    C6.setAtomTypes(
        [CType, CType, CType, CType, CType, CType,
         HType, HType, HType, HType, HType, HType]
    )
    yield C6

@pytest.fixture()
def particle():
    p = mp.Atom('particle', key='value', position=np.array([0., 0., 0.]))
    yield p

@pytest.fixture()
def CH4():
    CH4 = mp.Group('CH4')
    CH4.addAtom(mp.Atom('C', element='C'))
    CH4.addAtoms([mp.Atom(f'H{i}', element='H') for i in range(4)])
    CH4.addBondByName('C', 'H0')  #       H3         y
    CH4.addBondByName('C', 'H1')  #       |          |
    CH4.addBondByName('C', 'H2')  #  H0 - C - H2      -> x
    CH4.addBondByName('C', 'H3')  #       |
                                  #       H1
                                    
    
    CH4.setPositions(np.array([[0, 0, 0],
                                [-1, 0, 0],
                                [0, -1, 0],
                                [1, 0, 0],
                                [0, 1, 0]]))
    CType = mp.AtomType('C')
    HType = mp.AtomType('H')
    CH4.setAtomTypes([CType, HType, HType, HType, HType])
    
    yield CH4
    
@pytest.fixture()
def SPCEforcefield():
    ff = mp.ForceField('SPCE')
    
    ff.defAtomType('O', mass=15.9994, charge=-0.8476, element='O')
    ff.defAtomType('H', mass=1.008, charge=0.4238, element='H')
    
    ff.defBondType('OH', style='harmonic', k='1000.0', r0='1.0')
    
    ff.defAngleType('HOH', style='harmonic', k='1000.0', theta0='109.47', itomName='h1', jtomName='o', ktomName='h2')
    yield ff
    
@pytest.fixture()
def H2O(SPCEforcefield):
    ff = SPCEforcefield
    #
    # file "spce_simple.lt"
    #
    #  h1  h2
    #   \ /
    #    o
    o = mp.Atom('o', atomType=ff.atomTypes['O'], position=np.array([0.0000000, 0.000000, 0.00000]), key='value')
    h1 = mp.Atom('h1', atomType=ff.atomTypes['H'], position=np.array([0.8164904, 0.5773590, 0.00000]))
    h2 = mp.Atom('h2', atomType=ff.atomTypes['H'], position=np.array([-0.8164904, 0.5773590, 0.00000]))
    h2o = mp.Group('h2o')
    h2o.addAtoms([o, h1, h2])
    h2o.addBondByName('o', 'h1', bondType=ff.bondTypes['OH'])
    h2o.addBondByName('o', 'h2', bondType=ff.bondTypes['OH'])
    yield h2o
