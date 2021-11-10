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
