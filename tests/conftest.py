import molpy as mp
atom = mp.Atom('test')
group = mp.Group('test')

import pytest
import numpy as np

@pytest.fixture()
def C6():
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