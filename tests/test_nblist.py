import numpy as np
import molpy as mp
from molpy.core.neighborlist import NeighborList
def test_nblist():
    N = 5
    D = 3
    cutoff = 0.5
    # xyz = np.random.random((N, D)).astype(np.float32) * np.array([3., 4., 5.], np.float32)
    xyz = np.array([
        [0, 0, 0],
        [0.4, 0, 0],
        [0.8, 0, 0]
    ])
    box = mp.Box(np.array([1., 1., 1.]))
    print(box.diff(np.array([0, 0, 0]), np.array([0.8, 0, 0])))
    print(box.diff(np.array([0.8, 0, 0]), np.array([0, 0, 0])))
    # nblist = NeighborList(cutoff)
    # pairs = nblist.build(xyz, box)
    # dist = np.linalg.norm((xyz[:, None] - xyz[None])[0], axis=-1)
    # print(pairs)
    # print(dist)
    assert False