import numpy as np
import molpy as mp
from molpy.core.neighborlist import NeighborList
def test_nblist():
    N = 5
    D = 3
    cutoff = 1.0
    xyz = np.random.random((N, D)).astype(np.float32) * np.array([3., 4., 5.], np.float32)
    box = mp.Box(np.array([3., 4., 5.]))
    nblist = NeighborList(cutoff)
    pairs = nblist.build(xyz, box)
    dist = np.linalg.norm((xyz[:, None] - xyz[None])[0], axis=-1)