import numpy as np
from itertools import chain
from operator import itemgetter

NEIGHBOUR_GRID = np.array([
       [-1,  1,  0],
       [-1, -1,  1],
       [-1,  0,  1],
       [-1,  1,  1],
       [ 0, -1,  1],
       [ 0,  0,  1],
       [ 0,  1,  0],
       [ 0,  1,  1],
       [ 1, -1,  1],
       [ 1,  0,  0],
       [ 1,  0,  1],
       [ 1,  1,  0],
       [ 1,  1,  1]], np.int32)

class NeighborList:

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def build(self, xyz, box):
        return self.update(xyz, box)
    
    def update(self, xyz, box):
        large_dis = np.tril(np.ones((xyz.shape[0], xyz.shape[0])) * 999)
        # (N, N)
        dis = np.linalg.norm(xyz[None] - xyz[:, None], axis=-1) + large_dis
        # (N, M)
        neigh = np.argsort(dis, axis=-1)
        # (N, M)
        cut = np.take_along_axis(dis, neigh, axis=1)
        # (2, P)
        pairs = np.where(cut <= self.cutoff)
        # (P, )
        pairs_id0 = pairs[0]
        pairs_id1 = neigh[pairs]
        # (P, 2)
        sort_args = np.argsort(pairs_id0)
        return np.hstack((pairs_id0[..., None], pairs_id1[..., None]))[sort_args]