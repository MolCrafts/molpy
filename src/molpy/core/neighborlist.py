import itertools
import numpy as np
from itertools import chain
from operator import itemgetter

NEIGHBOUR_GRID = np.array([
        [0, 0, 0],
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
        cutoff = self.cutoff
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        space = max_xyz - min_xyz
        grids = np.ceil(space / self.cutoff).astype(np.int32)
        # num_grids = np.prod(grids)
        buffer = (grids * cutoff - space) / 2
        start_crd = min_xyz - buffer
        
        cell_coord = (xyz // cutoff).astype(int)
        cell_offset = np.array([1, grids[0], grids[0]*grids[1]], dtype=int)
        cell_idx = (cell_coord * cell_offset).sum(axis=-1)
        
        all_cell_coord = np.array(list(np.ndindex(*grids)))
        all_cell_idx = all_cell_coord * cell_offset
        grid_matrix = np.diag(grids)
        idx = np.arange(len(xyz))
        pairs = []
        for i, center_cell_coord in enumerate(all_cell_coord):
            center_xyz_mask = np.all(np.isin(cell_coord, center_cell_coord), axis=-1)
            center_xyz = xyz[center_xyz_mask]
            if len(center_xyz) == 0:
                continue
            around_cell = NEIGHBOUR_GRID + center_cell_coord
            reci_r = np.einsum('ij,nj->ni', np.linalg.inv(grid_matrix), around_cell)
            shifted_reci_r = reci_r - np.floor(reci_r)
            around_cell = np.einsum('ij,nj->ni', grid_matrix, shifted_reci_r)
            around_cell_idx = (around_cell * cell_offset).sum(axis=-1)
            around_cell_xyz_mask = np.isin(cell_idx, around_cell_idx, )
            around_xyz = xyz[around_cell_xyz_mask]
            rij = center_xyz[:, None, :] - around_xyz
            rij = box.wrap(rij)
            dij = np.linalg.norm(rij, axis=-1)
            cutoff_mask = np.nonzero(dij < cutoff)
            pairs.extend(list(zip(idx[center_xyz_mask][cutoff_mask[0]], idx[around_cell_xyz_mask][cutoff_mask[1]])))
        pairs = filter(lambda x: x[0] < x[1], pairs)
        pairs = set(pairs)
        return list(pairs)
    
    def update(self, xyz, box):
        pass

    def _calc_cell_idx(self, xyz, cutoff):
        cutoff = self.cutoff
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        space = max_xyz - min_xyz
        grids = np.ceil(space / self.cutoff).astype(np.int32)
        # num_grids = np.prod(grids)
        buffer = (grids * cutoff - space) / 2
        start_crd = min_xyz - buffer
        # (N, D)
        grid_coord = (xyz // cutoff).astype(int)
        grid_offset = np.array([1, grids[0], grids[0]*grids[1]], dtype=int)
        grid_id = (grid_coord * grid_offset).sum(axis=-1)
        return grid_id