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
        cutoff = self.cutoff
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        space = max_xyz - min_xyz
        grids = np.ceil(space / cutoff).astype(np.int32)
        num_grids = np.prod(grids)
        buffer = (grids * cutoff - space) / 2
        start_xyz = min_xyz - buffer
        # (N, D)
        grid_id = ((xyz - start_xyz) // cutoff).astype(np.int32)
        grid_coe = np.array([1, grids[0], grids[1]], np.int32)
        # (N, )
        grid_id_1d = np.sum(grid_id * grid_coe, axis=-1).astype(np.int32)
        # (N, 2)
        grid_id_dict = np.ndenumerate(grid_id_1d)
        # (G, *)
        grid_dict = dict.fromkeys(range(num_grids), ())
        for index, value in grid_id_dict:
            grid_dict[value] += index
        neighbour_grid = (NEIGHBOUR_GRID * grid_coe).sum(axis=-1).astype(np.int32)
        neighbour_pairs = []
    
        for i in range(num_grids):
            if grid_dict[i]:
                keeps = np.where((neighbour_grid + i < num_grids) & (neighbour_grid + i >= 0))[0]
                neighbour_grid_keep = neighbour_grid[keeps] + i
                grid_atoms = np.array(list(grid_dict[i]), np.int32)
                try:
                    grid_neighbours = np.array(list(chain(*itemgetter(*neighbour_grid_keep)(grid_dict))), np.int32)
                except TypeError:
                    if neighbour_grid_keep.size == 0:
                        grid_neighbours = np.array([], np.int32)
                    else:
                        grid_neighbours = np.array(list(itemgetter(*neighbour_grid_keep)(grid_dict)), np.int32)
                grid_xyzs = xyz[grid_atoms]
                grid_neighbour_xyzs = xyz[grid_neighbours]
                
                # 单格点内部原子间距
                grid_dis = np.linalg.norm(box.wrap((grid_xyzs[:, None] - grid_xyzs[None])[0]), axis=-1)
                grid_pairs = np.argsort(grid_dis, axis=-1)
                grid_cut = np.take_along_axis(grid_dis, grid_pairs, axis=-1)
                pairs = np.where(grid_cut <= cutoff)
                pairs_id0 = grid_atoms[pairs[0]]
                pairs_id1 = grid_atoms[grid_pairs[pairs]]
                neighbour_pairs.extend(list(np.hstack((pairs_id0[..., None], pairs_id1[..., None]))))
                # 中心格点-周边格点内原子间距
                grid_dis = np.linalg.norm(grid_xyzs[:, None] - grid_neighbour_xyzs[None], axis=-1)
                grid_pairs = np.argsort(grid_dis, axis=-1)
                grid_cut = np.take_along_axis(grid_dis, grid_pairs, axis=-1)
                pairs = np.where(grid_cut <= cutoff)
                pairs_id0 = grid_atoms[pairs[0]]
                pairs_id1 = grid_neighbours[grid_pairs[pairs]]
                neighbour_pairs.extend(list(np.hstack((pairs_id0[..., None], pairs_id1[..., None]))))
        neighbour_pairs = np.sort(np.array(neighbour_pairs), axis=-1)
        sort_args = np.argsort(neighbour_pairs[:, 0])
        pairs = neighbour_pairs[sort_args]
        unique_pairs = np.unique(pairs, axis=0)
        return np.atleast_2d(unique_pairs[unique_pairs[:, 0] != unique_pairs[:, 1]])   