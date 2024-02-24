import itertools
import numpy as np
from itertools import chain
from operator import itemgetter
import molpy as mp

NEIGHBOUR_GRID = np.array(
    [
        [-1, 1, 0],
        [-1, -1, 1],
        [-1, 0, 1],
        [-1, 1, 1],
        [0, -1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    np.int32,
)


class NeighborList:

    def __init__(self, cutoff):
        self.cutoff = cutoff
        self._xyz = np.array([])

    def build(self, frame):
        xyz = frame.atoms.positions
        self._xyz = xyz
        self._cell_shape, self._cell_offset, self._all_cell_coords = self._init_cell(
            xyz, self.cutoff
        )
        self._xyz_cell_coords, self._xyz_cell_idx = self._add_to_cell(
            xyz, self._cell_offset
        )

    def update(self, frame):
        xyz = frame.atoms.positions
        # TODO: check if xyz and box change a lot
        if len(xyz) != len(self._xyz):
            self.build(frame)
        pairs = self.find_all_pairs(frame.box)
        frame[mp.Alias.idx_i] = pairs[:, 0]
        frame[mp.Alias.idx_j] = pairs[:, 1]
        frame[mp.Alias.Rij] = frame.box.diff(xyz[pairs[:, 0]], xyz[pairs[:, 1]])
        return frame

    def __call__(self, frame):
        return self.update(frame)

    def find_all_pairs(self, box):

        results = []
        for cell_coord in self._xyz_cell_coords:
            pairs = self._find_pairs_around_center_cell(cell_coord, box)
            results.append(pairs)
            pairs = self._find_pairs_in_center_cell(cell_coord, box)
            results.append(pairs)
        results = filter(lambda x: len(x) > 0, results)
        results = list(results)
        if len(results) == 0:
            return np.array([])
        results = np.concatenate(list(results))
        # NOTE: neighbor cell has been halved to avoid double counting
        results = np.sort(results, axis=-1)
        results = np.unique(results, axis=0)
        return results

    def _find_pairs_in_center_cell(self, center_cell_coord, box):

        center_xyz, center_id = self._find_atoms_in_cell(
            self._xyz,
            self._xyz_cell_idx,
            self._cell_coord_to_idx(center_cell_coord),
        )
        distance = np.linalg.norm(box.all_diff(center_xyz, center_xyz), axis=-1)
        pair_id = np.array(list(itertools.product(center_id, center_id)))
        cutoff_mask = np.logical_and(distance < self.cutoff, distance > 0)
        pairs = pair_id[cutoff_mask]
        return pairs[
            pairs[:, 0] < pairs[:, 1]
        ]  # halve the pairs to avoid double counting

    def _find_pairs_around_center_cell(self, center_cell_coord, box):
        nbor_cell = self._find_neighbor_cell(self._cell_shape, center_cell_coord)
        center_xyz, center_id = self._find_atoms_in_cell(
            self._xyz,
            self._xyz_cell_idx,
            self._cell_coord_to_idx(center_cell_coord),
        )
        around_xyz, around_id = self._find_atoms_in_cell(
            self._xyz, self._xyz_cell_idx, self._cell_coord_to_idx(nbor_cell)
        )
        distance = np.linalg.norm(
            box.all_diff(center_xyz, around_xyz), axis=-1
        )  # (N*M, )
        pair_id = np.array(list(itertools.product(center_id, around_id)))
        cutoff_mask = np.logical_and(distance < self.cutoff, distance > 0)
        pairs = pair_id[cutoff_mask]
        return pairs

    def _init_cell(
        self,
        xyz,
        cutoff,
    ):
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        space = max_xyz - min_xyz
        space = np.where(space == 0, 1, space)
        _cell_shape = np.ceil(space / cutoff).astype(int)
        _cell_offset = np.array(
            [_cell_shape[0] * _cell_shape[1], _cell_shape[0], 1], dtype=int
        )
        _all_cell_coords = np.array(list(np.ndindex(*_cell_shape)))
        return _cell_shape, _cell_offset, _all_cell_coords

    def _add_to_cell(self, xyz, cell_offset):
        _xyz_cell_coords = (xyz - np.min(xyz, axis=0)) // self.cutoff  # (N, D)
        _xyz_cell_idx = (_xyz_cell_coords * cell_offset).sum(axis=-1)  # (N,)
        return _xyz_cell_coords.astype(int), _xyz_cell_idx.astype(int)

    def _find_atoms_in_cell(self, xyz, xyz_cell_idx, which_cell_idx):
        mask = np.isin(xyz_cell_idx, which_cell_idx)
        return xyz[mask], np.where(mask)[0]

    def _find_neighbor_cell(self, cell_shape, center_cell_coord):
        cell_matrix = np.diag(cell_shape)
        nbor_cell = NEIGHBOUR_GRID + center_cell_coord
        reci_r = np.einsum("ij,nj->ni", np.linalg.inv(cell_matrix), nbor_cell)
        shifted_reci_r = reci_r - np.floor(reci_r)
        nbor_cell = np.einsum("ij,nj->ni", cell_matrix, shifted_reci_r)
        return nbor_cell

    def _cell_coord_to_idx(self, cell_coord):
        cell_idx = (cell_coord * self._cell_offset).sum(axis=-1)
        return cell_idx
