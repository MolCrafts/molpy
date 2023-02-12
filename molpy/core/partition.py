# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-10
# version: 0.0.1

from molpy.core.nblist import NeighborList
from .box import Box
import numpy as np
from itertools import product


class CellList:

    def __init__(self, index_list, cells_per_side):
        self.cells_per_side = cells_per_side
        self.index_list = index_list

    def get_all_cell_indices(self, only_occupied=True):
        """Get the indices of all cells."""
        if only_occupied:
            return self.index_list
        else:
            return np.array([i for i in product(range(self.cells_per_side[0]), range(self.cells_per_side[1]), range(self.cells_per_side[2]))])

    def get_shell_cell_indices(self, index):
        """Get the cells that are within one shell of the cell containing the given atom."""
        # Compute the cell index for the given atom.
        dx = np.array([i for i in product([0, -1], repeat=3)])

        shell_cell_indices = index + dx

        shell_cell_indices = np.where(shell_cell_indices < 0, shell_cell_indices + self.cells_per_side, shell_cell_indices)

        return shell_cell_indices

    def get_indices(self, cell_indices):

        # Compute the cell index for the given atom.
        cell_index = cell_indices[:, 0] + cell_indices[:, 1] * self.cells_per_side[0] + cell_indices[:, 2] * self.cells_per_side[0] * self.cells_per_side[1]

        _cell_index = self.index_list[:, 0] + self.index_list[:, 1] * self.cells_per_side[0] + self.index_list[:, 2] * self.cells_per_side[0] * self.cells_per_side[1]

        # Compute the indices of the atoms in the given cell.
        indices = np.argwhere(np.isin(_cell_index, cell_index))

        return indices.reshape(-1)


def calc_cell_dimensions(box_matrix:np.ndarray, minimum_cell_size):
    """Compute the number of cells-per-side and total number of cells in a box."""

    # Compute the number of cells per side.
    cell_length = box_matrix.diagonal() / minimum_cell_size
    cells_per_side = np.floor(cell_length).astype(np.int32)
    # xy, xz, yz = box_matrix[0, 1], box_matrix[0, 2], box_matrix[1, 2]
    # cell_lattice = np.array(
    #     [cell_length[0], xy / cells_per_side[0], xz / cells_per_side[0], 0, cell_length[1], yz / cells_per_side[1], 0, 0, cell_length[2]]
    # ).reshape((3, 3))
    cell_lattice = box_matrix / cells_per_side

    return cell_lattice, cells_per_side

def create_cellList(box, xyz, rCutoff):

    if isinstance(box, np.ndarray):
        box = Box.from_matrix(box)

    # Compute the number of cells per side.
    # Assume box is a standard parallelepiped.
    cell_lattice, cells_per_side = calc_cell_dimensions(box._matrix, rCutoff)

    # Compute the cell indices for each atom.
    reci_r = np.dot(xyz, np.linalg.inv(cell_lattice).T)

    # Compute the cell coordinate for each atom.
    cell_indices = np.floor(reci_r).astype(int)

    # Compute the cell index for each atom.
    cell_index = cell_indices[:, 0] + cell_indices[:, 1] * cells_per_side[0] + cell_indices[:, 2] * cells_per_side[0] * cells_per_side[1]

    # Sort the atoms by cell index.
    # sorted_indices = np.argsort(cell_index)
    # sorted_xyz = xyz[sorted_indices]
    # sorted_cell_index = cell_index[sorted_indices]

    # Compute the number of atoms in each cell.
    # cell_counts = np.bincount(sorted_cell_index)

    # Compute the index of the first atom in each cell.
    # cell_offsets = np.zeros(len(cell_counts) + 1, dtype=np.int32)
    # cell_offsets[1:] = np.cumsum(cell_counts)

    return CellList(cell_indices, cells_per_side)

class NeighborList:

    def __init__(self, indices):
        self.indices = indices

    def get_pairs(self):
        """
        get the indices of atoms that are within the cutoff distance
        """
        return 

    def get_neighbors(self):
        """
        get the indices of an atom that are within its cutoff distance
        """
        pass


def create_neighborList(box, xyz, rCutoff, isCellList:bool=True):

    box = Box.from_matrix(box)
    
    if isCellList:
        indices = []
        cellList = create_cellList(box, xyz, rCutoff)
        cell_indices = cellList.get_all_cell_indices()
        for cell_index in cell_indices:
            shell_cell_indices = cellList.get_shell_cell_indices(cell_index)

            _indices = cellList.get_indices(shell_cell_indices)
            part_xyz = xyz[_indices]
            # Compute the distance between each pair of atoms.
            dr = box.displacement(part_xyz[:, None, :], part_xyz[None, :, :])
            r = np.linalg.norm(dr, axis=-1)
            r_triu = np.triu(r)
            mask = np.argwhere(np.logical_and(r_triu < rCutoff, r_triu!=0))
            
            indices.append(_indices[mask])

        indices = np.sort(np.concatenate(indices), axis=0)

    else:
        # Compute the distance between each pair of atoms.
        dr = box.displacement(xyz[:, None, :], xyz[None, :, :])
        r = np.linalg.norm(dr, axis=-1)
        r_triu = np.triu(r)
        indices = np.argwhere(np.logical_and(r_triu < rCutoff, r_triu!=0))

    return NeighborList(indices)
