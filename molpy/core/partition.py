# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-10
# version: 0.0.1

from molpy.core.nblist import NeighborList
from .box import Box
import numpy as np
from itertools import product

def calc_cell_index_by_coord(cell_coords, cells_per_side):
    """
    Calculate the cell index by the cell coordinates.

    Parameters
    ----------
    cell_coords : (N, 3)
        cell coordinates
    cells_per_side : (3, )
        # of cell in each direction

    Returns
    -------
    (N, )
        cell index
    """
    return cell_coords[:, 0] + cell_coords[:, 1] * cells_per_side[0] + cell_coords[:, 2] * cells_per_side[0] * cells_per_side[1]

class CellList:

    def __init__(self, xyz, xyz_cell_coord, cells_per_side, cell_lattice):
        self.cells_per_side = cells_per_side
        self.xyz = xyz
        self.xyz_cell_coord = xyz_cell_coord
        self.cell_lattice = cell_lattice
        self.xyz_cell_index = calc_cell_index_by_coord(xyz_cell_coord, cells_per_side)

    def get_xyz_by_cell_coords(self, coords):
        """
        Get the xyz of the atoms in the given cell coordinates.

        Parameters
        ----------
        coords : (m, n, 3)
            The cell coordinates of the cell, where m is # of coords set, n is # of coords in each set, 3 is dimension. 

        Returns
        -------
        (m*n, 3)
            xyz of the atoms in the given cell coordinates.
        """
        xyz_cell_index = self.xyz_cell_index
        cell_index = np.zeros(coords.shape[:-1], dtype=np.int32)
        # convert coords to cell index
        for i, coord in enumerate(coords):
            cell_index[i] = calc_cell_index_by_coord(coord, self.cells_per_side)
        
        # Compute the indices of the atoms in the given cell.
        indices = np.isin(xyz_cell_index, cell_index.reshape((-1, 3)))

        return self.xyz[indices]


def calc_cell_dimensions(box_matrix:np.ndarray, minimum_cell_size):
    """Compute the number of cells-per-side and total number of cells in a box."""

    box_length = box_matrix.diagonal()
    cells_per_side = np.floor(box_length / minimum_cell_size)
    cell_lattice = box_matrix / cells_per_side

    return cell_lattice, cells_per_side

def calc_cell_coord(xyz, cell_lattice, cells_per_side):
    """Compute the cell coordinate for each atom."""

    # Compute the cell coordinate for each atom.
    inv_cell_lattice = np.linalg.inv(cell_lattice)
    atom_indices = np.floor(np.dot(xyz, inv_cell_lattice.T))

    # Compute the cell index for each atom.
    # atom_index = atom_indices[:, 0] + atom_indices[:, 1] * cells_per_side[0] + atom_indices[:, 2] * cells_per_side[0] * cells_per_side[1]

    # Sort the atoms by cell index.
    # sorted_indices = np.argsort(cell_index)
    # sorted_xyz = xyz[sorted_indices]
    # sorted_cell_index = cell_index[sorted_indices]

    # Compute the number of atoms in each cell.
    # cell_counts = np.bincount(atom_index)

    # Compute the index of the first atom in each cell.
    # cell_offsets = np.zeros(len(cell_counts) + 1, dtype=np.int32)
    # cell_offsets[1:] = np.cumsum(cell_counts)

    return atom_indices

def create_cellList(box, xyz, rCutoff):

    # 1. standardize periodic boundary condition
    # Assume box is a standard parallelepiped.
    if isinstance(box, np.ndarray):
        box = Box.from_matrix(box)

    # 2. Compute the size of cells per side.
    cell_lattice, cells_per_side = calc_cell_dimensions(box._matrix, rCutoff)

    # 3. Compute the cell coordinate for each atom.
    xyz_cell_coord = calc_cell_coord(xyz, cell_lattice, cells_per_side)

    return CellList(xyz, xyz_cell_coord, cells_per_side, cell_lattice)

class NeighborList:

    def __init__(self, indices, dr, r):
        self.indices = indices
        self.dr = dr
        self.r = r

    def get_pairs(self):
        """
        get the indices of atoms that are within the cutoff distance
        """
        

    def get_neighbors(self):
        """
        get the indices of an atom that are within its cutoff distance
        """
        pass

def get_shell_cells(centerCellIndices, cellPerSide, rShell=1, isSymmetricCell:bool=True):

    dx = np.array([i for i in product(range(-rShell, rShell+1), repeat=3)])
    if isSymmetricCell:
        dx = dx[:int(len(dx) / 2)+1]
        shell_cell_coord = centerCellIndices + dx
        shell_cell_coord = np.where(shell_cell_coord < 0, shell_cell_coord + cellPerSide, shell_cell_coord)  # periodic boundary condition
    else:
        shell_cell_coord = centerCellIndices + dx
        shell_cell_coord = np.where(shell_cell_coord >= cellPerSide, shell_cell_coord - cellPerSide, shell_cell_coord)
        shell_cell_coord = np.where(shell_cell_coord < 0, shell_cell_coord + cellPerSide, shell_cell_coord)

    return shell_cell_coord

def get_check_cells(cell_per_side, rShell=1, isSymmetricPair:bool=True):
    """Get the cells that are within one shell of the cell containing the given atom."""
    centerCellCoord = np.array([i for i in product(range(cell_per_side[0]), range(cell_per_side[1]), range(cell_per_side[2]))])

    shellCells = np.zeros((len(centerCellCoord), (2*rShell+1)**3, 3), dtype=int)
    for i, center in enumerate(centerCellCoord):
        shellCells[i] = get_shell_cells(center, cell_per_side, rShell, isSymmetricPair)

    return centerCellCoord, shellCells


def create_neighborList(box, xyz, rCutoff, rSkin=0, isCellList:bool=True, isSymmetricPair:bool=True):

    # 1. standardize periodic boundary condition
    # Assume box is a standard parallelepiped.
    if isinstance(box, np.ndarray):
        box = Box.from_matrix(box)

    # 2. pre-compute arguments
    cutoff = rCutoff + rSkin

    # 3. get the candidate pairs to compute distance
    if isCellList:
        indices = []
        cellList = create_cellList(box, xyz, rCutoff)

        centerCellIndices, shellCells = get_check_cells(cellList.cells_per_side, rShell=1, isSymmetricPair=isSymmetricPair)

        centerCellXyz = cellList.get_xyz_by_cell_coords(centerCellIndices)
        shellCellXyz = cellList.get_xyz_by_cell_coords(shellCells)

        # calculate distance pair-wisely
        centerCellXyzMat = np.tile(centerCellXyz[:, None, :], (1, shellCellXyz.shape[1], 1))
        shellCellXyzMat = np.tile(shellCellXyz, (centerCellXyz.shape[0], 1, 1))
        dr = box.displacement(centerCellXyzMat, shellCellXyzMat)

    else:
        # Compute the distance between each pair of atoms.
        dr = box.displacement(xyz[:, None, :], xyz[None, :, :])
    r = np.linalg.norm(dr, axis=-1)
    r_triu = np.triu(r)
    indices = np.argwhere(np.logical_and(r_triu < cutoff, r_triu!=0))

    return NeighborList(indices, dr, r)
