# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-10
# version: 0.0.1

from .box import Box
import numpy as np

# class CellList:

#     def __init__(self, box, minimum_cell_size):

#         self.box = Box.from_matrix(box)
#         self.minimum_cell_size = minimum_cell_size

#     def calc_cell_dimensions(self, box_size, minimum_cell_size):
#         """Compute the number of cells-per-side and total number of cells in a box."""
            
#         # Compute the number of cells per side.
#         cells_per_side = np.ceil(box_size / minimum_cell_size).astype(np.int32)

#         # Compute the total number of cells.
#         cell_count = np.prod(cells_per_side)

#         return cells_per_side, cell_count

#     def create_buffer(self, xyz, box_size, minimum_cell_size):
#         """Create a buffer around the box to avoid periodic boundary conditions."""
#         pass



#     def update(self, xyz, minimum_cell_size, buffer_size_multiplier:float = 1.25):
#         """Update the cell list with a new set of coordinates."""

#         # Create a buffer around the box to avoid periodic boundary conditions.
#         buffer_xyz, buffer_box = self.create_buffer(xyz, self.box.lengths, minimum_cell_size)

#         # Compute the number of cells per side and the total number of cells.
#         cells_per_side, cell_count = self.calc_cell_dimensions(buffer_box, minimum_cell_size)

#         # Compute the cell indices for each atom.
#         cell_indices = np.floor(buffer_xyz / minimum_cell_size).astype(np.int32)

#         # Compute the cell index for each atom.
#         cell_index = cell_indices[:, 0] + cell_indices[:, 1] * cells_per_side[0] + cell_indices[:, 2] * cells_per_side[0] * cells_per_side[1]

#         # Sort the atoms by cell index.
#         sorted_indices = np.argsort(cell_index)
#         sorted_xyz = buffer_xyz[sorted_indices]
#         sorted_cell_index = cell_index[sorted_indices]

#         # Compute the number of atoms in each cell.
#         cell_counts = np.bincount(sorted_cell_index)

#         # Compute the index of the first atom in each cell.
#         cell_offsets = np.zeros(cell_count + 1, dtype=np.int32)
#         cell_offsets[1:] = np.cumsum(cell_counts)

#         # Compute the cell index for each atom.
#         cell_index = cell_indices[:, 0] + cell_indices[:, 1] * cells_per_side[0] + cell_indices[:, 2] * cells_per_side[0] * cells_per_side[1]

#         # Sort the atoms by cell index.
#         sorted_indices = np.argsort(cell_index)
#         sorted_xyz = buffer_xyz[sorted_indices]
#         sorted_cell_index = cell_index[sorted_indices]

#         # Compute the number of atoms in each cell.
#         cell_counts = np.bincount(sorted_cell_index)

#         # Compute the index of the first atom in each cell.
#         cell_offsets = np.zeros(cell_count + 1, dtype=np.int32)
#         cell_offsets[1:] = np.cumsum(cell_counts)

#         # Store the cell list.
#         self.box = Box.from_matrix(buffer_box)
#         self.cell_size = cell_size
#         self.cell_count = cell_count
#         self.cell_offsets = cell_offsets
#         self.cell_counts = cell_counts
#         self.cell_indices = cell_indices
#         self.cell_index = cell_index
#         self.sorted_xyz

class CellList:

    def __init__(self, xyz, cell_offsets):
        self.xyz = xyz
        self.cell_offsets = cell_offsets


def calc_cell_dimensions(box_matrix:np.ndarray, minimum_cell_size):
    """Compute the number of cells-per-side and total number of cells in a box."""

    # Compute the number of cells per side.
    cell_lattice = box_matrix / minimum_cell_size
    cells_per_side = np.floor(cell_lattice.diagonal()).astype(np.int32)

    return cell_lattice, cells_per_side

def create_cellList(box, xyz, rCutoff):

    box = Box.from_matrix(box)

    # Compute the number of cells per side.
    # Assume box is a standard parallelepiped.
    cell_lattice, cells_per_side = calc_cell_dimensions(box._matrix, rCutoff)

    # Compute the cell indices for each atom.
    reci_r = np.dot(xyz, np.linalg.inv(cell_lattice).T)

    # Compute the cell indices for each atom.
    cell_indices = np.floor(reci_r).astype(int)

    # Compute the cell index for each atom.
    cell_index = cell_indices[:, 0] + cell_indices[:, 1] * cells_per_side[0] + cell_indices[:, 2] * cells_per_side[0] * cells_per_side[1]

    # Sort the atoms by cell index.
    sorted_indices = np.argsort(cell_index)
    sorted_xyz = xyz[sorted_indices]
    sorted_cell_index = cell_index[sorted_indices]

    # Compute the number of atoms in each cell.
    cell_counts = np.bincount(sorted_cell_index)

    # Compute the index of the first atom in each cell.
    cell_offsets = np.zeros(len(cell_counts) + 1, dtype=np.int32)
    cell_offsets[1:] = np.cumsum(cell_counts)

    return CellList(sorted_xyz, cell_offsets)

