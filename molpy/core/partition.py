# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-10
# version: 0.0.1

from .box import Box
import numpy as np

class CellList:

    def __init__(self, box, minimum_cell_size):

        self.box = Box.from_matrix(box)
        self.minimum_cell_size = minimum_cell_size

    def calc_cell_dimensions(self, box_size, minimum_cell_size):
        """Compute the number of cells-per-side and total number of cells in a box."""
        cells_per_side = np.floor(box_size / minimum_cell_size)
        cell_size = box_size / cells_per_side

        cell_count = np.prod(cells_per_side)

    def create_buffer(self, xyz, box_size, minimum_cell_size):
        """Create a buffer around the box to avoid periodic boundary conditions."""
        cell_size = self.calc_cell_dimensions(box_size, minimum_cell_size)
        buffer_size = cell_size / 2
        buffer_box = box_size + 2 * buffer_size
        buffer_xyz = xyz + buffer_size

        return buffer_xyz, buffer_box

