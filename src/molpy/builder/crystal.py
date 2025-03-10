import numpy as np
from molpy.core import Region, Atom

class FCC:

    def __init__(self, region: Region, lattice_constant):
        self.region = region
        self.lattice_constant = lattice_constant

    def create_lattice(self)->np.ndarray:

        xlo, xhi = self.region.xlo, self.region.xhi
        ylo, yhi = self.region.ylo, self.region.yhi
        zlo, zhi = self.region.zlo, self.region.zhi

        dx = xhi - xlo
        dy = yhi - ylo
        dz = zhi - zlo
        nx = int(dx // self.lattice_constant)
        ny = int(dy // self.lattice_constant)
        nz = int(dz // self.lattice_constant)

        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])

        grid = np.array(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))).T.reshape(-1, 3)
        lattice_points = grid[:, None, :] + basis[None, :, :]
        lattice_points = lattice_points.reshape(-1, 3) * self.lattice_constant
        lattice_points[:, 0] += xlo
        lattice_points[:, 1] += ylo
        lattice_points[:, 2] += zlo
        return lattice_points[self.region.isin(lattice_points)]
    
    def fill(self, atom: Atom):
        lattice = self.create_lattice()
        for x, y, z in lattice:
            atom.add_atom(x, y, z)
        return atom