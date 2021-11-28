# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from typing import Literal
import numpy as np
from math import pi

# halfPI = 0.5 * pi


def factory_ndarray_dtype_list(shape):
    return np.frompyfunc(list, 0, 1)(np.empty(shape, dtype=list))


def getPos(atom):
    return getattr(atom, "position")


map_getPos = np.vectorize(getPos, otypes=[np.ndarray])


class Box:
    def __init__(self, boundary_condition: Literal["ppp"], **kwargs) -> None:

        # ref:
        # https://docs.lammps.org/Howto_triclinic.html
        # https://hoomd-blue.readthedocs.io/en/stable/box.html

        """Describes the properties of the box"""

        self.boundary_condition = boundary_condition
        self._boundary_condition = self.boundary_condition
        self._x_bc, self._y_bc, self._z_bc = self.boundary_condition

        self._pbc = np.asarray([i == "p" for i in boundary_condition])

        if "xhi" in kwargs:
            self.defByEdge(**kwargs)

        elif "a1" in kwargs:
            self.defByLatticeVectors(**kwargs)

        elif "lx" in kwargs:
            self.defByBoxLength(**kwargs)

    def defByEdge(
        self, xhi, yhi, zhi, xlo=0.0, ylo=0.0, zlo=0.0, xy=0.0, xz=0.0, yz=0.0
    ):
        """define Box via start and end which in LAMMPS and hoomd-blue style"""

        self.xlo = xlo
        self.ylo = ylo
        self.zlo = zlo
        self.xhi = xhi
        self.yhi = yhi
        self.zhi = zhi

        self.lx = self.xhi - self.xlo
        self.ly = self.yhi - self.ylo
        self.lz = self.zhi - self.zlo

        self.xy = xy
        self.xz = xz
        self.yz = yz
        radian2degree = 180 / np.pi

        b = np.sqrt(self.ly ** 2 + xy ** 2)
        c = np.sqrt(self.lz ** 2 + xz ** 2 + yz ** 2)
        self.gamma = np.arccos(xy / b) * radian2degree
        self.beta = np.arccos(xz / c) * radian2degree
        self.alpha = np.arccos((xy * xz + self.ly * yz) / (c * b)) * radian2degree
        self._post_def_()

    def defByLatticeVectors(self, a1, a2, a3):
        """define Box via lattice vector

        Args:
            a1 (np.ndarray): Must lie on the x-axis
            a2 (np.ndarray): Must lie on the xy-plane
            a3 (np.ndarray)
        """

        """
        self.lx = np.linalg.norm(a1)
        a2x = a1@a2/np.linalg.norm(a1)
        self.ly = np.sqrt(a2@a2-a2x**2)
        self.xy = a2x/self.ly
        crossa1a2 = np.cross(a1, a2)
        self.lz = np.linalg.norm(a3) * (crossa1a2/np.linalg.norm(crossa1a2))
        a3x = a1@a3/np.linalg.norm(a1)
        self.xz = a3x/self.lz
        self.yz = (a2@a3-a2x*a3x)/self.ly/self.lz
        """
        if a1[0] <= 0 or a1[1] != 0 or a1[2] != 0:
            raise ValueError("a1 vector must lie on the positive x-axis")
        else:
            self.xhi = self.lx = a1[0]

        if a2[1] <= 0 or a2[2] != 0:
            raise ValueError(
                "a2 vector must lie on the xy plane, with strictly positive y component"
            )
        else:
            self.yhi = self.ly = a2[1]
            self.xy = a2[0]

        if a3[2] <= 0:
            raise ValueError("a2 vector must own a strictly positive z component")
        else:
            self.zhi = self.lz = a3[2]
            self.xz = a3[0]
            self.yz = a3[1]

        self.xlo = 0
        self.ylo = 0
        self.zlo = 0

        self._post_def_()

    def defByBoxLength(self, lx, ly, lz, alpha=90, beta=90, gamma=90):
        """define the Box via edge lengthes and angles between the edges"""
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

        degree2rad = np.radians(1.0)

        self.xlo = 0
        self.ylo = 0
        self.zlo = 0
        self.xhi = lx
        xy = ly * np.cos(gamma * degree2rad)
        xz = lz * np.cos(beta * degree2rad)
        self.yhi = np.sqrt(ly * ly - xy ** 2)
        yz = (ly * lz * np.cos(alpha * degree2rad) - xy * xz) / self.yhi
        self.zhi = np.sqrt(lz * lz - xz ** 2 - yz ** 2)

        self.lx = self.xhi - self.xlo
        self.ly = self.yhi - self.ylo
        self.lz = self.zhi - self.zlo

        self.xy = xy
        self.xz = xy
        self.yz = yz

        self._post_def_()

    def _post_def_(self):
        # box basis
        self._basis = np.array(
            [[self.lx, 0, 0], [self.xy, self.ly, 0], [self.xz, self.yz, self.lz]]
        )
        self._orthorhombic = not (np.flatnonzero(self._basis) % 4).any()

    @property
    def basis(self):
        return self._basis

    @property
    def x_vec(self):
        return self._basis[0]

    @property
    def y_vec(self):
        return self._basis[1]

    @property
    def z_vec(self):
        return self._basis[2]

    @property
    def origin(self):
        return np.array([self.xlo, self.ylo, self.zlo])

    @property
    def volume(self):
        return np.abs(np.linalg.det(self._basis))

    def angles(self):
        return np.asarray([self.alpha, self.beta, self.gamma])

    def lengths(self):
        return np.linalg.norm(self._basis, axis=1)

    @property
    def orthorhombic(self):
        return self._orthorhombic

    @property
    def pbc(self):
        return self._pbc

    def wrap(self, position):
        """wrap position(s) array back into box

        Args:
            position (np.ndarray)

        Returns:
            np.ndarray: wrapped position(s)
        """
        print("wrap")
        edge_length = np.array([self.lx, self.ly, self.lz])
        print(edge_length)
        offset = np.floor_divide(position, edge_length)
        return position - offset * edge_length

    def displace(self, pos1, pos2):
        """find the distance between two positions in minimum image convention(mic)

        Args:
            pos1 (np.ndarray)
            pos2 (np.ndarray)

        Returns:
            np.ndarray
        """
        return self.wrap(pos1) - self.wrap(pos2)


class CellList:
    def __init__(self, box, rcutoff) -> None:

        assert isinstance(box, Box), TypeError(
            f"box must be an instance of Box class, but {type(box)}"
        )

        self.box = box
        self.rcutoff = rcutoff

    def build(self):

        box = self.box.basis
        rc = self.rcutoff

        box_inv = np.linalg.inv(box)
        len_abc_star = np.sqrt(np.sum(box_inv * box_inv, axis=0))
        h_box = 1 / len_abc_star  # the "heights" of the box

        # cell_vec := n_cell_x, n_cell_y, n_cell_z
        self.cell_vec = np.floor_divide(h_box, rc).astype(np.int32)
        self.cell_list = factory_ndarray_dtype_list(self.cell_vec)
        self.ncell = np.prod(self.cell_vec, dtype=int)
        self.cell_size = np.diag(1.0 / self.cell_vec).dot(box)
        self.cell_inv = np.linalg.inv(self.cell_size)

    def reset(self):

        self.build()

    def update(self, atoms, rebuild=True):

        if rebuild:
            self.build

        positions = np.vstack(map_getPos(atoms))

        wrapped_pos = self.box.wrap(positions)

        indices = np.floor(np.dot(wrapped_pos, self.cell_inv)).astype(int)

        for cell, atom in zip(
            self.cell_list[indices[:, 0], indices[:, 1], indices[:, 2]], atoms
        ):
            cell.append(atom)

    def getAdjCell(self, center_index, radius=1):

        cell_list = self.cell_list

        # edge length
        length = radius * 2 + 1

        # length^3 exclude (0, 0, 0)
        adj_cell_list = factory_ndarray_dtype_list((length ** 3 - 1))  # 1-d ndarray

        index = 0
        for adj_increment_index in np.ndindex((length, length, length)):

            adj_index = center_index + np.array(adj_increment_index) - radius
            if np.array_equal(adj_increment_index, np.array([0, 0, 0])):
                continue

            # wrap
            offset = np.floor_divide(adj_index, self.cell_vec)
            adj_index -= offset * self.cell_vec

            adj_cell_list[index] = cell_list[tuple(adj_index)]
            index += 1

        return adj_cell_list


class NeighborList:
    def __init__(self, cellList) -> None:

        self.rcutoff = cellList.rcutoff
        self.cellList = cellList
        self.box = cellList.box
        self.build()

    def build(self):

        self.neighbor_list = {}  # DefaultDict(list)

    def reset(self):

        self.build()

    def update(self, atoms=None):

        if atoms is not None:
            self.cellList.update(atoms)

        cellList = self.cellList.cell_list
        for index, cell in np.ndenumerate(cellList):
            adj_cells = self.cellList.getAdjCell(index)  # (nadj, ) dtype=list
            for atom in cell:

                # TODO: get positions from adj cells cost half of time
                cen_pos = atom.position  # center atom's position
                adj_atoms = []
                for cell in adj_cells:
                    adj_atoms.extend(cell)
                # map_extend = np.vectorize(adj_atoms.extend)
                # map_extend(adj_cells)
                adj_atoms = np.asarray(adj_atoms)

                pair_pos = np.vstack(map_getPos(adj_atoms))

                # pair_pos = np.array(map(lambda atom: getattr(atom, 'position'), adj_atoms))
                dr = pair_pos - cen_pos
                distance = np.linalg.norm(dr)
                nei_atoms = adj_atoms[distance < self.rcutoff]
                self.neighbor_list[atom] = {
                    "neighbors": nei_atoms,
                    "dr": dr,
                    "distance": distance,
                }