# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1

from .struct import ArrayDict
from .graph import Graph
import numpy as np

class Topology:

    def __init__(self):

        self.reset()

    def reset(self):

        self.bonds = ArrayDict()
        self.angles = ArrayDict()
        self.dihedrals = ArrayDict()
        self.impropers = ArrayDict()

    @property
    def nbonds(self):
        return self.bonds.length

    @property
    def nangles(self):
        return self.angles.length

    @property
    def ndihedrals(self):
        return self.dihedrals.length

    @property
    def nimpropers(self):
        return self.impropers.length

    @property
    def connect(self)->np.ndarray:
        return np.array(self.bonds.get('index', []))

    def add_bonds(self, connect, **properties):
        connect = np.atleast_2d(connect)
        assert connect.ndim == 2 and connect.shape[-1] == 2
        connect = np.sort(connect, axis=1)
        self.bonds['index'] = connect

        for key, value in properties.items():
            if len(value) == self.bonds:
                self.bonds[key] == value

        # for i, j in connect:
        #     self._graph.set_edge(i, j)

    def add_angles(self, connect, **properties):

        assert connect.ndim == 2 and connect.shape[-1] == 3
        edges = np.sort(connect[:, :2], axis=1)  # ijk, i < k
        connect[:, [0, 1]] = edges
        self.angles['index'] = connect

        for key, value in properties.items():
            if len(value) == self.angles:
                self.angles[key] == value

        # for i, j, k in connect:
        #     self._graph.set_edge(i, j)
        #     self._graph.set_edge(j, k)

    def add_dihedrals(self, connect, **properties):

        assert connect.ndim == 2 and connect.shape[-1] == 4
        mask = connect[:, 1] > connect[:, 2]  # ijkl, j > k
        dihe = np.where(mask.reshape((-1, 1)), connect, connect[:, [3, 2, 1, 0]])  # ijkl->lkji

        self.dihedrals['index'] = dihe

        for key, value in properties.items():
            if len(value) == self.dihedrals:
                self.dihedrals[key] == value

        # for i, j, k, l in connect:
        #     self._graph.set_edge(i, j)
        #     self._graph.set_edge(j, k)
        #     self._graph.set_edge(k, l)

    def add_impropers(self, connect, **properties):

        assert connect.ndim == 2 and connect.shape[-1] == 4
        connect[:, 1:] = np.sort(connect[:, 1:], axis=1) # ijkl, j < k < l
        self.impropers['index'] = connect

        for key, value in properties.items():
            if len(value) == self.impropers:
                self.impropers[key] == value

        # for i, j, k, l in connect:
        #     self._graph.set_edge(i, j)
        #     self._graph.set_edge(j, k)
        #     self._graph.set_edge(k, l)

    # def add_residue(self, id, name, mask, **properties):

    #     self.residues['id'].append(id)
    #     self.residues['name'].append(name)
    #     self.residues['mask'].append(mask)
    #     for key, value in properties.items():
    #         self.residues[key].append(value)

    # def get_residue(self, name)->dict:

    #     idx = self.residues.index('name', name)
    #     if idx is None:
    #         raise KeyError('Residue {} not found'.format(name))
    #     else:
    #         return self.residues[idx]
