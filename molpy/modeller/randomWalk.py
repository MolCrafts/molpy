# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-12
# version: 0.0.1

from molpy_kernel import SimpleRandomWalk as SRW
import numpy as np
from numpy.random import default_rng

from molpy.core.frame import StaticFrame

class SimpleRandomWalk(SRW):

    def __init__(self, box):
        super().__init__()
        self.box = box

    def find_start(self, seed=None):

        rng = default_rng(seed)
        start_point = rng.random((3, )) * self.box.lengths
        return start_point

    def linear(self, nsteps, step_size, start_point=None, seed=None)->StaticFrame:
        """
        generate a linear molecule

        Parameters
        ----------
        nsteps : int
            nsteps of linear molecule
        start_point : (3, ), optional
            start coordinate of linear. If none, start at random position. By default None
        """
        if start_point is None:
            start_point = self.find_start()

        rng = default_rng(seed)

        traj = self.walk(nsteps, step_size, start_point, rng.integers(0, 2**10))
        atom_idx = np.arange(nsteps, dtype=int)
        topo = np.concatenate([atom_idx, atom_idx+1], dtype=int).T
        sframe = StaticFrame.from_dict({'x': traj[:, 0], 'y': traj[:, 1], 'z': traj[:, 2]}, box=self.box)
        for i, j in topo:
            sframe.add_bond(i, j)
        return sframe
