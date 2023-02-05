# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-05
# version: 0.0.1

from molpy import Box
from molpy.core.entity import Atom, Molecule
from molpy.core.typing import List
from molpy.core.utils import clone
import numpy as np

class Solvent:

    def __init__(self, box:Box, ):

        self.box = box

    def add_solvent(self, nSolvent, template:Molecule, seed=None)->List[Molecule]:

        rng = np.random.default_rng(seed)

        moles:List[Molecule] = []
        target = rng.uniform(0, self.box.L, size=(nSolvent, 3))

        for dr in target:
            m = clone(template)
            m.move_to(dr)
            moles.append(m)

        return moles
