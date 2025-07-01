import numpy as np
from molpy.core.atomistic import Atomistic

class CH3(Atomistic):
    def __init__(self):
        super().__init__()
        C1 = self.def_atom(name="C1", type="C", xyz=np.array([0, 0, 0]))
        H1 = self.def_atom(name="H1", type="H", xyz=np.array([0, 0, 1]))
        H2 = self.def_atom(name="H2", type="H", xyz=np.array([1, 0, 0]))
        H3 = self.def_atom(name="H3", type="H", xyz=np.array([0, 1, 0]))
        self.def_bond(C1, H1)
        self.def_bond(C1, H2)
        self.def_bond(C1, H3)
