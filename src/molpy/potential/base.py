from typing import Callable
import numpy as np
from molpy import Alias

class Potential:

    F:Callable|None = None

    def __new__(self, *args, **kwargs):
        if self.F is None:
            raise NotImplementedError("F method must be implemented")
        return super().__new__(self, *args, **kwargs)

    def __init__(self, name:str, type:str):
        self.name = name
        self.type = type

    def __call__(self,input):
        return self.forward(input)

    def forward(self):
        pass

    def energy(self):
        raise NotImplementedError("energy method must be implemented")

    def forces(self):
        raise NotImplementedError("energy method must be implemented")

class Potentials:

    def __init__(self, *potentials):
        self._potentials = potentials

    def __call__(self, frame):
        return self.forward(frame)

    def forward(self, frame):
        frame[Alias.energy] = 0
        frame.atoms[Alias.forces] = np.zeros((frame.n_atoms, 3))
        for potential in self._potentials:
            potential(frame)

        return frame

    @property
    def pairs(self):
        return self._potentials['pair']