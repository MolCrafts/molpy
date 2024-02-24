import numpy as np
from molpy import Alias

class Potential:

    def __init__(self, name:str, type:str):
        self.name = name
        self.type = type

    def __call__(self,input):
        return self.forward(input)

    def forward(self):
        pass

    def energy(self):
        pass

    def forces(self):
        pass

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