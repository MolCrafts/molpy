import numpy as np
import molpy as mp

from ..base import Calculator

class Minimizer(Calculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def minimize(self, *args, **kwargs):
        pass

    def before_minimize(self, frame, potentials):

        if mp.Alias.R not in frame.atoms:
            frame.atoms[mp.Alias.R] = frame.positions
        if mp.Alias.energy not in frame.atoms:
            frame[mp.Alias.energy] = 0
        if mp.Alias.forces not in frame.atoms:
            frame.atoms[mp.Alias.forces] = np.zeros((frame.n_atoms, 3))
        if mp.Alias.momenta not in frame.atoms:
            frame.atoms[mp.Alias.momenta] = 0
        if mp.Alias.step not in frame.atoms:
            frame[mp.Alias.step] = 0
        