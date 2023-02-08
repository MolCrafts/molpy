# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-08
# version: 0.0.1

from typing import Callable, Literal, Optional
from ..core.box import Box
try:
    import jax.numpy as np
    JAX_AVAILABLE = True
    import jax
except:
    import numpy as np
    JAX_AVAILABLE = False


class BasePairPotential:

    def __init__(self, name:str):
        self.name = name

    def energy(self):
        raise NotImplementedError

    def force(self):
        raise NotImplementedError

    def energy_and_force(self):
        raise NotImplementedError

class LJ126(BasePairPotential):

    def __init__(self, rCutoff:float, rSwitch:float=0.0, isPBC:bool=True):

        super().__init__('lj126')
        self.rCutoff = rCutoff
        self.rSwitch = rSwitch
        self.isSwitch = rSwitch > 0
        self.isPBC = isPBC

    def energy_kernel(self, dr, eps, sig):

        dr_norm = np.linalg.norm(dr, axis=1)

        # NOTE: all dr_norm < rCutoff 
        # It is guaranteed by the neighborlist
        # cutoffMask = dr_norm < self.rCutoff
        # dr_norm = dr_norm[cutoffMask]
        # eps = eps[cutoffMask]
        # sig = sig[cutoffMask]

        dr_inv = 1.0 / dr_norm
        sig_dr = sig * dr_inv
        sig_dr6 = np.power(sig_dr, 6)
        sig_dr12 = np.power(sig_dr6, 2)
        E = 4.0 * eps * (sig_dr12 - sig_dr6)

        if self.isSwitch:
            x = (dr_norm - self.rSwitch) / (self.rCutoff - self.rSwitch)
            S = 1 - 6. * x ** 5 + 15. * x ** 4 - 10. * x ** 3
            np.where(dr_norm > self.rSwitch, E, E * S)
        
        E = np.where(dr_norm > self.rCutoff, 0.0, E)
        return E

    def energy(self, xyz, pairs, types, epsilon, sigma, box, tDistance, mScales):
        type1 = types[pairs[:, 0]]
        type2 = types[pairs[:, 1]]
        eps = epsilon[type1, type2]
        sig = sigma[type1, type2]
        box = Box.from_matrix(box)
        dr = box.displacement(xyz[pairs[:, 0]], xyz[pairs[:, 1]])

        return self.energy_kernel(dr, eps, sig)

    def force_kernel(self, dr, eps, sig):

        dr_norm = np.linalg.norm(dr, axis=1)
        dr_inv = 1.0 / dr_norm
        sig_dr = sig * dr_inv
        sig_dr13 = np.power(sig_dr, 13)
        sig_dr7 = np.power(sig_dr, 7)

        F = 48.0 * eps/sig * (sig_dr13 - 0.5 * sig_dr7)
        E = np.where(dr_norm > self.rCutoff, 0.0, E)
        return F

    def force(self, xyz, pairs, types, epsilon, sigma, box, tDistance, mScales):

        type1 = types[pairs[:, 0]]
        type2 = types[pairs[:, 1]]
        eps = epsilon[type1, type2]
        sig = sigma[type1, type2]
        box = Box.from_matrix(box)     
        dr = box.displacement(xyz[pairs[:, 0]], xyz[pairs[:, 1]])   

        if not JAX_AVAILABLE:
            self.force_kernel(dr, eps, sig)
        else:
            _force_kernel = lambda *args, **kwargs: np.sum(self.energy_kernel(*args, **kwargs))
            return jax.grad(_force_kernel)(dr, eps, sig)