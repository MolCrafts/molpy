# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-10
# version: 0.0.1

import numpy as np
from numpy.random import default_rng

boltz = 1.380649e-23  # m^2 kg s^-2 K^-1

class Command:

    pass

class Compute:

    pass

class ComputeTemperature(Compute):

    def compute_scalar(self, velocity, mass):
        """
        compute temperature from velocity

        Parameters
        ----------
        velocity : array
            velocity of atoms

        Returns
        -------
        temperature : float
            temperature of atoms
        """
        t = np.sum(np.square(velocity)) * mass
        dof = 3 * len(velocity)  # degree of freedom
        # dof -= extra_dof + fix_dof
        # if dof > 0
        # mvv2e = 48.88821291 * 48.88821291
        tfactor = 1 / (dof * boltz)
        return t * tfactor
        

class Velocity(Command):

    def create(self, natoms, T, seed):

        rng = default_rng(seed)
        velocity = rng.uniform(-0.5, 0.5, size=(natoms, 3))
        return self.scale(velocity, T)

    def scale(self, velocity, mass, T_desired):

        t = self.compute_temperature(velocity, mass)

        return self.rescale(velocity, t, T_desired)

    def rescale(self, velocity, T_old, T_new):
        """
        rescale velocities of atoms to a new temperature

        Parameters
        ----------
        T_old : float
        T_new : float
        """
        factor = np.sqrt(T_new / T_old)
        return velocity * factor

    def compute_temperature(self, velocity, mass):
        """
        compute temperature from velocity

        Parameters
        ----------
        velocity : array
            velocity of atoms

        Returns
        -------
        temperature : float
            temperature of atoms
        """
        
        self._compute_temperature = ComputeTemperature()
        return self._compute_temperature.compute_scalar(velocity, mass)

