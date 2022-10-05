# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-05
# version: 0.0.1

import numpy as np

def create_state(xyz, velocity, mass):

    state = {
        'xyz': xyz,
        'velocity': velocity,
        'mass': mass,
        'force': np.zeros_like(xyz),
        'potential': 0.0,
        'kinetic': 0.0,
        'temperature': 0.0,
    }

    return state

class Integrator:

    pass

class Verlet(Integrator):

    """
    Verlet integrator implements the leap-frog Verlet integration method. The positions and velocities stored in the context are offset from each other by half a time step. In each step, they are updated as follows:

    $$
        v(t + 1/2 dt) = v(t) + 1/2 dt * f(t) / m
        r(t + dt) = r(t) + dt * v(t + 1/2 dt)
        v(t + dt) = v(t + 1/2 dt) + 1/2 dt * f(t + dt) / m
    $$

    """

    def __init__(self, dt):
        self.dt = dt

    def initial_integrate(self, r, v, f, m):
        """
        first half-step

        Parameters
        ----------
        r : coordinates
        v : velocities
        f : forces
        m : masses

        Returns
        -------
        v : mid-step velocities
        r : full-updated coordinates
        """
        v += 0.5 * self.dt * f / np.repeat(m, 3).reshape(-1, 3)
        r += self.dt * v
        return r, v

    def final_integrate(self, v, f, m):
        """
        second half-step

        Parameters
        ----------
        v : mid-step velocities
        f : forces
        m : masses

        Returns
        -------
        v : full-updated velocities
        """
        v += 0.5 * self.dt * f / np.repeat(m, 3).reshape(-1, 3)
        return v