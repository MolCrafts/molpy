from .base import Integrator
from molpy import Frame, Alias

class VelocityVerlet(Integrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.

    Args:
        timestep (float): Integration time step in femto seconds.
    """

    ring_polymer = False
    pressure_control = False

    def __init__(self, timestep: float):
        super().__init__(timestep)

    def _main_step(self, frame: Frame):
        r"""
        Propagate the positions of the frame according to:

        ..math::
            q = q + \frac{p}{m} \delta t

        Args:
            frame (schnetpack.md.Frame): Frame class containing all molecules and their
                             replicas.
        """
        frame.atoms[Alias.xyz] = frame.atoms[Alias.xyz] + self.timestep * frame.atoms[Alias.momenta] / frame.atoms[Alias.mass]
