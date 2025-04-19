from .frame import Frame
from .forcefield import ForceField
from .trajectory import Trajectory

class System:

    def __init__(
        self
    ):
        self._trajectory = Trajectory()
        self._forcefield = None

    def set_forcefield(self, forcefield: ForceField):
        """Set the forcefield for the system."""
        self.forcefield = forcefield

    @property
    def frame(self):
        return self._trajectory.current_frame
    
    def add_frame(self, timestep: int, frame: Frame):
        """Add a frame to the system."""
        self._trajectory.add_frame(timestep, frame)
        