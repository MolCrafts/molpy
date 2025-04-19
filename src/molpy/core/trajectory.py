# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2024-02-05
# version: 0.0.1

from .frame import Frame

class Trajectory:

    def __init__(self):
        self._frames = {}
        self._current_frame: Frame|None = None

    def add_frame(self, timestep: int, frame: Frame):
        """Add a frame to the trajectory."""
        self._frames[timestep] = frame

    def get_frame(self, timestep: int) -> Frame:
        """Get a frame from the trajectory."""
        if timestep not in self._frames:
            raise ValueError(f"Frame at timestep {timestep} not found.")
        return self._frames[timestep]
    
    @property
    def current_frame(self) -> Frame:
        """Get the current frame."""
        return self._current_frame
    
    def set_current(self, timestep: int):
        """Set the current frame."""
        self._current_frame = self.get_frame(timestep)