# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-05
# version: 0.0.1

from .frame import Frame

class Trajectory:

    def __init__(self):
        self._frames = {}
        self._current_frame = None

    def add_frame(self, frame: Frame, timestep: int|None):
        """Add a frame to the trajectory."""
        self._frames[timestep] = frame

    def get_frame(self, timestep: int|None = None) -> Frame:
        """Get a frame from the trajectory."""
        return self._frames.get(timestep, None)
    
    @property
    def current_frame(self) -> Frame:
        """Get the current frame."""
        if self._current_frame is None:
            raise ValueError("No current frame set.")
        return self._frames[self._current_frame]
    
    def set_current(self, timestep: int|None):
        """Set the current frame."""
        if timestep not in self._frames:
            raise ValueError(f"Frame {timestep} not found in trajectory.")
        self._current_frame = self._frames[timestep]
        return self._current_frame
