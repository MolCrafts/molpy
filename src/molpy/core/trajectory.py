# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2024-02-05
# version: 0.0.1

from .frame import Frame

class Trajectory:

    def __init__(self):
        self._frames = []
        self._current_frame: int = 0

    def add_frame(self, frame: Frame):
        """Add a frame to the trajectory."""
        self._frames.append(frame)

    def get_frame(self, idx) -> Frame:
        """Get a frame from the trajectory."""
        return self._frames[idx]
    
    @property
    def current_frame(self) -> Frame:
        """Get the current frame."""
        return self._frames[self._current_frame]
    
    def set_current(self, idx: int):
        """Set the current frame."""
        if idx < 0 or idx >= len(self._frames):
            raise IndexError("Frame index out of range.")
        self._current_frame = idx
