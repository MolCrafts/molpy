# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2024-02-05
# version: 0.0.1

from .frame import Frame
import numpy as np

class Trajectory:

    def __init__(self, frames: list[Frame] = []):
        self._frames = frames

    def append(self, frame: Frame):
        """Add a frame to the trajectory."""
        self._frames.append(frame)

    def get_frame(self, timestep: int) -> Frame|None:
        """Get a frame from the trajectory."""
        return next((frame for frame in self._frames if frame['timestep'] == timestep), None)
    
    @property
    def timesteps(self) -> list[int]:
        """Get the list of timesteps."""
        return [self['timestep'] for frame in self.frames]
    
    @property
    def frames(self) -> list[Frame]:
        """Get the frames in the trajectory."""
        return self._frames

    def __getitem__(self, key: int|str) -> Frame|np.ndarray:
        """Get a frame by timestep or index."""
        if isinstance(key, int):
            return self._frames[key]
        # TODO: traj["atoms"][["x", "y", "z"]] -> ArrayDict?
        # elif isinstance(key, str):
        #     return (frame[key] for frame in self._frames)