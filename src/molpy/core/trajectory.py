# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-05
# version: 0.0.1

from .frame import Frame

class Trajectory:

    def __init__(self, name):
        self.name = name
        self._frames = []

    def add_frame(self, frame: Frame):
        self._frames.append(frame)

    @property
    def n_frames(self):
        return len(self._frames)
    
    def __getitem__(self, T):
        return self._frames[T]