# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from .box import Box
from .frame import Frame

class Trajectory:

    def __init__(self, ):

        self._frame = []

    def add_frame(self, frame:Frame):

        self._frame.append(frame)

    @property
    def n_frames(self)->int:
        return len(self._frame)

    def __getitem__(self, idx:int)->Frame:
        return self._frame[idx]

    def append(self, frame:Frame):

        self._frame.append(frame)

    def extend(self, traj:'Trajectory'):

        self._frame.extend(traj)
