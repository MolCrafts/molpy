# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .io_utils import load_trajectory, Trajectory
from .frame import Frame

class Trajectory:

    def __init__(self):

        self._file_handler = None
        self.current_frame = None

    def __del__(self):

        if isinstance(self._file_handler, Trajectory):
            self._file_handler.close()

    @classmethod
    def load(cls, fileName):

        molpy_traj = cls()

        chemfile_traj = load_trajectory(fileName)

        molpy_traj._file_handler = chemfile_traj

        return molpy_traj

    @property
    def file_handler(self)->Trajectory:

        return self._file_handler

    @file_handler.setter
    def file_handler(self, handler:Trajectory):
            
        self._file_handler = handler

    def read_step(self, step):

        chemfile_frame = self._file_handler.read_step(step)
        molpy_frame = Frame.from_chemfile_frame(chemfile_frame)
        self.current_frame = molpy_frame
        return molpy_frame

    def read(self):

        chemfile_frame = self._file_handler.read()
        molpy_frame = Frame.from_chemfile_frame(chemfile_frame)
        self.current_frame = molpy_frame
        return molpy_frame

    @property
    def nsteps(self):
        return self._file_handler.nsteps
    
    @property
    def path(self):
        return self._file_handler.path
    