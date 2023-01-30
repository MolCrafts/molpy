# author: Roy Kid
# contact: lijichen365@126.com
<<<<<<< HEAD
# date: 2022-08-10
# version: 0.0.1

from .box import Box
=======
# date: 2023-01-08
# version: 0.0.1

from pathlib import Path
from .io_utils import load_trajectory, ChemfilesTrajectory
>>>>>>> cbf11e643d6cec0d32adcd29c5fc912790756dd4
from .frame import Frame

class Trajectory:

<<<<<<< HEAD
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

    def __len__(self):
        return len(self._frame)
=======
    def __init__(self):

        self._file_handler:ChemfilesTrajectory = None
        self.current_frame = None

    def __del__(self):

        self.close()

    def close(self):
        self._file_handler.close()

    @classmethod
    def load(cls, fileName:str|Path, mode:str='r', format:str=''):

        molpy_traj = cls()

        chemfile_traj = load_trajectory(str(fileName), mode, format)

        molpy_traj._file_handler = chemfile_traj

        return molpy_traj

    @property
    def file_handler(self)->ChemfilesTrajectory:

        return self._file_handler

    @file_handler.setter
    def file_handler(self, handler:ChemfilesTrajectory):
        
        if isinstance(handler, ChemfilesTrajectory):
            self._file_handler = handler
        else:
            raise TypeError('file_handler must be a ChemfilesTrajectory instance')

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
    
>>>>>>> cbf11e643d6cec0d32adcd29c5fc912790756dd4
