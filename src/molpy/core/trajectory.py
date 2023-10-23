# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2023-01-08
# # version: 0.0.1

# from pathlib import Path
# from .io_utils import load_trajectory, ChemfilesTrajectory
# from .frame import Frame

# class Trajectory:

#     def __init__(self):

#         self._file_handler:ChemfilesTrajectory = None
#         self.current_frame = None

#     def __del__(self):

#         self.close()

#     def close(self):
#         self._file_handler.close()

#     @classmethod
#     def load(cls, fileName:str|Path,  format:str=''):

#         molpy_traj = cls()

#         chemfile_traj = load_trajectory(str(fileName), 'r', format)

#         molpy_traj._file_handler = chemfile_traj

#         return molpy_traj

#     @property
#     def file_handler(self)->ChemfilesTrajectory:

#         return self._file_handler

#     @file_handler.setter
#     def file_handler(self, handler:ChemfilesTrajectory):
        
#         if isinstance(handler, ChemfilesTrajectory):
#             self._file_handler = handler
#         else:
#             raise TypeError('file_handler must be a ChemfilesTrajectory instance')

#     def read_step(self, step):

#         chemfile_frame = self._file_handler.read_step(step)
#         molpy_frame = Frame.from_chemfile_frame(chemfile_frame)
#         self.current_frame = molpy_frame
#         return molpy_frame

#     def read(self):

#         chemfile_frame = self._file_handler.read()
#         molpy_frame = Frame.from_chemfile_frame(chemfile_frame)
#         self.current_frame = molpy_frame
#         return molpy_frame

#     @property
#     def nsteps(self):
#         return self._file_handler.nsteps
    
#     @property
#     def path(self):
#         return self._file_handler.path
    
