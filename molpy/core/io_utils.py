# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from chemfiles import Trajectory as ChemfilesTrajectory
from chemfiles import UnitCell

def load_trajectory(fileName, mode:str='r', format:str='')->ChemfilesTrajectory:

    return ChemfilesTrajectory(fileName, mode, format)

def box2cell(box):

    return UnitCell((box.lengths[0], box.lengths[1], box.lengths[2]), (0, 0, 0))