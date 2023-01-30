# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from chemfiles import Trajectory as ChemfilesTrajectory

def load_trajectory(fileName, mode:str='r', format:str='')->ChemfilesTrajectory:

    return ChemfilesTrajectory(fileName, mode, format)