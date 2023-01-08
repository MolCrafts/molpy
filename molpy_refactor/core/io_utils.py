# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from chemfiles import Trajectory

def load_trajectory(fileName)->Trajectory:

    with Trajectory(fileName) as traj:
        return traj