# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from chemfiles import Trajectory as ChemfilesTrajectory
from chemfiles import UnitCell, Atom
import numpy as np

def load_trajectory(fileName, mode:str='r', format:str='')->ChemfilesTrajectory:

    return ChemfilesTrajectory(fileName, mode, format)

def box2cell(box):
    angles = np.array([
            np.rad2deg(np.arccos(box.xy / box.L[1])),
            np.rad2deg(np.arccos(box.xz / box.L[2])),
            np.rad2deg(np.arccos(box.yz / box.L[0]))
        ])
    return UnitCell(box.L, angles)

def atom2atom(atom):
    _atom = Atom(atom.name, atom.type)
    _atom.mass = atom.mass
    _atom.charge = atom.charge
    return Atom(atom['name'])