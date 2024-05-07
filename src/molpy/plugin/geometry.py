from molpy import Plugin
from molpy import Alias
import numpy as np
import molpy as mp

def _move(xyz, vec):
    return xyz + vec

def move(self, vec):
    self._atoms[Alias.xyz] = _move(self._atoms[Alias.xyz], vec)

def _rot(xyz, degree, axis):
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degree)
    c = np.cos(theta)
    s = np.sin(theta)
    r = np.eye(3) * c + np.outer(axis, axis) * (1 - c) + np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ]) * s
    return np.dot(xyz, r.T)

def rot(self, degree:int|float, axis:np.ndarray):
    """
    rotate structu by using Rodrigues' rotation formula

    Args:
        degree (int|float): rotation degree
        axis (np.ndarray): rotation axis
    """
    self._atoms[Alias.xyz] = _rot(self._atoms[Alias.xyz], degree, axis)
    

class Geometry(Plugin):

    def when_import(self):

        setattr(mp.StaticStruct, 'move', move)
        setattr(mp.StaticStruct, 'rot', rot)