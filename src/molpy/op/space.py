from molpy import Plugin
from molpy import Alias
import numpy as np
import molpy as mp
from functools import singledispatch


def move_op(xyz, vec):
    return xyz + vec


@singledispatch
def move(
    xyz_or_struct: np.ndarray | mp.Struct, vec: np.ndarray
) -> np.ndarray | mp.Struct:
    """
    Move the xyz coordinates by vec.

    Args:
        xyz (np.ndarray|mp.Struct): The xyz coordinates or the structure to move.
        vec (np.ndarray): The vector to move by.
    """
    ...


@move.register(mp.Struct)
def _(struct: mp.Struct, vec: np.ndarray)->mp.Struct:
    """
    Move the xyz coordinates by vec.

    Args:
        xyz (mp.Struct): The xyz coordinates or the structure to move.
        vec (np.ndarray): The vector to move by.
    """
    struct.atoms[Alias.xyz] = move_op(struct.atoms[Alias.xyz], vec)
    return struct


@move.register(np.ndarray)
def _(xyz:np.ndarray, vec:np.ndarray) -> np.ndarray:
    """
    Move the xyz coordinates by vec.

    Args:
        xyz (np.ndarray): The xyz coordinates or the structure to move.
        vec (np.ndarray): The vector to move by.
    """
    return move_op(xyz, vec)


def rot_op(xyz, degree, axis):
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degree)
    c = np.cos(theta)
    s = np.sin(theta)
    r = (
        np.eye(3) * c
        + np.outer(axis, axis) * (1 - c)
        + np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        * s
    )
    return np.dot(xyz, r.T)
