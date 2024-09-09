import numpy as np


def rotate(xyz, axis, theta):
    """ Rotate coordinates around an axis by an angle theta. Using the Rodrigues' rotation formula.

    Args:
        xyz (np.ndarray): coordinates
        axis (np.ndarray): rotation axis
        theta (float): rotation angle

    Returns:
        np.ndarray: rotated coordinates
    """
    axis = axis / np.linalg.norm(axis)

    rot = (
        xyz * np.cos(theta)
        + np.cross(axis, xyz) * np.sin(theta)
        + axis * np.dot(xyz, axis) * (1 - np.cos(theta))
    )

    return rot

def translate(xyz, vector):
    """ Translate coordinates by a vector.
    
    Args:
        xyz (np.ndarray): coordinates
        vector (np.ndarray): translation vector

    Returns:
        np.ndarray: translated coordinates
    """
    return xyz + vector