import numpy as np

def rotate_by_rodrigues(xyz, axis, theta):
    """ Rotate coordinates around an axis by an angle theta. Using the Rodrigues' rotation formula.

    Args:
        xyz (np.ndarray): coordinates
        axis (np.ndarray): rotation axis
        theta (float): rotation angle

    Returns:
        np.ndarray: rotated coordinates
    """
    xyz = np.asarray(xyz)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    original_shape = xyz.shape
    was_1d = xyz.ndim == 1
    
    xyz = np.atleast_2d(xyz)

    rot = (
        xyz * np.cos(theta)
        + np.cross(axis, xyz) * np.sin(theta)
        + axis * np.dot(xyz, axis)[..., None] * (1 - np.cos(theta))
    )

    # If input was 1D, squeeze the result back to 1D
    if was_1d and rot.shape[0] == 1:
        rot = rot.squeeze(0)
    
    return rot

def rotate_by_quaternion(xyz, q):
    """ Rotate coordinates using a quaternion.

    Args:
        xyz (np.ndarray): coordinates
        q (np.ndarray): quaternion

    Returns:
        np.ndarray: rotated coordinates
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rot_mat = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])
    return np.dot(xyz, rot_mat)


def translate(xyz, vector):
    """ Translate coordinates by a vector.
    
    Args:
        xyz (np.ndarray): coordinates
        vector (np.ndarray): translation vector

    Returns:
        np.ndarray: translated coordinates
    """
    return xyz + vector