# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-04
# version: 0.0.1

try:
    import jax.numpy as np
except:
    import numpy as np

def make_pbc(xyz: np.array, box: np.array):
    """
    dealing with the pbc shifts of vectors

    Parameters
    ----------
    xyz : np.array
        $(N, 3)$, a list of real space vectors in Cartesian coordinates
    box : np.array
        $(3, 3)$, the box matrix

    Returns
    -------
    np.array
        $(N, 3)$, the pbc-shifted vectors
    """
    unshifted_xyz = xyz.dot(np.linalg.inv(box))
    dsvecs = unshifted_xyz - np.floor(unshifted_xyz + 0.5)
    return dsvecs.dot(box)


    