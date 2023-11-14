import numpy as np
from . import global_figure
from . import default_palette

__all__ = ["plot_atoms"]

def plot_atoms(position, atype, ax=None, **kwargs):
    """plot atom(s) with position and type"""
    ax = ax or global_figure.ax
    if isinstance(position, np.ndarray):
        position = position.reshape(-1, 3)
    if isinstance(atype, np.ndarray):
        atype = atype.reshape(-1)
    assert position.shape[0] == atype.shape[0], "position and atomic type must have same length"
    for i in range(position.shape[0]):
        atom_repr = default_palette.atoms[atype[i]]
        color = atom_repr.CPK
        radius = float(atom_repr.atomic_radius)
        ax.scatter(*position[i], c=color, s=radius, **kwargs)
        # ax.text(*position[i], atype[i], ha="center", va="center", **kwargs)
    return ax
