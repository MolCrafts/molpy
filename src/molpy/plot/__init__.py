from ._matplotlib import *

_backend = MatplotlibBackend()

__all__ = ["plot_box", "plot_atoms"]

def plot_box(box):
    _backend.plot_box(box)

def plot_atoms(positions, type, colors=None, radius=None, isLabel=False):
    _backend.plot_atoms(positions, type, colors=None, radius=None, isLabel=False)

def show():
    return _backend.show()

def gcf():
    return _backend.fig

def gca():
    return _backend.ax