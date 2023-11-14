from ._matplotlib import *

system = MatplotlibSystem()

__all__ = ["plot_box", "plot_atoms"]

def plot_box(box):
    system.plot_box(box)

def plot_atoms(positions, type, colors=None, radius=None, isLabel=False):
    system.plot_atoms(positions, type, colors=None, radius=None, isLabel=False)

def plot_molecule(positions, connect, types=None, colors=None, radius=None, isLabel=False):
    system.plot_molecule(positions, connect, types, colors, radius, isLabel)

def show():
    return system.show()

def gcf():
    return system.fig

def gca():
    return system.ax