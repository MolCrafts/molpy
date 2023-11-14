# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-11-14
# version: 0.0.1

import matplotlib.pyplot as plt
import numpy as np
import molpy as mp

class System:
    pass

class MatplotlibSystem(System):

    def __init__(self):
        super().__init__()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_atoms(self, positions, types=None, colors=None, radius=None, isLabel=False):
        """plot atom(s) with positions and type"""
        from .palette import default_palette
        ax = self.ax
        if isinstance(positions, np.ndarray):
            positions = positions.reshape(-1, 3)
            positions = self.box.wrap(positions)

        if types is not None:
            if isinstance(types, np.ndarray):
                types = types.reshape(-1)
            assert positions.shape[0] == types.shape[0], "positions and atomic type must have same length"

            colors = []
            radius = []

            for i in range(positions.shape[0]):
                atom_repr = default_palette.atoms[types[i]]
                colors.append(atom_repr.CPK)
                radius.append(float(atom_repr.atomic_radius))

        colors = np.asarray(colors)
        radius = np.asarray(radius)

        assert len(colors) == len(positions), "colors and positions must have same length"
        assert len(radius) == len(positions), "radii and positions must have same length"

        ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, s=radius)
        if isLabel:
            ax.text(positions[:,0], positions[:,1], positions[:,2], types, ha="center", va="center")

    def plot_molecule(self, positions, connect, types=None, colors=None, radius=None, isLabel=False):
        self.plot_atoms(positions, types, colors, radius, isLabel)
        connect_xyz = positions[connect]
        self.ax.plot(connect_xyz[:,0], connect_xyz[:,1], connect_xyz[:,2], c="black")

    def plot_box(self, box: np.ndarray | mp.Box, title=None, image=[0, 0, 0], *args, **kwargs):
        """Helper function to plot a :class:`~.box.Box` object.

        Args:
            box (:class:`~.box.Box`):
                Simulation box.
            title (str):
                Title of the graph. (Default value = :code:`None`).
            ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
                If :code:`None`, make a new axes and figure object.
                If plotting a 3D box, the axes must be 3D.
                (Default value = :code:`None`).
            image (list):
                The periodic image location at which to draw the box (Default
                value = :code:`[0, 0, 0]`).
            ``*args``, ``**kwargs``:
                All other arguments are passed on to
                :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.
        """
        ax = self.ax
        is2D = False  # 2d is not supported yet
        box = mp.Box(box)
        self.box = box
        if is2D:
            # Draw 2D box
            corners = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
            # Need to copy the last point so that the box is closed.
            corners.append(corners[0])
            corners = np.asarray(corners)
            corners += np.asarray(image)
            corners = box.make_absolute(corners)[:, :2]
            color = kwargs.pop("color", "k")
            ax.plot(corners[:, 0], corners[:, 1], color=color, *args, **kwargs)
            ax.set_aspect("equal", "datalim")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
        else:
            # Draw 3D box
            corners = np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            )
            corners += np.asarray(image)
            corners = box.make_absolute(corners)
            paths = [
                corners[[0, 1, 3, 2, 0]],
                corners[[4, 5, 7, 6, 4]],
                corners[[0, 4]],
                corners[[1, 5]],
                corners[[2, 6]],
                corners[[3, 7]],
            ]
            for path in paths:
                color = kwargs.pop("color", "k")
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ratio = corners[-1] - corners[0]
            ax.set_box_aspect(ratio)

    def show(self):
        return self.fig