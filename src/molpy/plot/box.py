import io
import molpy as mp
import numpy as np
from . import global_ax

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


__all__ = ["plot_box"]

def _ax_to_bytes(ax):
    """Helper function to convert figure to png file.

    Args:
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.

    Returns:
        bytes: Byte representation of the diagram in png format.
    """
    f = io.BytesIO()
    # Sets an Agg backend so this figure can be rendered
    fig = ax.figure
    FigureCanvasAgg(fig)
    fig.savefig(f, format="png")
    fig.clf()
    return f.getvalue()


def _set_3d_axes_equal(ax, limits=None):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
        ax (:class:`matplotlib.axes.Axes`): Axes object.
        limits (:math:`(3, 2)` :class:`np.ndarray`):
            Axis limits in the form
            :code:`[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`. If
            :code:`None`, the limits are auto-detected (Default value =
            :code:`None`).
    """
    # Adapted from https://stackoverflow.com/a/50664367

    if limits is None:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    else:
        limits = np.asarray(limits)
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    return ax


def plot_box(box: np.ndarray | mp.Box, title=None, ax=None, image=[0, 0, 0], *args, **kwargs):
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

    is2D = False  # 2d is not supported yet
    ax = ax or global_ax
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
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        limits = [
            [corners[0, 0], corners[-1, 0]],
            [corners[0, 1], corners[-1, 1]],
            [corners[0, 2], corners[-1, 2]],
        ]
        _set_3d_axes_equal(ax, limits)

    return ax
