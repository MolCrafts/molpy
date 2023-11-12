import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

__all__ = ["global_ax"]

fig = plt.figure()
global_ax = fig.add_subplot(111, projection="3d")