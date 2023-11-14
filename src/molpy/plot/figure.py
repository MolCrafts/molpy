import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

__all__ = ["Figure"]

class Figure:

    def __init__(self):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection="3d")

    @property
    def fig(self):
        return self._fig
    
    @property
    def ax(self):
        return self._ax