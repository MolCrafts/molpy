from abc import abstractmethod
import numpy as np
from ..target import Target 

class Packer:

    def __init__(self):
        self.targets = []

    def add_target(self, target: Target):
        self.targets.append(target)

    def def_target(self, frame, number, constraint, is_fixe=False, optimizer=None, name=""):
        """
        Define a target for packing.
        """
        target = Target(frame, number, constraint, is_fixe, optimizer, name)
        self.add_target(target)
        return target

    @abstractmethod
    def pack(self):
        ...

    @property
    def n_points(self):
        return sum([t.n_points for t in self.targets])
    
    @property
    def points(self):
        return np.concatenate([t.points for t in self.targets], axis=0)
