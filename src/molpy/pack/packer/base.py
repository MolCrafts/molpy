from abc import abstractmethod
import numpy as np

class Packer:

    def __init__(self):
        self.targets = []

    def add_target(self, target):
        self.targets.append(target)

    @abstractmethod
    def pack(self):
        ...

    @property
    def n_points(self):
        return sum([t.n_points for t in self.targets])
    
    @property
    def points(self):
        return np.concatenate([t.points for t in self.targets], axis=0)
