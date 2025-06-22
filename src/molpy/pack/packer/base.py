from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
from ..target import Target 

if TYPE_CHECKING:
    import molpy as mp 

class Packer:

    def __init__(self):
        self.targets = []

    def add_target(self, target: Target):
        self.targets.append(target)

    def def_target(self, frame, number, constraint, is_fixed=False, optimizer=None, name=""):
        """
        Define a target for packing.
        """
        target = Target(frame, number, constraint, is_fixed, optimizer, name)
        self.add_target(target)
        return target

    @abstractmethod
    def pack(self, targets=None, max_steps: int = 1000, seed: int | None = None) -> 'mp.Frame':
        ...

    @property
    def n_points(self):
        return sum([t.n_points for t in self.targets])
    
    @property
    def points(self):
        if not self.targets:
            return np.empty((0, 3))
        
        target_points = [t.points for t in self.targets]
        # Filter out empty arrays
        non_empty_points = [p for p in target_points if p.size > 0]
        
        if not non_empty_points:
            return np.empty((0, 3))
        
        return np.concatenate(non_empty_points, axis=0)
