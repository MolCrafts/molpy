import numpy as np
from abc import ABC, abstractmethod
from molpy.core import Struct


class BaseBuilder(ABC):
    ...

class StructBuilder(BaseBuilder):
    """
    Base class for building structures.
    """

    @abstractmethod
    def build_struct(self) -> Struct:
        """
        Build the structure.
        Returns:
            Struct: The built structure.
        """
        pass


class Lattice:

    def __init__(self, sites: np.ndarray, cell: np.ndarray):
        self.sites = sites
        self.cell = cell

    def repeat(self, *shape):
        """
        repeat sites in different directions
        """
        assert len(shape) == 3

        for x, vec in zip(shape, self.cell):
            if x != 1 and not vec.any():
                raise ValueError('Cannot repeat along undefined lattice '
                                'vector')

        M = np.prod(shape)
        n = len(self)

        positions = np.empty((n * M, 3))
        i0 = 0
        for m0 in range(shape[0]):
            for m1 in range(shape[1]):
                for m2 in range(shape[2]):
                    i1 = i0 + n
                    positions[i0:i1] += np.dot((m0, m1, m2), self.cell)
                    i0 = i1

        self.cell = np.array([shape[c] * self.cell[c] for c in range(3)])

    def __len__(self):
        return len(self.sites)

    def fill(self, struct: Struct) -> Struct: ...

class LatticeBuilder(StructBuilder):
    """
    Base class for building structures with a lattice.
    """

    @abstractmethod
    def create_sites(self) -> Lattice:
        """
        Create the sites of the structure.
        Returns:
            np.ndarray: The coordinates of the sites.
        """
        pass