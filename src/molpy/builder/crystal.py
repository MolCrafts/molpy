import numpy as np

from molpy.core import Region, Struct
from .base import LatticeBuilder, StructBuilder
from .base import BaseBuilder


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

    def fill(self, struct: Struct) -> Struct: ...


class CrystalBuilder(BaseBuilder):

    def __init__(
        self,
        a: float = None,
        b: float = None,
        c: float = None,
        *,
        alpha: float = None,
        covera: float = None,
        u: float = None,
        orthorhombic: bool = False,
        cubic: bool = False,
        basis=None,
    ):

        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.covera = covera
        self.u = u
        self.orthorhombic = orthorhombic
        self.cubic = cubic
        self.basis = basis

    def create_sites(self, region):
        ...


class FCC(CrystalBuilder):

    def __init__(
        self,
        a: float,
        *,
        orthorhombic: bool = False,
        cubic: bool = False,
    ):
        b = 0.5*a
        super().__init__(a=a, b=b, orthorhombic=orthorhombic, cubic=cubic)
        self.cell = ((0, b, b), (b, 0, b), (b, b, 0))
        self.basis = ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5))

    def create_lattice(self):
        asites = Lattice(
            self.basis * self.a,
            self.cell
        )
        return asites

class CrystalLatticeBuilder(LatticeBuilder):
    def create_sites(self,
                     element: str = 'X',   # tylko metadana, nie używana przy samej siatce
                     nx: int = 1,
                     ny: int = 1,
                     nz: int = 1,
                     a: float = 1.0):
        coords = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    coords.append((i * a, j * a, k * a))
        return coords


class CrystalStructBuilder(StructBuilder):
    def populate(self, sites, element: str = 'Cu', **params):
        atoms = []
        for pos in sites:
            atoms.append({
                'type': element,
                'position': pos
            })
        return atoms