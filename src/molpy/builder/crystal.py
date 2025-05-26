import numpy as np

from molpy.core import Region, Struct

from .base import Lattice, LatticeBuilder, StructBuilder

class CrystalBuilder(LatticeBuilder, StructBuilder):

    def __init__(
        self,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        *,
        alpha: float | None = None,
        covera: float | None = None,
        u: float | None = None,
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

    def create_sites(self) -> Lattice:
        ...

    def build_struct(self) -> Struct:
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
        self.cell = np.array([[0, b, b], [b, 0, b], [b, b, 0]])
        self.basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])

    def create_lattice(self):
        asites = Lattice(
            self.basis * self.a,
            self.cell
        )
        return asites
    
    def build_struct(self, struct: Struct) -> Struct:
        lattice = self.create_lattice()
        return lattice.fill(struct)