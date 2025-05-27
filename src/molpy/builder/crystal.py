import numpy as np
from molpy.core import Struct
from .base import LatticeBuilder, StructBuilder, set_struct

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
        basis: np.ndarray | None = None,
        cell: np.ndarray | None = None,
    ):
        self.a = a
        self.b = b or a
        self.c = c or a
        self.alpha = alpha
        self.covera = covera
        self.u = u
        self.orthorhombic = orthorhombic
        self.cubic = cubic
        self.basis = basis
        self.cell = cell

    def create_sites(self) -> np.ndarray:
        basis = self.basis * np.array([self.a, self.b, self.c])
        return basis

    def populate(self, sites: np.ndarray, struct: Struct) -> Struct:
        result = Struct()
        for pos in sites:
            s = struct.copy()
            set_struct(s, pos)
            result = Struct.merge([result, s])
        return result

class FCC(CrystalBuilder):
    def __init__(
        self,
        a: float,
        *,
        orthorhombic: bool = False,
        cubic: bool = False,
    ):
        b = 0.5 * a
        cell = np.array([[0, b, b], [b, 0, b], [b, b, 0]]) * 2
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ])
        super().__init__(
            a=a,
            b=b,
            c=a,
            orthorhombic=orthorhombic,
            cubic=cubic,
            basis=basis,
            cell=cell,
        )

    def create_sites(self) -> np.ndarray:
        return super().create_sites()
