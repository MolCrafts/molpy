import numpy as np
from molpy.core import Struct, Atom
from .base import LatticeBuilder, StructBuilder, set_struct

class Lattice:
    def __init__(self, sites: np.ndarray, cell: np.ndarray):
        self.sites = sites
        self.cell = np.asarray(cell, float)

    def repeat(self, nx: int = 1, ny: int = 1, nz: int = 1):
        shape = (nx, ny, nz)
        basis = self.sites
        cell = self.cell
        reps = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = i*cell[0] + j*cell[1] + k*cell[2]
                    reps.append(basis + offset)
        sites = np.vstack(reps)
        cell = cell * np.array(shape)
        return Lattice(sites, cell)

    def fill(self, struct: Struct) -> Struct:
        result = Struct()
        for pos in self.sites:
            s = struct.copy()
            set_struct(s, pos)
            result = Struct.merge([result, s])
        return result

class CrystalBuilder(LatticeBuilder, StructBuilder):
    def __init__(
        self,
        a: float,
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
        repeat: tuple[int,int,int] = (1,1,1),
    ):
        self.a = a
        self.b = b or a
        self.c = c or a
        self.alpha = alpha
        self.covera = covera
        self.u = u
        self.orthorhombic = orthorhombic
        self.cubic = cubic
        self.basis = np.asarray(basis, float)
        self.cell = np.asarray(cell, float)
        self.repeat_dims = repeat

    def create_sites(self) -> np.ndarray:
        sites = self.basis * np.array([self.a, self.b, self.c])
        lattice = Lattice(sites, self.cell)
        lattice = lattice.repeat(*self.repeat_dims)
        return lattice.sites

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
        repeat: tuple[int,int,int] = (1,1,1),
        orthorhombic: bool = False,
        cubic: bool = False,
    ):
        b = 0.5 * a
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
        cell = np.array([[0.0, b, b], [b, 0.0, b], [b, b, 0.0]])
        super().__init__(
            a=a,
            b=b,
            c=a,
            orthorhombic=orthorhombic,
            cubic=cubic,
            basis=basis,
            cell=cell,
            repeat=repeat,
        )
