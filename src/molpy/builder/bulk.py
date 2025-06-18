import numpy as np
from itertools import product
from typing import Sequence, Iterable

from molpy.core.struct import AtomicStructure
from molpy.core.region import Region

class CrystalLattice:
    """Lattice represented by a cell matrix and fractional basis."""

    def __init__(self, cell: Sequence[Sequence[float]], basis: Iterable[Sequence[float]]):
        self.cell = np.asarray(cell, dtype=float)
        if self.cell.shape != (3, 3):
            raise ValueError("cell must be a 3x3 matrix")
        basis = np.asarray(list(basis), dtype=float)
        if basis.ndim == 1:
            basis = basis.reshape(1, 3)
        if basis.shape[1] != 3:
            raise ValueError("basis must be of shape (n, 3)")
        self.basis = basis

    def _get_bounds(self, region: Region):
        if hasattr(region, "bounding_box"):
            lo, hi = region.bounding_box()
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
        elif hasattr(region, "bounds"):
            bounds = np.asarray(region.bounds, dtype=float)
            if bounds.shape == (2, 3):
                lo, hi = bounds[0], bounds[1]
            elif bounds.shape == (3, 2):
                lo, hi = bounds[:, 0], bounds[:, 1]
            else:
                raise ValueError("Unknown bounds shape")
        else:
            raise ValueError("region must provide bounding_box() or bounds")
        return lo, hi

    def generate_positions(self, region: Region) -> np.ndarray:
        """Return all lattice positions within *region*."""
        lo, hi = self._get_bounds(region)

        corners = np.array(list(product(*zip(lo, hi))))
        inv_cell = np.linalg.inv(self.cell)
        frac_corners = corners @ inv_cell
        fmin = frac_corners.min(axis=0)
        fmax = frac_corners.max(axis=0)

        n_min = []
        n_max = []
        eps = 1e-8
        for i in range(3):
            lower = np.min(np.ceil(fmin[i] - self.basis[:, i] - eps))
            upper = np.max(np.floor(fmax[i] - self.basis[:, i] - eps))
            n_min.append(int(lower))
            n_max.append(int(upper))
        n_min = np.array(n_min, dtype=int)
        n_max = np.array(n_max, dtype=int)

        positions = []
        for nx in range(n_min[0], n_max[0] + 1):
            for ny in range(n_min[1], n_max[1] + 1):
                for nz in range(n_min[2], n_max[2] + 1):
                    trans = np.array([nx, ny, nz], dtype=float)
                    for b in self.basis:
                        frac = b + trans
                        xyz = frac @ self.cell
                        if region.isin(np.array([xyz]))[0]:
                            positions.append(xyz)
        return np.array(positions)

class CrystalBuilder:
    """Replicate a template structure on a crystal lattice."""

    def __init__(self, lattice: CrystalLattice):
        self.lattice = lattice

    def build(self, region: Region, template: AtomicStructure) -> AtomicStructure:
        positions = self.lattice.generate_positions(region)
        replicas = []
        for pos in positions:
            inst = template.clone()
            inst.xyz = inst.xyz + pos
            replicas.append(inst)
        return AtomicStructure.concat("crystal", replicas)

class FCCBuilder(CrystalBuilder):
    """Face-centered cubic lattice builder."""

    def __init__(self, a: float):
        cell = np.diag([a, a, a])
        basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
        super().__init__(CrystalLattice(cell, basis))

class BCCBuilder(CrystalBuilder):
    """Body-centered cubic lattice builder."""

    def __init__(self, a: float):
        cell = np.diag([a, a, a])
        basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
        super().__init__(CrystalLattice(cell, basis))

class HCPBuilder(CrystalBuilder):
    """Hexagonal close-packed lattice builder."""

    def __init__(self, a: float, c: float):
        cell = np.array([
            [a, 0.0, 0.0],
            [0.5 * a, np.sqrt(3) / 2 * a, 0.0],
            [0.0, 0.0, c],
        ])
        basis = [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
        ]
        super().__init__(CrystalLattice(cell, basis))

def bulk(symbol: str, crystalstructure: str, a: float | None = None, c: float | None = None, region: Region | None = None):
    cs = crystalstructure.lower()
    if cs == "fcc":
        if a is None:
            raise ValueError("Parameter 'a' must be provided for fcc")
        builder = FCCBuilder(a)
    elif cs == "bcc":
        if a is None:
            raise ValueError("Parameter 'a' must be provided for bcc")
        builder = BCCBuilder(a)
    elif cs == "hcp":
        if a is None or c is None:
            raise ValueError("Parameters 'a' and 'c' must be provided for hcp")
        builder = HCPBuilder(a, c)
    else:
        raise ValueError(f"Unknown crystal structure: {crystalstructure}")

    if region is None:
        return builder

    template = AtomicStructure(symbol)
    template.def_atom(name=symbol, element=symbol, xyz=[0.0, 0.0, 0.0])
    return builder.build(region, template)
