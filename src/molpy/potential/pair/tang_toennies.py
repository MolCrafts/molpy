"""Tang-Toennies charge / induced-dipole damping (CL&Pol short-range damping).

Damps the Coulomb interaction between a charge and an induced dipole (a Drude
shell) at short range for small, highly-charged atoms, preventing the
"polarization catastrophe":

    f_n(r) = 1 - c exp(-b r) sum_{k=0}^{n} (b r)^k / k!

so the damped energy of a pair is ``f_n(r) * q_i q_j / r``. CL&Pol canonical
settings: n = 4, b = 4.5 (1/A), c = 1.0. Units: r in A.

Reference: Tang & Toennies, J. Chem. Phys. 80 (1984) 3726, DOI 10.1063/1.447150;
as emitted by paduagroup/clandpol coul_tt (LAMMPS pair_coul_tt).
"""

from math import factorial

import numpy as np
from numpy.typing import NDArray

from .base import PairPotential


class TangToennies(PairPotential):
    """Tang-Toennies damping of charge-charge interactions.

    Args:
        charge: Per-atom-type partial charge [e].
        b: Tang-Toennies range parameter [1/A] (CL&Pol default 4.5).
        n: Tang-Toennies order (CL&Pol default 4).
        c: Prefactor (CL&Pol default 1.0).
    """

    name = "coul/tt"
    type = "pair"

    def __init__(
        self,
        charge: NDArray[np.floating] | float,
        b: float = 4.5,
        n: int = 4,
        c: float = 1.0,
    ) -> None:
        self.charge = np.atleast_1d(np.asarray(charge, dtype=np.float64))
        self.b = float(b)
        self.n = int(n)
        self.c = float(c)

    def damping(self, dr_norm: NDArray[np.floating]) -> NDArray[np.floating]:
        """Tang-Toennies damping factor f_n(r) for each pair distance."""
        r = dr_norm.reshape(-1) if dr_norm.ndim > 1 else dr_norm
        br = self.b * r
        series = sum((br**k) / factorial(k) for k in range(self.n + 1))
        return 1.0 - self.c * np.exp(-br) * series

    def _dfdr(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        # d/dr of f_n collapses to a single term: c b e^{-br} (br)^n / n!
        br = self.b * r
        return self.c * self.b * np.exp(-br) * br**self.n / factorial(self.n)

    def calc_energy(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray[np.integer],
        pair_types_j: NDArray[np.integer],
    ) -> float:
        """Total Tang-Toennies-damped Coulomb energy over the pair list."""
        if len(pair_types_i) == 0:
            return 0.0
        r = dr_norm.reshape(-1) if dr_norm.ndim > 1 else dr_norm
        f = self.damping(r)
        qq = self.charge[pair_types_i] * self.charge[pair_types_j]
        return float(np.sum(f * qq / r))

    def calc_forces(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray[np.integer],
        pair_types_j: NDArray[np.integer],
        pair_idx: NDArray[np.integer],
        n_atoms: int,
    ) -> NDArray[np.floating]:
        """Per-atom forces, the negative gradient of :meth:`calc_energy`."""
        if len(pair_types_i) == 0:
            return np.zeros((n_atoms, 3), dtype=np.float64)
        r = dr_norm.reshape(-1) if dr_norm.ndim > 1 else dr_norm
        f = self.damping(r)
        fp = self._dfdr(r)
        qq = self.charge[pair_types_i] * self.charge[pair_types_j]
        # V = f * qq / r ; dV/dr = qq (f'/r - f/r^2)
        dvdr = qq * (fp / r - f / r**2)
        force_mag = (-dvdr / r)[:, None]  # forces = force_mag * dr (LJ convention)
        forces = force_mag * dr
        per_atom = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom, pair_idx[:, 0], -forces)
        np.add.at(per_atom, pair_idx[:, 1], forces)
        return per_atom
