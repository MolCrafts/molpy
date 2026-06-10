"""Thole dipole-dipole screening (CL&Pol short-range damping).

Screens the Coulomb interaction between Drude-related point charges at short
range with the exponential Thole function

    T_ij(r) = 1 - (1 + s_ij r / 2) exp(-s_ij r)
    s_ij    = a_ij / (alpha_i alpha_j)^(1/6),   a_ij = (a_i + a_j) / 2

so the damped energy of a pair is ``T_ij(r) * q_i q_j / r``. Default Thole
parameter a = 2.6 (alpha.ff). Units: r in A, alpha in A^3, a dimensionless.

Reference: Thole, Chem. Phys. 59 (1981) 341, DOI 10.1016/0301-0104(81)85176-2;
as emitted by paduagroup/clandpol polarizer (LAMMPS pair_style thole).
"""

import numpy as np
from numpy.typing import NDArray

from .base import PairPotential


class Thole(PairPotential):
    """Exponential Thole screening of charge-charge interactions.

    Args:
        charge: Per-atom-type partial charge [e].
        alpha: Per-atom-type atomic polarizability [A^3].
        a_thole: Per-atom-type Thole damping parameter [dimensionless].

    Energy and force methods take separate endpoint type indices
    (``pair_types_i``/``pair_types_j``) because the screening depends on both
    atoms' polarizabilities.
    """

    name = "thole"
    type = "pair"

    def __init__(
        self,
        charge: NDArray[np.floating] | float,
        alpha: NDArray[np.floating] | float,
        a_thole: NDArray[np.floating] | float,
    ) -> None:
        self.charge = np.atleast_1d(np.asarray(charge, dtype=np.float64))
        self.alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        self.a_thole = np.atleast_1d(np.asarray(a_thole, dtype=np.float64))

    def _screen(
        self, ti: NDArray[np.integer], tj: NDArray[np.integer]
    ) -> NDArray[np.floating]:
        a_ij = 0.5 * (self.a_thole[ti] + self.a_thole[tj])
        return a_ij / (self.alpha[ti] * self.alpha[tj]) ** (1.0 / 6.0)

    def damping(
        self,
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray[np.integer],
        pair_types_j: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """Thole screening factor T_ij(r) for each pair."""
        r = dr_norm.reshape(-1) if dr_norm.ndim > 1 else dr_norm
        x = self._screen(pair_types_i, pair_types_j) * r
        return 1.0 - (1.0 + x / 2.0) * np.exp(-x)

    def calc_energy(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray[np.integer],
        pair_types_j: NDArray[np.integer],
    ) -> float:
        """Total Thole-screened Coulomb energy over the pair list."""
        if len(pair_types_i) == 0:
            return 0.0
        r = dr_norm.reshape(-1) if dr_norm.ndim > 1 else dr_norm
        t = self.damping(r, pair_types_i, pair_types_j)
        qq = self.charge[pair_types_i] * self.charge[pair_types_j]
        return float(np.sum(t * qq / r))

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
        s = self._screen(pair_types_i, pair_types_j)
        x = s * r
        e = np.exp(-x)
        t = 1.0 - (1.0 + x / 2.0) * e
        tp = (s / 2.0) * (1.0 + x) * e  # dT/dr
        qq = self.charge[pair_types_i] * self.charge[pair_types_j]
        # V = T * qq / r ; dV/dr = qq (T'/r - T/r^2)
        dvdr = qq * (tp / r - t / r**2)
        force_mag = (-dvdr / r)[:, None]  # forces = force_mag * dr (LJ convention)
        forces = force_mag * dr
        per_atom = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom, pair_idx[:, 0], -forces)
        np.add.at(per_atom, pair_idx[:, 1], forces)
        return per_atom
