"""Onsager transport coefficients from collective mean-displacement correlations.

The Onsager phenomenological coefficients ``L_ij`` describe coupled transport of
species ``i`` and ``j``. They follow from the cross-correlation of the
*collective* (summed) displacements of each species::

    L_ij(tau) = <DP_i(tau) . DP_j(tau)>_t ,
        P_s(t)  = sum_{a in species s} r_a(t)   (unwrapped),
        DP_s(tau) = P_s(t+tau) - P_s(t).

The diagonal ``L_ii`` is the collective MSD of species ``i``; off-diagonal
``L_ij`` captures the cross-correlated drift distinguishing the Onsager picture
from the bare Nernst-Einstein sum. A long-time linear fit of ``L_ij`` yields the
transport coefficient (left to the caller).

The numerically heavy windowed (all-time-origins) cross-correlation runs in Rust
(``molrs.transport.Onsager``); this wrapper extracts coordinates, performs the
minimum-image unwrap via :meth:`molrs.Box.delta`, and builds the per-species
collective coordinates.

Adapted from the tame library (https://github.com/Roy-Kid/tame),
``tame/recipes/onsager.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molrs.transport import Onsager as _MolrsOnsager

from .base import Compute
from .result import OnsagerResult

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory


class Onsager(Compute):
    """Compute Onsager collective-displacement cross-correlations.

    Args:
        tags: Species pairs ``"i,j"`` (e.g. ``["1,1", "1,2", "2,2"]``). Each
            yields the cross-correlation of the collective displacements of
            species ``i`` and ``j``.
        max_dt: Maximum time lag in ps.
        dt: Timestep in ps.
        center_of_mass: Optional ``{type: mass}`` mapping; when given, the
            system center of mass is removed each frame before forming the
            collective coordinates.

    Examples:
        >>> from molpy.io import read_h5_trajectory
        >>> traj = read_h5_trajectory("electrolyte.h5")
        >>> ons = Onsager(tags=["1,1", "1,2", "2,2"], max_dt=20.0, dt=0.01)
        >>> result = ons(traj)
        >>> result.correlations["1,2"]  # L_12(tau), shape (n_cache,)
    """

    def __init__(
        self,
        tags: list[str],
        max_dt: float,
        dt: float,
        center_of_mass: dict[int, float] | None = None,
    ):
        super().__init__(tags=tags, max_dt=max_dt, dt=dt, center_of_mass=center_of_mass)
        self.tags = tags
        self.max_dt = max_dt
        self.dt = dt
        self.center_of_mass = center_of_mass
        self.n_cache = int(max_dt / dt)

    def __call__(self, trajectory: "Trajectory") -> OnsagerResult:
        coords_list: list[NDArray] = []
        elems_list: list[NDArray] = []
        boxes: list = []

        for frame in trajectory:
            if "atoms" not in frame:
                raise ValueError("Frame must contain 'atoms' block")
            atoms = frame["atoms"]
            for col in ("x", "y", "z", "type"):
                if col not in atoms:
                    raise ValueError(f"Atoms block must contain '{col}'")
            coords_list.append(np.column_stack([atoms["x"], atoms["y"], atoms["z"]]))
            elems_list.append(np.asarray(atoms["type"]))
            if frame.box is None:
                raise ValueError("Frame must contain box information")
            boxes.append(frame.box)

        coords_traj = np.asarray(coords_list, dtype=np.float64)  # (F, N, 3)
        elems = np.asarray(elems_list[0])
        n_frames = coords_traj.shape[0]
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")

        # Minimum-image unwrap (Rust Box.delta), identical to MCD/PMSD.
        coords_unwrapped = np.zeros_like(coords_traj)
        coords_unwrapped[0] = coords_traj[0]
        for i in range(1, n_frames):
            dr = boxes[i].delta(coords_traj[i - 1], coords_traj[i], minimum_image=True)
            coords_unwrapped[i] = coords_unwrapped[i - 1] + dr

        if self.center_of_mass is not None:
            masses = np.array(
                [self.center_of_mass.get(int(e), 1.0) for e in elems], dtype=np.float64
            )
            total = masses.sum()
            com = np.sum(coords_unwrapped * masses[None, :, None], axis=1) / total
            coords_unwrapped = coords_unwrapped - com[:, None, :]

        # Collective coordinate per species: P_s(t) = sum_{a in s} r_a(t).
        collective: dict[int, NDArray] = {}

        def get_p(t: int) -> NDArray:
            if t not in collective:
                mask = elems == t
                collective[t] = np.sum(coords_unwrapped[:, mask, :], axis=1)  # (F, 3)
            return collective[t]

        max_lag = self.n_cache - 1
        correlations: dict[str, NDArray] = {}
        for tag in self.tags:
            i_type, j_type = (int(s) for s in tag.split(","))
            res = _MolrsOnsager.correlation(
                np.ascontiguousarray(get_p(i_type)),
                np.ascontiguousarray(get_p(j_type)),
                self.dt,
                max_lag,
            )
            correlations[tag] = res["correlation"]

        time_array = np.arange(self.n_cache, dtype=np.float64) * self.dt
        return OnsagerResult(time=time_array, correlations=correlations)
