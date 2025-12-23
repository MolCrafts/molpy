"""Polarization Mean Square Displacement (PMSD) computation.

This module implements the PMSD method for computing polarization fluctuations
in ionic systems from molecular dynamics trajectories.

Adapted from the tame library (https://github.com/Roy-Kid/tame).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Compute
from .result import PMSDResult
from .time_series import TimeAverage, TimeCache

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory


class PMSDCompute(Compute["Trajectory", PMSDResult]):
    """Compute Polarization Mean Square Displacement for ionic systems.

    This class computes the PMSD which measures polarization fluctuations in
    ionic systems. The polarization is defined as:

        P(t) = Σ_cations r_i(t) - Σ_anions r_j(t)

    And the PMSD is:

        PMSD(dt) = <(P(t+dt) - P(t))²>_t

    Args:
        cation_type: Atom type index for cations
        anion_type: Atom type index for anions
        max_dt: Maximum time lag in ps
        dt: Timestep in ps

    Examples:
        >>> from molpy.io import read_h5_trajectory
        >>> traj = read_h5_trajectory("ionic_liquid.h5")
        >>>
        >>> # Compute PMSD for Li+ (type 1) and PF6- (type 2)
        >>> pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01)
        >>> result = pmsd(traj)
        >>>
        >>> # Plot results
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result.time, result.pmsd)
        >>> plt.xlabel("Time lag (ps)")
        >>> plt.ylabel("PMSD (Å²)")
        >>> plt.show()
    """

    def __init__(
        self,
        cation_type: int,
        anion_type: int,
        max_dt: float,
        dt: float,
    ):
        self.cation_type = cation_type
        self.anion_type = anion_type
        self.max_dt = max_dt
        self.dt = dt
        self.n_cache = int(max_dt / dt)

    def compute(self, trajectory: "Trajectory") -> PMSDResult:
        """Compute PMSD from trajectory.

        Args:
            trajectory: Input trajectory object

        Returns:
            PMSDResult containing time points and PMSD values
        """
        # Extract data from trajectory
        coords_list = []
        elems_list = []
        box_list = []

        for frame in trajectory:
            if "atoms" not in frame:
                raise ValueError("Frame must contain 'atoms' block")

            atoms = frame["atoms"]
            if "x" not in atoms or "y" not in atoms or "z" not in atoms:
                raise ValueError("Atoms block must contain x, y, z coordinates")
            if "type" not in atoms:
                raise ValueError("Atoms block must contain 'type' field")

            coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
            coords_list.append(coords)
            elems_list.append(atoms["type"])

            # Get box from metadata
            if "box" in frame.metadata:
                box = frame.metadata["box"]
                box_list.append(box)
            else:
                raise ValueError("Frame must contain box information in metadata")

        coords_traj = np.array(coords_list)  # (n_frames, n_atoms, 3)
        elems_traj = np.array(elems_list)  # (n_frames, n_atoms)

        # Get masks for cations and anions (use first frame types)
        cation_mask = elems_traj[0] == self.cation_type
        anion_mask = elems_traj[0] == self.anion_type

        # Extract coordinates for cations and anions
        coords_cations = coords_traj[:, cation_mask, :]
        coords_anions = coords_traj[:, anion_mask, :]

        # Unwrap coordinates using Box.diff_dr() directly
        n_frames = coords_traj.shape[0]

        # Unwrap cations
        coords_cations_unwrapped = np.zeros_like(coords_cations)
        coords_cations_unwrapped[0] = coords_cations[0].copy()
        for i in range(1, n_frames):
            displacement = coords_cations[i] - coords_cations[i - 1]
            displacement_mic = box_list[i].diff_dr(displacement)
            coords_cations_unwrapped[i] = (
                coords_cations_unwrapped[i - 1] + displacement_mic
            )

        # Unwrap anions
        coords_anions_unwrapped = np.zeros_like(coords_anions)
        coords_anions_unwrapped[0] = coords_anions[0].copy()
        for i in range(1, n_frames):
            displacement = coords_anions[i] - coords_anions[i - 1]
            displacement_mic = box_list[i].diff_dr(displacement)
            coords_anions_unwrapped[i] = (
                coords_anions_unwrapped[i - 1] + displacement_mic
            )

        # Compute total polarization at each frame
        # P(t) = Σ r_cation - Σ r_anion
        polarization = np.sum(coords_cations_unwrapped, axis=1) - np.sum(
            coords_anions_unwrapped, axis=1
        )  # (n_frames, 3)

        # Compute PMSD using time cache
        pmsd = self._compute_pmsd(polarization)

        time_array = np.arange(self.n_cache) * self.dt

        return PMSDResult(
            time=time_array,
            pmsd=pmsd,
        )

    def _compute_pmsd(self, polarization: NDArray) -> NDArray:
        """Compute PMSD from polarization time series.

        Args:
            polarization: Polarization vectors (n_frames, 3)

        Returns:
            PMSD values at each time lag (n_cache,)
        """
        n_frames, n_dim = polarization.shape

        # Initialize cache and accumulator
        cache = TimeCache(self.n_cache, shape=(n_dim,))
        avg = TimeAverage(shape=(self.n_cache,), dropnan="partial")

        # Iterate through trajectory
        for frame_idx in range(n_frames):
            current_P = polarization[frame_idx]
            cache.update(current_P)

            # Compute displacement from current to all cached frames
            cached_P = cache.get()  # (n_cache, 3)
            dP = cached_P - current_P[None, :]  # (n_cache, 3)

            # Compute squared displacement
            pmsd_frame = np.sum(dP**2, axis=1)  # (n_cache,)

            # Accumulate time average
            avg.update(pmsd_frame)

        return avg.get()
