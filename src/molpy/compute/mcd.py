"""Mean Displacement Correlation (MCD) computation for diffusion analysis.

This module implements the MCD method for computing self and distinct diffusion
coefficients from molecular dynamics trajectories.

Adapted from the tame library (https://github.com/Roy-Kid/tame).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Compute
from .result import MCDResult
from .time_series import TimeAverage, TimeCache

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory


def linear_fit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_min: float,
    x_max: float,
) -> tuple[float, float]:
    """Perform linear fit in specified range.

    Args:
        x: X values
        y: Y values
        x_min: Minimum x value for fitting range
        x_max: Maximum x value for fitting range

    Returns:
        Tuple of (slope, intercept)
    """
    mask = (x >= x_min) & (x <= x_max)
    if not mask.any():
        raise ValueError(f"No data points in range [{x_min}, {x_max}]")

    x_fit = x[mask]
    y_fit = y[mask]

    # Linear regression: y = slope * x + intercept
    coeffs = np.polyfit(x_fit, y_fit, 1)
    slope, intercept = coeffs[0], coeffs[1]

    return slope, intercept


class MCDCompute(Compute["Trajectory", MCDResult]):
    """Compute Mean Displacement Correlations for diffusion analysis.

    This class implements the MCD method which computes time correlation functions
    of particle displacements to extract diffusion coefficients. It supports:

    - Self diffusion: D_i = <(r_i(t+dt) - r_i(t))²> / 6dt
    - Distinct diffusion: D_ij = <(r_i(t+dt) - r_i(t)) · (r_j(t+dt) - r_j(t))> / 6dt

    Args:
        tags: List of atom type specifications. Each tag can be:
            - Single integer (e.g., "3"): Self-diffusion of type 3
            - Two integers separated by comma (e.g., "3,4"): Distinct diffusion between types 3 and 4
        max_dt: Maximum time lag in ps
        dt: Timestep in ps
        fit_range: Tuple of (min_time, max_time) in ps for linear fitting
        center_of_mass: Optional dict mapping element types to masses for COM removal.
            Format: {element_type: mass}, e.g., {1: 1.008, 6: 12.011}

    Examples:
        >>> from molpy.io import read_h5_trajectory
        >>> traj = read_h5_trajectory("trajectory.h5")
        >>>
        >>> # Compute self-diffusion of atom type 3
        >>> mcd = MCDCompute(tags=["3"], max_dt=30.0, dt=0.01)
        >>> result = mcd(traj)
        >>> print(f"D = {result.diffusion_coefficients['3']:.4f} Å²/ps")
        >>>
        >>> # Compute distinct diffusion between types 3 and 4
        >>> mcd = MCDCompute(tags=["3,4"], max_dt=30.0, dt=0.01)
        >>> result = mcd(traj)
    """

    def __init__(
        self,
        tags: list[str],
        max_dt: float,
        dt: float,
        fit_range: tuple[float, float] = (5.0, 20.0),
        center_of_mass: dict[int, float] | None = None,
    ):
        self.tags = tags
        self.max_dt = max_dt
        self.dt = dt
        self.fit_range = fit_range
        self.center_of_mass = center_of_mass
        self.n_cache = int(max_dt / dt)

    def compute(self, trajectory: "Trajectory") -> MCDResult:
        """Compute MCD from trajectory.

        Args:
            trajectory: Input trajectory object

        Returns:
            MCDResult containing correlation functions and diffusion coefficients
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

        # Unwrap coordinates using Box.diff_dr() directly
        n_frames, n_atoms, n_dim = coords_traj.shape
        coords_unwrapped = np.zeros_like(coords_traj)
        coords_unwrapped[0] = coords_traj[0].copy()

        for i in range(1, n_frames):
            box = box_list[i]
            # Compute displacement and apply minimum image convention
            displacement = coords_traj[i] - coords_traj[i - 1]
            displacement_mic = box.diff_dr(displacement)
            coords_unwrapped[i] = coords_unwrapped[i - 1] + displacement_mic

        # Apply center of mass correction if requested
        if self.center_of_mass is not None:
            coords_unwrapped = self._remove_com(coords_unwrapped, elems_traj[0])

        # Compute correlations for each tag
        correlations = self._compute_correlations(coords_unwrapped, elems_traj[0])

        # Fit diffusion coefficients
        time_array = np.arange(self.n_cache) * self.dt
        diffusion_coeffs = {}

        for tag, corr in correlations.items():
            if "cnt" not in tag:  # Skip count arrays
                try:
                    slope, _ = linear_fit(
                        time_array, corr, self.fit_range[0], self.fit_range[1]
                    )
                    # D = slope / 6 (for 3D diffusion)
                    diffusion_coeffs[tag] = slope / 6.0
                except ValueError:
                    # Fitting failed, set to NaN
                    diffusion_coeffs[tag] = np.nan

        return MCDResult(
            time=time_array,
            correlations=correlations,
            diffusion_coefficients=diffusion_coeffs,
        )

    def _remove_com(self, coords: NDArray, elems: NDArray) -> NDArray:
        """Remove center of mass motion.

        Args:
            coords: Unwrapped coordinates (n_frames, n_atoms, 3)
            elems: Element types (n_atoms,)

        Returns:
            Coordinates with COM removed
        """
        n_frames, n_atoms, _ = coords.shape
        masses = np.array([self.center_of_mass.get(int(e), 1.0) for e in elems])
        total_mass = masses.sum()

        coords_centered = np.zeros_like(coords)
        for i in range(n_frames):
            com = np.sum(coords[i] * masses[:, None], axis=0) / total_mass
            coords_centered[i] = coords[i] - com[None, :]

        return coords_centered

    def _compute_correlations(
        self, coords: NDArray, elems: NDArray
    ) -> dict[str, NDArray]:
        """Compute displacement correlations for all tags.

        Args:
            coords: Unwrapped coordinates (n_frames, n_atoms, 3)
            elems: Element types (n_atoms,)

        Returns:
            Dictionary mapping tag names to correlation arrays
        """
        n_frames, n_atoms, n_dim = coords.shape
        correlations = {}

        # Cache for displacement arrays
        dr_cache: dict[int, NDArray] = {}

        def get_dr(atom_type: int) -> NDArray:
            """Get displacement array for given atom type."""
            if atom_type not in dr_cache:
                # Select atoms of this type
                mask = elems == atom_type
                coords_type = coords[:, mask, :]  # (n_frames, n_type, 3)

                # Create cache for this type
                n_type = coords_type.shape[1]
                cache = TimeCache(self.n_cache, shape=(n_type, n_dim))
                avg = TimeAverage(shape=(self.n_cache, n_type, n_dim))

                # Compute displacements
                for frame_idx in range(n_frames):
                    current = coords_type[frame_idx]
                    cache.update(current)
                    cached = cache.get()  # (n_cache, n_type, 3)
                    dr = cached - current[None, :, :]  # Displacement from current
                    avg.update(dr)

                dr_cache[atom_type] = avg.get()  # (n_cache, n_type, 3)

            return dr_cache[atom_type]

        # Process each tag
        for tag in self.tags:
            if "," not in tag:
                # Self-diffusion: <dr_i²>
                atom_type = int(tag)
                dr = get_dr(atom_type)  # (n_cache, n_type, 3)

                # Compute MSD and average over particles
                msd = np.sum(dr**2, axis=2)  # (n_cache, n_type)
                corr = np.mean(msd, axis=1)  # (n_cache,)
                correlations[tag] = corr

            else:
                # Distinct diffusion: <dr_i · dr_j>
                type1, type2 = map(int, tag.split(","))
                dr_i = get_dr(type1)  # (n_cache, n_i, 3)
                dr_j = get_dr(type2)  # (n_cache, n_j, 3)

                if type1 == type2:
                    # Same species: exclude self-correlation
                    # <dr_i · (Σ_j dr_j - dr_i)>
                    dr_j_sum = np.sum(dr_j, axis=1, keepdims=True)  # (n_cache, 1, 3)
                    dr_j_others = dr_j_sum - dr_i  # (n_cache, n_i, 3)
                    corr = np.mean(np.sum(dr_i * dr_j_others, axis=2), axis=1)
                else:
                    # Different species: <Σ_i dr_i · Σ_j dr_j>
                    dr_i_sum = np.sum(dr_i, axis=1)  # (n_cache, 3)
                    dr_j_sum = np.sum(dr_j, axis=1)  # (n_cache, 3)
                    corr = np.sum(dr_i_sum * dr_j_sum, axis=1)  # (n_cache,)

                correlations[tag] = corr

        return correlations
