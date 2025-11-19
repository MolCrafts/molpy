# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""Example compute implementations for developers.

This module demonstrates how to implement custom compute operations using the
MolPy compute abstraction. Use these as templates when creating your own
compute operations.

Developer Guidelines
--------------------

When implementing a new compute operation:

1. **Define a Result class**:
   - Inherit from `Result`
   - Use `@dataclass` decorator
   - Add fields specific to your computation
   - Keep it simple and immutable

2. **Implement a Compute class**:
   - Inherit from `Compute[InT, OutT]` with appropriate types
   - Override the `compute(self, input: InT) -> OutT` method
   - Optionally override `before()` for input validation
   - Optionally override `after()` for post-processing

3. **Follow naming conventions**:
   - Result classes: `<Operation>Result`
   - Compute classes: `<Operation>Compute`

4. **Use type annotations**:
   - Always provide full type hints
   - Use modern Python 3.10+ syntax (X | None)
   - Helps with IDE autocomplete and type checking

5. **Handle edge cases**:
   - Check if required data exists
   - Provide sensible defaults
   - Don't raise exceptions unless truly exceptional

Examples Below
--------------

This module provides three example implementations demonstrating different
complexity levels:

1. **CountAtomsCompute**: Simple compute with minimal logic
2. **CenterOfMassCompute**: Moderate compute with fallback handling
3. **MeanSquaredDisplacementCompute**: Complex compute with trajectory processing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from molpy.core.frame import Frame
from molpy.core.trajectory import Trajectory

from .base import Compute
from .result import Result

# =============================================================================
# Example 1: Simple Compute
# =============================================================================


@dataclass
class CountAtomsResult(Result):
    """Result from counting atoms in a frame.

    This is the simplest possible result: just a single integer value.

    Attributes:
        value: Number of atoms in the frame.
    """

    value: int = 0


class CountAtomsCompute(Compute[Frame, CountAtomsResult]):
    """Compute the number of atoms in a frame.

    This is a minimal example showing the basic structure of a compute
    operation. It demonstrates:

    - Checking if required data exists
    - Accessing data from a Frame
    - Returning a typed result

    Examples:
        >>> compute = CountAtomsCompute()
        >>> result = compute(frame)
        >>> print(f"Frame contains {result.value} atoms")

    Developer Notes:
        - No constructor parameters needed for simple operations
        - Always check if the required block exists before accessing
        - Use descriptive result names like "count_atoms"
    """

    def compute(self, input: Frame) -> CountAtomsResult:
        """Count the atoms in the frame.

        Args:
            input: Input frame containing atomic data.

        Returns:
            Result containing the atom count.
        """
        # Check if the atoms block exists
        if "atoms" not in input:
            return CountAtomsResult(name="count_atoms", value=0)

        # Get the atoms table from the frame
        atoms_table = input["atoms"]

        # Count the number of rows in the atoms table
        n_atoms = atoms_table.nrows

        return CountAtomsResult(name="count_atoms", value=n_atoms)


# =============================================================================
# Example 2: Moderate Complexity with Fallbacks
# =============================================================================


@dataclass
class CenterOfMassResult(Result):
    """Result from computing center of mass.

    Demonstrates a result with a numpy array field and metadata.

    Attributes:
        com: Center of mass coordinates as a 3D vector [x, y, z].

    Developer Notes:
        - Use None as default for array fields
        - Initialize in __post_init__ if needed
        - Store additional information in metadata
    """

    com: NDArray[np.floating] = None

    def __post_init__(self) -> None:
        """Initialize with default zero vector if needed."""
        if self.com is None:
            self.com = np.array([0.0, 0.0, 0.0])


class CenterOfMassCompute(Compute[Frame, CenterOfMassResult]):
    """Compute the center of mass of a frame.

    This compute demonstrates:

    - Handling missing data columns gracefully
    - Working with numpy arrays
    - Using metadata for additional information
    - Providing sensible defaults

    Examples:
        >>> compute = CenterOfMassCompute()
        >>> result = compute(frame)
        >>> print(f"Center of mass: {result.com}")
        >>> print(f"Total mass: {result.meta['total_mass']} amu")

    Developer Notes:
        - Use fallback values when optional data is missing
        - Store derived quantities in metadata
        - Avoid division by zero errors
    """

    def compute(self, input: Frame) -> CenterOfMassResult:
        """Calculate the center of mass.

        Args:
            input: Input frame containing atomic data with coordinates and masses.

        Returns:
            Result containing the center of mass coordinates.
        """
        # Check if the atoms block exists
        if "atoms" not in input:
            return CenterOfMassResult(
                name="center_of_mass", com=np.array([0.0, 0.0, 0.0])
            )

        atoms_table = input["atoms"]

        if atoms_table.nrows == 0:
            return CenterOfMassResult(
                name="center_of_mass", com=np.array([0.0, 0.0, 0.0])
            )

        # Extract coordinates (x, y, z columns)
        x = atoms_table["x"][:]
        y = atoms_table["y"][:]
        z = atoms_table["z"][:]
        coords = np.column_stack([x, y, z])

        # Extract masses (with fallback to uniform mass if not available)
        if "mass" in atoms_table:
            masses = atoms_table["mass"][:]
        else:
            # If mass column doesn't exist, assume uniform mass
            masses = np.ones(len(x))

        # Calculate mass-weighted center of mass
        total_mass = np.sum(masses)

        if total_mass == 0:
            com = np.array([0.0, 0.0, 0.0])
        else:
            com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

        return CenterOfMassResult(
            name="center_of_mass", com=com, meta={"total_mass": float(total_mass)}
        )


# =============================================================================
# Example 3: Complex Trajectory-Based Compute
# =============================================================================


@dataclass
class MeanSquaredDisplacementResult(Result):
    """Result from mean squared displacement calculation.

    The MSD measures how much particles move over time, which is crucial for
    characterizing diffusion and Brownian motion.

    Attributes:
        msd: Mean squared displacement values for each time lag.
            Shape: (n_frames,)
        particle_msd: Per-particle MSD values for each time lag.
            Shape: (n_frames, n_particles)
        mode: Calculation mode used ('window' or 'direct').

    Developer Notes:
        - For trajectory-based computes, results are typically arrays
        - Store computation parameters in metadata
        - Provide both aggregate and per-particle results when useful
    """

    msd: NDArray[np.floating] = None
    particle_msd: NDArray[np.floating] = None
    mode: str = "window"

    def __post_init__(self) -> None:
        """Initialize with empty arrays if needed."""
        if self.msd is None:
            self.msd = np.array([])
        if self.particle_msd is None:
            self.particle_msd = np.array([])


class MeanSquaredDisplacementCompute(
    Compute[Trajectory, MeanSquaredDisplacementResult]
):
    """Compute mean squared displacement from a trajectory.

    The mean squared displacement (MSD) measures particle mobility over time.
    Two calculation modes are supported:

    - **'window'** (default): Average over all windows of length m
      MSD(m) = ⟨(r(t+m) - r(t))²⟩ averaged over all t and particles

    - **'direct'**: Direct displacement from initial position
      MSD(t) = ⟨(r(t) - r(0))²⟩ averaged over all particles

    The window mode uses FFT-based autocorrelation for efficient computation,
    following the algorithm described in Calandrini et al. (2011).

    Args:
        mode: Calculation mode, either 'window' or 'direct'.

    Examples:
        >>> compute = MeanSquaredDisplacementCompute(mode='window')
        >>> result = compute(trajectory)
        >>> print(f"MSD at lag 10: {result.msd[10]:.3f}")
        >>> print(f"Diffusion coefficient: {result.meta['diffusion_coeff']:.3e}")

    Developer Notes:
        - For trajectory computes, input is typically a Trajectory object
        - Use efficient algorithms (like FFT) for large datasets
        - Validate that trajectory properties are constant (box, n_particles)
        - Provide both aggregate and per-particle results
        - Use context to share expensive computations (e.g., unwrapped positions)

    References:
        Calandrini et al., "nMoldyn 3: Using task farming for a parallel
        spectroscopy-oriented analysis of molecular dynamics simulations",
        J. Comput. Chem. 32(8), 1477-1487 (2011).
    """

    def __init__(self, mode: str = "window", context=None) -> None:
        """Initialize the MSD compute.

        Args:
            mode: Calculation mode ('window' or 'direct').
            context: Optional shared context for expensive intermediates.
        """
        super().__init__(context)

        if mode not in ["window", "direct"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'window' or 'direct'.")

        self.mode = mode

    def before(self, input: Trajectory) -> None:
        """Validate input before computation.

        Args:
            input: Input trajectory.

        Raises:
            ValueError: If trajectory is invalid for MSD calculation.
        """
        # Check that we have at least 2 frames
        if len(input) < 2:
            raise ValueError(
                "Trajectory must have at least 2 frames for MSD calculation"
            )

        # Note: In a real implementation, you would also check that:
        # - Box is constant over trajectory
        # - Number of particles is constant
        # - Positions are unwrapped (or provide images)

    def compute(self, input: Trajectory) -> MeanSquaredDisplacementResult:
        """Calculate the mean squared displacement.

        Args:
            input: Input trajectory with particle positions over time.

        Returns:
            Result containing MSD values.
        """
        # Extract positions from trajectory
        # Note: This is a simplified example. In practice, you would:
        # 1. Unwrap positions if needed
        # 2. Handle images
        # 3. Check for constant box and particle count

        positions = self._extract_positions(input)

        if self.mode == "window":
            particle_msd = self._compute_window_msd(positions)
        else:  # mode == "direct"
            particle_msd = self._compute_direct_msd(positions)

        # Average over particles
        msd = particle_msd.mean(axis=-1)

        # Calculate diffusion coefficient (slope of MSD vs time)
        # D = lim_{t->∞} MSD(t) / (2 * d * t) where d is dimensionality
        if len(msd) > 10:
            # Use linear fit on latter half of data
            t = np.arange(len(msd) // 2, len(msd))
            slope, _ = np.polyfit(t, msd[t], 1)
            diffusion_coeff = slope / 6.0  # 2 * 3 dimensions
        else:
            diffusion_coeff = 0.0

        return MeanSquaredDisplacementResult(
            name="mean_squared_displacement",
            msd=msd,
            particle_msd=particle_msd,
            mode=self.mode,
            meta={
                "n_frames": len(msd),
                "n_particles": particle_msd.shape[1] if particle_msd.ndim > 1 else 0,
                "diffusion_coeff": float(diffusion_coeff),
            },
        )

    def _extract_positions(self, trajectory: Trajectory) -> NDArray[np.floating]:
        """Extract positions from trajectory.

        Args:
            trajectory: Input trajectory.

        Returns:
            Position array with shape (n_frames, n_particles, 3).
        """
        # This is a placeholder implementation
        # In a real implementation, you would iterate over frames and extract
        # the positions from each frame's atoms block

        positions_list = []
        for frame in trajectory:
            if "atoms" in frame:
                atoms = frame["atoms"]
                x = atoms["x"][:]
                y = atoms["y"][:]
                z = atoms["z"][:]
                coords = np.column_stack([x, y, z])
                positions_list.append(coords)

        return np.array(positions_list)

    def _compute_window_msd(
        self, positions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute MSD using windowed averaging with FFT optimization.

        This implements the efficient FFT-based algorithm for computing the
        windowed MSD, as described in Calandrini et al. (2011).

        Args:
            positions: Position array with shape (n_frames, n_particles, 3).

        Returns:
            Per-particle MSD with shape (n_frames, n_particles).
        """
        N = positions.shape[0]

        # First term: <r²(k+m)> - <r²(k)>
        D = np.square(positions).sum(axis=2)
        D = np.append(D, np.zeros(positions.shape[:2]), axis=0)
        Q = 2 * D.sum(axis=0)

        S1 = np.zeros(positions.shape[:2])
        for m in range(N):
            Q -= D[m - 1, :] + D[N - m, :]
            S1[m, :] = Q / (N - m)

        # Second term: compute via autocorrelation using FFT
        S2 = np.zeros(positions.shape[:2])
        for i in range(3):  # x, y, z
            S2 += self._autocorrelation(positions[:, :, i])

        return S1 - 2 * S2

    def _compute_direct_msd(
        self, positions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute MSD using direct displacement from initial position.

        Args:
            positions: Position array with shape (n_frames, n_particles, 3).

        Returns:
            Per-particle MSD with shape (n_frames, n_particles).
        """
        # Direct calculation: (r(t) - r(0))²
        displacements = positions - positions[[0], :, :]
        return np.sum(displacements**2, axis=-1)

    def _autocorrelation(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute autocorrelation using FFT.

        Args:
            x: Input array with shape (n_frames, n_particles).

        Returns:
            Autocorrelation with shape (n_frames, n_particles).
        """
        N = x.shape[0]

        # Use numpy FFT (in production, could use scipy.fftpack or pyfftw)
        F = np.fft.fft(x, n=2 * N, axis=0)
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD, axis=0)
        res = (res[:N]).real

        # Normalize by number of points
        n = np.arange(1, N + 1)[::-1]  # N to 1
        return res / n[:, np.newaxis]


# =============================================================================
# Summary for Developers
# =============================================================================

"""
Summary of Best Practices
--------------------------

1. **Keep compute logic simple and focused**
   - One compute = one calculation
   - Don't try to do too much in a single compute
   - Use composition for complex workflows

2. **Handle edge cases gracefully**
   - Check for missing data
   - Provide sensible defaults
   - Don't crash on empty inputs

3. **Use type annotations everywhere**
   - Helps with IDE support
   - Catches errors early
   - Makes code self-documenting

4. **Leverage context for expensive operations**
   - Unwrapping positions
   - Building neighbor lists
   - Computing distance matrices

5. **Store useful metadata**
   - Computation parameters
   - Derived quantities
   - Timestamps, version info, etc.

6. **Write comprehensive docstrings**
   - Explain what the compute does
   - Document all parameters
   - Provide usage examples
   - Reference papers for algorithms

7. **Test thoroughly**
   - Empty inputs
   - Single particle/frame
   - Edge cases
   - Known analytical results

Next Steps
----------

To create your own compute:

1. Copy one of the examples above as a starting point
2. Modify the Result class to hold your output
3. Implement the compute() method with your logic
4. Add tests in tests/test_compute.py
5. Update __init__.py to export your new compute

Happy computing! 🚀
"""
