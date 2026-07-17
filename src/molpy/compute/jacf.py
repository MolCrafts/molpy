"""Ionic conductivity from the charge-current autocorrelation (Green-Kubo).

The DC ionic conductivity follows from the Green-Kubo relation for the
collective charge current ``J(t) = sum_a q_a v_a(t)``::

    sigma = 1 / (3 V kB T) * integral_0^inf <J(0).J(t)> dt .

This wrapper assembles the collective current ``J = sum v_cation - sum v_anion``
(unit charges +/-1) from per-atom velocities and delegates the current
autocorrelation + trapezoidal Green-Kubo integral to Rust via the explicit
raw-compute + fit composition (``molrs.GreenKuboConductivity`` for the raw
current ACF, then ``molrs.RunningIntegral`` for the cumulative integral). The
only Python-side step is multiplying by the fixed Green-Kubo unit prefactor
``e^2 A^2 ps^-1 / (3 A^3 kB)`` to convert the integral to SI ``S/m`` — no ACF /
integration math is reimplemented in Python.

Units (LAMMPS *real*, matching :mod:`molpy.compute.dielectric`): velocities in
``A/ps`` (so ``J`` is ``e*A/ps``), ``dt`` in ps, volume in ``A^3``,
temperature in K; the output ``sigma`` is SI ``S/m``. Pass velocities already in
``A/ps``; conductivity scales linearly so other velocity units can be rescaled
afterwards.

Adapted from the tame library (https://github.com/Roy-Kid/tame),
``tame/recipes/jacf.py`` (whose published version never evaluates the
autocorrelation before integrating — corrected here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molrs import GreenKuboConductivity as _MolrsGreenKuboConductivity
from molrs import RunningIntegral as _MolrsRunningIntegral

from .base import Compute
from .result import JACFResult

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory

# SI constants for the Green-Kubo unit prefactor (CODATA 2018), matching
# molrs::units::constants used by the legacy Rust kernel.
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_BOLTZMANN_SI = 1.380649e-23
_ANGSTROM_M = 1e-10
_PICOSECOND_S = 1e-12
# σ = prefactor · ∫⟨J(0)·J(t)⟩ dt / (V·T), Green-Kubo factor 1/3. Folds in e²,
# Å→m, and ps→s so the caller works in LAMMPS *real* units in / SI S/m out.
_GREEN_KUBO_PREFACTOR = (
    _ELEMENTARY_CHARGE_C
    * _ELEMENTARY_CHARGE_C
    * _ANGSTROM_M
    * _ANGSTROM_M
    / _PICOSECOND_S
) / (3.0 * _ANGSTROM_M * _ANGSTROM_M * _ANGSTROM_M * _BOLTZMANN_SI)


class JACF(Compute):
    """Green-Kubo ionic conductivity from the charge-current autocorrelation.

    Args:
        cation_type: Atom type index for cations (charge +1).
        anion_type: Atom type index for anions (charge -1).
        max_dt: Maximum correlation time in ps.
        dt: Timestep in ps.
        temperature: Temperature in K.
        volume: System volume in A^3. If ``None``, the mean box volume over the
            trajectory is used.

    Examples:
        >>> from molpy.io import read_h5_trajectory
        >>> traj = read_h5_trajectory("ionic_liquid.h5")
        >>> jacf = JACF(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01,
        ...             temperature=300.0)
        >>> result = jacf(traj)
        >>> result.sigma          # DC ionic conductivity, S/m
        >>> result.jacf           # <J(0).J(t)>, shape (n_cache,)
    """

    def __init__(
        self,
        cation_type: int,
        anion_type: int,
        max_dt: float,
        dt: float,
        temperature: float,
        volume: float | None = None,
    ):
        super().__init__(
            cation_type=cation_type,
            anion_type=anion_type,
            max_dt=max_dt,
            dt=dt,
            temperature=temperature,
            volume=volume,
        )
        self.cation_type = cation_type
        self.anion_type = anion_type
        self.max_dt = max_dt
        self.dt = dt
        self.temperature = temperature
        self.volume = volume
        self.n_cache = int(max_dt / dt)

    def __call__(self, trajectory: "Trajectory") -> JACFResult:
        current_list: list[NDArray] = []
        volumes: list[float] = []

        for frame in trajectory:
            if "atoms" not in frame:
                raise ValueError("Frame must contain 'atoms' block")
            atoms = frame["atoms"]
            for col in ("vx", "vy", "vz", "type"):
                if col not in atoms:
                    raise ValueError(f"Atoms block must contain '{col}'")
            vel = np.column_stack([atoms["vx"], atoms["vy"], atoms["vz"]])
            elems = np.asarray(atoms["type"])
            c_mask = elems == self.cation_type
            a_mask = elems == self.anion_type
            j = np.sum(vel[c_mask], axis=0) - np.sum(vel[a_mask], axis=0)  # (3,)
            current_list.append(j)
            if frame.simbox is None or frame.simbox.is_free:
                raise ValueError(
                    "Frame must carry a non-free Box (volume is required for the "
                    "Green-Kubo conductivity prefactor)."
                )
            volumes.append(float(frame.simbox.volume()))

        current = np.asarray(current_list, dtype=np.float64)  # (F, 3)
        if current.shape[0] < 2:
            raise ValueError(f"Need at least 2 frames, got {current.shape[0]}")
        volume = self.volume if self.volume is not None else float(np.mean(volumes))

        # Raw current ACF in Rust (no fitted sigma), then the cumulative
        # trapezoidal integral in Rust; the only Python step is the SI prefactor.
        raw = _MolrsGreenKuboConductivity().compute(
            np.ascontiguousarray(current),
            self.dt,
            self.n_cache - 1,
        )
        integ = _MolrsRunningIntegral().fit(raw["jacf"], self.dt)
        sigma_running = (
            _GREEN_KUBO_PREFACTOR * integ["integral"] / (volume * self.temperature)
        )
        time_array = np.arange(self.n_cache, dtype=np.float64) * self.dt
        return JACFResult(
            time=time_array,
            jacf=raw["jacf"],
            sigma_running=sigma_running,
            sigma=float(sigma_running[-1]),
        )
