"""Dielectric susceptibility Compute classes.

Thin glue layers bridging molpy `Trajectory` to molrs computational
kernels. The Python side does only data extraction (positions, charges)
and vectorized NumPy assembly (dipole moment via `einsum`, minimum-image
unwrap via `Box.diff_dr`); all spectral physics — ACF, windowing, FFT,
prefactors — is performed in Rust by the raw computes
(`molrs.DebyeRelaxation`, `molrs.GreenKuboConductivity`) and the ε(ω) Fits
(`molrs.EinsteinHelfandSpectrum`, `molrs.GreenKuboSpectrum`), plus the raw
`molrs.dielectric` observables and `molrs.signal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molrs import DebyeRelaxation as _MolrsDebyeRelaxation
from molrs import EinsteinConductivity as _MolrsEinsteinConductivity
from molrs import EinsteinHelfandSpectrum as _MolrsEinsteinHelfandSpectrum
from molrs import GreenKuboConductivity as _MolrsGreenKuboConductivity
from molrs import GreenKuboSpectrum as _MolrsGreenKuboSpectrum
from molrs import LinearFit as _MolrsLinearFit
from molrs.dielectric import Dielectric
from molrs.signal import acf_fft, apply_window, frequency_grid

from ..core.box import Box
from .base import Compute
from .result import (
    ACFResult,
    ConductivityResult,
    DielectricResult,
    DielectricSusceptibilityResult,
    SpectralResult,
)

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory

# Treat ACF lag-0 values below this threshold as numerical zero (would
# otherwise blow up the normalization step in ACFAnalyzer).
_ACF_ZERO_LAG_EPSILON = 1e-30

# SI constants for the Einstein-Helfand conductivity unit prefactor (CODATA
# 2018), matching molrs::units::constants used by the legacy Rust kernel.
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_BOLTZMANN_SI = 1.380649e-23
_ANGSTROM_M = 1e-10
_PICOSECOND_S = 1e-12
# σ = prefactor · slope / (V·T), Einstein factor 1/6. Folds in e², Å→m, ps→s so
# the caller works in LAMMPS *real* units in / SI S/m out.
_EINSTEIN_HELFAND_PREFACTOR = (
    _ELEMENTARY_CHARGE_C
    * _ELEMENTARY_CHARGE_C
    * _ANGSTROM_M
    * _ANGSTROM_M
    / _PICOSECOND_S
) / (6.0 * _ANGSTROM_M * _ANGSTROM_M * _ANGSTROM_M * _BOLTZMANN_SI)


def _unwrap_inplace(coords: np.ndarray, frames: list) -> None:
    """Minimum-image unwrap of a ``(n_frames, n_atoms, 3)`` array, in place.

    Frame 0 is kept; each later frame is rebuilt from the previous
    (already-unwrapped) frame plus the minimum-image displacement, so a particle
    crossing a periodic boundary stays continuous. Uses the previous frame's box
    (NPT-correct) and caches the wrapped :class:`~molpy.core.box.Box` per unique
    cell matrix, so a constant-cell (NVT) trajectory wraps the box exactly once
    instead of once per frame.
    """
    cache: dict[bytes, Box] = {}
    for i in range(1, len(frames)):
        rs_box = frames[i - 1].box
        key = np.asarray(rs_box.matrix).tobytes()
        box = cache.get(key)
        if box is None:
            box = Box.from_box(rs_box)
            cache[key] = box
        coords[i] = coords[i - 1] + box.diff_dr(coords[i] - coords[i - 1])


class ACFAnalyzer(Compute):
    """Compute autocorrelation function from trajectory data.

    Extracts per-atom columns from each frame, optionally unwraps coordinates
    via Box.diff_dr, delegates to molrs.signal.acf_fft(), normalizes the
    ACF (divides by zero-lag value), and returns an ACFResult.
    """

    def __init__(
        self,
        columns: list[str],
        max_lag: int,
        *,
        unwrap: bool = True,
        **config_kwargs,
    ):
        super().__init__(
            columns=columns, max_lag=max_lag, unwrap=unwrap, **config_kwargs
        )
        self.columns = columns
        self.max_lag = max_lag
        self.unwrap = unwrap

    def __call__(self, trajectory: Trajectory) -> ACFResult:
        # Materialize once: trajectories may be one-shot iterators.
        frames = list(trajectory)
        n_frames = len(frames)
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")

        frame0 = frames[0]
        if self.unwrap and (frame0.box is None or frame0.box.is_free):
            raise ValueError(
                "Trajectory frames must have a non-free Box when unwrap=True"
            )
        for col in self.columns:
            if col not in frame0["atoms"]:
                raise ValueError(f"Missing column '{col}' in atoms block")

        n_dim = len(self.columns)
        n_atoms = len(frame0["atoms"]["x"])
        dt = frame0.metadata.get("dt", 1.0)

        data = np.empty((n_frames, n_atoms, n_dim), dtype=np.float64)
        for i, frame in enumerate(frames):
            for d, col in enumerate(self.columns):
                data[i, :, d] = frame["atoms"][col]

        # Unwrap via minimum-image convention (only meaningful for 3-component
        # columns). Shared helper caches the box wrap across frames.
        if self.unwrap and n_dim == 3:
            _unwrap_inplace(data, frames)

        # Compute ACF per dimension, average, normalize.
        max_lag = min(self.max_lag, n_frames - 1)
        acf_sum = np.zeros(max_lag + 1)
        for d in range(n_dim):
            col_data = data[:, :, d].mean(axis=1)  # (n_frames,) average over atoms
            acf_sum += acf_fft(col_data, max_lag)
        acf_sum /= n_dim
        if acf_sum[0] > _ACF_ZERO_LAG_EPSILON:
            acf_sum /= acf_sum[0]

        lag_times = np.arange(max_lag + 1, dtype=np.float64) * dt
        return ACFResult(time=lag_times, acf=acf_sum, n_lags=max_lag + 1)


class SpectralAnalyzer(Compute):
    """Convert time-domain ACF to frequency-domain spectrum.

    Applies a window function, generates the frequency grid, and performs
    the time→frequency conversion. All computation delegated to molrs.signal.
    """

    def __init__(
        self,
        dt: float,
        *,
        window_type: str = "hann",
        **config_kwargs,
    ):
        super().__init__(dt=dt, window_type=window_type, **config_kwargs)
        self.dt = dt
        self.window_type = window_type

    def __call__(self, acf_result: ACFResult) -> SpectralResult:
        acf = acf_result.acf
        n_lags = len(acf)

        # Apply window via molrs.signal
        windowed = apply_window(acf, self.window_type, axis=0)

        # Generate frequency grid via molrs.signal
        n_fft = 2 * (n_lags - 1)
        freq = frequency_grid(n_fft, self.dt)

        # Windowed ACF — the actual time→frequency FT happens downstream
        # in molrs.dielectric.{einstein_helfand,green_kubo}_spectrum.
        return SpectralResult(frequency=freq, spectrum=windowed)


class DielectricSusceptibility(Compute):
    """Frequency-dependent dielectric susceptibility from an MD trajectory.

    Extracts atomic positions and charges per frame, unwraps coordinates
    via minimum-image convention, builds the total dipole moment series,
    and runs one or more spectral routes (Einstein-Helfand and/or
    Green-Kubo) as the explicit raw-compute + ε(ω)-Fit composition:
    `molrs.DebyeRelaxation` (raw fluctuation dipole ACF) +
    `molrs.EinsteinHelfandSpectrum`, and `molrs.GreenKuboConductivity`
    (raw current ACF) + `molrs.GreenKuboSpectrum`. No spectra math runs in
    Python. The static dielectric constant is computed once via Neumann's
    fluctuation formula and attached to every result.

    Args:
        dt: Frame spacing in **ps**.
        temperature: Temperature in **K**.
        max_correlation_time: Longest ACF lag in **frames** (clamped to
            `n_frames - 1`). Practical choice: ≤ n_frames / 10.
        epsilon_inf: High-frequency (electronic) permittivity, dimensionless.
            Use 1.0 for non-polarizable force fields.
        window_type: `"hann"` or `"blackman"` window applied to the ACF
            before FFT.
        routes: Subset of `["einstein-helfand", "green-kubo"]`. Default
            runs both.
        volume: System volume in **Å³**. If `None`, uses `frame.box.volume`
            from the first frame (assumes NVT/NVE).

    Inputs:
        Each frame's `atoms` block must contain canonical columns
        `x`, `y`, `z` (**Å**) and `charge` (**e**). Frames must carry a
        non-free `Box`.
    """

    def __init__(
        self,
        dt: float,
        temperature: float,
        max_correlation_time: int,
        *,
        epsilon_inf: float = 1.0,
        window_type: str = "hann",
        routes: list[str] | None = None,
        volume: float | None = None,
        **config_kwargs,
    ):
        super().__init__(
            dt=dt,
            temperature=temperature,
            max_correlation_time=max_correlation_time,
            epsilon_inf=epsilon_inf,
            window_type=window_type,
            routes=routes,
            volume=volume,
            **config_kwargs,
        )
        self.dt = dt
        self.temperature = temperature
        self.max_correlation_time = max_correlation_time
        self.epsilon_inf = epsilon_inf
        self.window_type = window_type
        self.routes = routes or ["einstein-helfand", "green-kubo"]
        self._volume = volume

    def __call__(self, trajectory: Trajectory) -> DielectricSusceptibilityResult:
        frames = list(trajectory)
        n_frames = len(frames)
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")

        frame0 = frames[0]
        if frame0.box is None or frame0.box.is_free:
            raise ValueError("Trajectory frames must have a non-free Box")

        for col in ["x", "y", "z", "charge"]:
            if col not in frame0["atoms"]:
                raise ValueError(f"Missing column '{col}' in atoms block")

        n_atoms = len(frame0["atoms"]["x"])
        volume = self._volume if self._volume is not None else frame0.box.volume()

        positions = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        # Charges are taken once from frame 0: the dipole / current formulas
        # assume fixed per-atom charges (standard non-polarizable FF), so they
        # are intentionally not re-read per frame.
        charges = np.asarray(frame0["atoms"]["charge"], dtype=np.float64)
        for i, frame in enumerate(frames):
            positions[i, :, 0] = frame["atoms"]["x"]
            positions[i, :, 1] = frame["atoms"]["y"]
            positions[i, :, 2] = frame["atoms"]["z"]

        # Minimum-image unwrap (shared helper; caches the box wrap per cell).
        _unwrap_inplace(positions, frames)

        # Dipole moment per frame: M[f, d] = Σ_a charges[a] · positions[f, a, d]
        dipole_moments = np.ascontiguousarray(
            np.einsum("a,fad->fd", charges, positions)
        )

        # Static dielectric constant is route-independent — compute once so
        # every result carries it (avoids the dual problem of the GK route
        # returning a silent 0.0 fallback when EH was not requested).
        eps_stat = Dielectric.static_dielectric_constant(
            dipole_moments, volume, self.temperature, self.epsilon_inf
        )

        results: dict[str, DielectricResult] = {}

        for route in self.routes:
            if route == "einstein-helfand":
                # Raw fluctuation dipole ACF + ⟨M²⟩ (DebyeRelaxation) → ε(ω) Fit.
                raw = _MolrsDebyeRelaxation(
                    volume, self.temperature, "tinfoil"
                ).compute(dipole_moments, self.dt, self.max_correlation_time)
                spec = _MolrsEinsteinHelfandSpectrum(
                    self.dt,
                    volume,
                    self.temperature,
                    self.epsilon_inf,
                    raw["zero_lag_variance"],
                ).fit(raw["acf"])
                results["EH-full"] = DielectricResult(
                    frequency=spec["frequencies"],
                    epsilon_real=spec["eps_real"],
                    epsilon_imag=spec["eps_imag"],
                    epsilon_static=eps_stat,
                    epsilon_inf=self.epsilon_inf,
                    route="einstein-helfand",
                    component="full",
                )

            if route == "green-kubo":
                # Current density (raw; row 0 is NaN, no previous frame). Skip
                # row 0, then take the raw current ACF (GreenKuboConductivity)
                # over the post-NaN series and transform it with the ε(ω) Fit.
                current_density = Dielectric.compute_current_density(
                    dipole_moments, self.dt, volume
                )
                current_post = np.ascontiguousarray(current_density[1:])
                raw = _MolrsGreenKuboConductivity().compute(
                    current_post, self.dt, self.max_correlation_time
                )
                spec = _MolrsGreenKuboSpectrum(
                    self.dt,
                    volume,
                    self.temperature,
                    self.epsilon_inf,
                    self.window_type,
                ).fit(raw["jacf"])
                results["GK-full"] = DielectricResult(
                    frequency=spec["frequencies"],
                    epsilon_real=spec["eps_real"],
                    epsilon_imag=spec["eps_imag"],
                    epsilon_static=eps_stat,
                    epsilon_inf=self.epsilon_inf,
                    route="green-kubo",
                    component="full",
                )

        return DielectricSusceptibilityResult(
            results=results,
            metadata={
                "dt": self.dt,
                "temperature": self.temperature,
                "n_frames": n_frames,
                "volume": volume,
            },
        )


class IonicConductivity(Compute):
    """Static ionic conductivity sigma via the Einstein-Helfand relation.

    Builds the **ionic translational dipole** M_J(t) = sum_i q_i r_i(t) from the
    trajectory (minimum-image unwrapped, same as
    :class:`DielectricSusceptibility`), then composes the raw collective-dipole
    MSD (:class:`molrs.EinsteinConductivity`) with the diffusive-window slope
    (:class:`molrs.LinearFit`) and a ``slope / (6 V k_B T)`` S/m prefactor:

        sigma = lim_{t->inf} (1 / (6 V k_B T)) d/dt <|M_J(t) - M_J(0)|^2>.

    Decomposition is the caller's responsibility and is done with selection,
    not arithmetic: pass a trajectory whose ``charge`` column is non-zero **only
    on the mobile ions** (e.g. via a :class:`~molpy.Selector` over the ion
    atoms, or by zeroing solvent charges). Including the solvent rotational
    dipole here would contaminate the translational MSD.

    Args:
        dt: Frame spacing in **ps**.
        temperature: Temperature in **K**.
        max_correlation_time: Longest MSD lag in **frames** (clamped to
            ``n_frames - 1``). Practical choice: <= ``n_frames / 5``.
        volume: System volume in **A^3**. If ``None``, uses ``frame.box.volume``
            from the first frame (assumes NVT/NVE).
        fit_start_frac: Fraction of ``max_lag`` where the linear-fit window
            over the diffusive regime starts (default 0.1).
        fit_end_frac: Fraction of ``max_lag`` where that window ends
            (default 0.5). ``sigma`` is window-sensitive for few-carrier
            systems; report a range rather than a single digit.

    Inputs:
        Each frame's ``atoms`` block must contain ``x``, ``y``, ``z`` (**A**)
        and ``charge`` (**e**); frames must carry a non-free ``Box``.
    """

    def __init__(
        self,
        dt: float,
        temperature: float,
        max_correlation_time: int,
        *,
        volume: float | None = None,
        fit_start_frac: float = 0.1,
        fit_end_frac: float = 0.5,
        **config_kwargs,
    ):
        super().__init__(
            dt=dt,
            temperature=temperature,
            max_correlation_time=max_correlation_time,
            volume=volume,
            fit_start_frac=fit_start_frac,
            fit_end_frac=fit_end_frac,
            **config_kwargs,
        )
        self.dt = dt
        self.temperature = temperature
        self.max_correlation_time = max_correlation_time
        self._volume = volume
        self.fit_start_frac = fit_start_frac
        self.fit_end_frac = fit_end_frac

    def __call__(self, trajectory: Trajectory) -> ConductivityResult:
        frames = list(trajectory)
        n_frames = len(frames)
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")

        frame0 = frames[0]
        if frame0.box is None or frame0.box.is_free:
            raise ValueError("Trajectory frames must have a non-free Box")

        for col in ["x", "y", "z", "charge"]:
            if col not in frame0["atoms"]:
                raise ValueError(f"Missing column '{col}' in atoms block")

        n_atoms = len(frame0["atoms"]["x"])
        volume = self._volume if self._volume is not None else frame0.box.volume()

        positions = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        # Charges are taken once from frame 0: the dipole / current formulas
        # assume fixed per-atom charges (standard non-polarizable FF), so they
        # are intentionally not re-read per frame.
        charges = np.asarray(frame0["atoms"]["charge"], dtype=np.float64)
        for i, frame in enumerate(frames):
            positions[i, :, 0] = frame["atoms"]["x"]
            positions[i, :, 1] = frame["atoms"]["y"]
            positions[i, :, 2] = frame["atoms"]["z"]

        # Minimum-image unwrap (same convention as DielectricSusceptibility).
        _unwrap_inplace(positions, frames)

        # Ionic translational dipole M_J[f, d] = sum_a charges[a] * pos[f, a, d].
        translational_dipole = np.einsum("a,fad->fd", charges, positions)

        # Explicit raw-compute + fit: the collective-dipole MSD is measured in
        # Rust (no fitted sigma), then the diffusive-window OLS slope is the
        # analyst's LinearFit choice. The only Python step is the SI prefactor.
        raw = _MolrsEinsteinConductivity().compute(
            np.ascontiguousarray(translational_dipole),
            self.dt,
            self.max_correlation_time,
        )
        fit = _MolrsLinearFit(self.fit_start_frac, self.fit_end_frac).fit(
            raw["lag_times"], raw["msd"]
        )
        sigma = _EINSTEIN_HELFAND_PREFACTOR * fit["slope"] / (volume * self.temperature)
        return ConductivityResult(
            time=raw["lag_times"],
            msd=raw["msd"],
            sigma=sigma,
            slope=fit["slope"],
            fit_start=fit["fit_start"],
            fit_end=fit["fit_end"],
            meta={
                "dt": self.dt,
                "temperature": self.temperature,
                "n_frames": n_frames,
                "volume": volume,
            },
        )
