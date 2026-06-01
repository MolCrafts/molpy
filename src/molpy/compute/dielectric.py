"""Dielectric susceptibility Compute classes.

Thin glue layers bridging molpy `Trajectory` to molrs computational
kernels. The Python side does only data extraction (positions, charges)
and vectorized NumPy assembly (dipole moment via `einsum`, minimum-image
unwrap via `Box.diff_dr`); all spectral physics — ACF, windowing, FFT,
prefactors — is performed in Rust by `molrs.dielectric` and
`molrs.signal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molrs.dielectric import (
    compute_current_density,
    einstein_helfand_conductivity,
    einstein_helfand_spectrum,
    green_kubo_spectrum,
    static_dielectric_constant,
)
from molrs.signal import acf_fft, apply_window, frequency_grid

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
# otherwise blow up the normalization step). Same magnitude as the DC
# cutoff used on the Rust side in `dielectric::green_kubo_spectrum`.
_ACF_ZERO_LAG_EPSILON = 1e-30


class ACFAnalyzer(Compute["Trajectory", ACFResult]):
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

    def _compute(self, trajectory: Trajectory) -> ACFResult:
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

        # Unwrap via minimum-image convention. Box.diff_dr accepts the
        # whole (n_atoms, 3) displacement in one call, so the inner per-
        # atom Python loop collapses to a single vectorized call per
        # frame. Only meaningful for 3-component columns.
        if self.unwrap and n_dim == 3:
            for i in range(1, n_frames):
                dr = data[i] - data[i - 1]
                data[i] = data[i - 1] + frames[i].box.diff_dr(dr)

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


class SpectralAnalyzer(Compute[ACFResult, SpectralResult]):
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

    def _compute(self, acf_result: ACFResult) -> SpectralResult:
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


class DielectricSusceptibility(Compute["Trajectory", DielectricSusceptibilityResult]):
    """Frequency-dependent dielectric susceptibility from an MD trajectory.

    Extracts atomic positions and charges per frame, unwraps coordinates
    via minimum-image convention, builds the total dipole moment series,
    and runs one or more spectral routes (Einstein-Helfand and/or
    Green-Kubo) through `molrs.dielectric`. The static dielectric constant
    is also computed once via Neumann's fluctuation formula and attached
    to every result.

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

    def _compute(self, trajectory: Trajectory) -> DielectricSusceptibilityResult:
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
        volume = self._volume or frame0.box.volume

        positions = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        charges = np.asarray(frame0["atoms"]["charge"], dtype=np.float64)
        for i, frame in enumerate(frames):
            positions[i, :, 0] = frame["atoms"]["x"]
            positions[i, :, 1] = frame["atoms"]["y"]
            positions[i, :, 2] = frame["atoms"]["z"]

        # Vectorized minimum-image unwrap: Box.diff_dr takes the whole
        # (n_atoms, 3) slice at once. Use frames[i-1].box to match prior
        # semantics (relevant for NPT trajectories with per-frame boxes).
        for i in range(1, n_frames):
            dr = positions[i] - positions[i - 1]
            positions[i] = positions[i - 1] + frames[i - 1].box.diff_dr(dr)

        # Dipole moment per frame: M[f, d] = Σ_a charges[a] · positions[f, a, d]
        dipole_moments = np.einsum("a,fad->fd", charges, positions)

        # Compute current density (only needed for the GK route)
        current_density = None
        if "green-kubo" in self.routes:
            current_density = compute_current_density(dipole_moments, self.dt, volume)

        # Static dielectric constant is route-independent — compute once so
        # every result carries it (avoids the dual problem of the GK route
        # returning a silent 0.0 fallback when EH was not requested).
        eps_stat = static_dielectric_constant(
            dipole_moments, volume, self.temperature, self.epsilon_inf
        )

        results: dict[str, DielectricResult] = {}

        for route in self.routes:
            if route == "einstein-helfand":
                spec = einstein_helfand_spectrum(
                    dipole_moments,
                    self.dt,
                    volume,
                    self.temperature,
                    self.epsilon_inf,
                    self.max_correlation_time,
                    self.window_type,
                )
                results["EH-full"] = DielectricResult(
                    frequency=spec["frequencies"],
                    epsilon_real=spec["epsilon_real"],
                    epsilon_imag=spec["epsilon_imag"],
                    epsilon_static=eps_stat,
                    epsilon_inf=self.epsilon_inf,
                    route="einstein-helfand",
                    component="full",
                )

            if route == "green-kubo":
                spec = green_kubo_spectrum(
                    current_density,
                    self.dt,
                    volume,
                    self.temperature,
                    self.epsilon_inf,
                    self.max_correlation_time,
                    self.window_type,
                )
                results["GK-full"] = DielectricResult(
                    frequency=spec["frequencies"],
                    epsilon_real=spec["epsilon_real"],
                    epsilon_imag=spec["epsilon_imag"],
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


class IonicConductivity(Compute["Trajectory", ConductivityResult]):
    """Static ionic conductivity sigma via the Einstein-Helfand relation.

    Builds the **ionic translational dipole** M_J(t) = sum_i q_i r_i(t) from the
    trajectory (minimum-image unwrapped, same as
    :class:`DielectricSusceptibility`), then delegates the collective-MSD,
    linear fit, and S/m unit conversion to
    ``molrs.dielectric.einstein_helfand_conductivity``:

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
        fit_start_frac, fit_end_frac: Fractions of ``max_lag`` bounding the
            linear-fit window over the diffusive regime (default 0.1, 0.5).
            ``sigma`` is window-sensitive for few-carrier systems; report a
            range rather than a single digit.

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

    def _compute(self, trajectory: Trajectory) -> ConductivityResult:
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
        volume = self._volume or frame0.box.volume

        positions = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        charges = np.asarray(frame0["atoms"]["charge"], dtype=np.float64)
        for i, frame in enumerate(frames):
            positions[i, :, 0] = frame["atoms"]["x"]
            positions[i, :, 1] = frame["atoms"]["y"]
            positions[i, :, 2] = frame["atoms"]["z"]

        # Vectorized minimum-image unwrap (same convention as
        # DielectricSusceptibility): use frames[i-1].box for per-frame boxes.
        for i in range(1, n_frames):
            dr = positions[i] - positions[i - 1]
            positions[i] = positions[i - 1] + frames[i - 1].box.diff_dr(dr)

        # Ionic translational dipole M_J[f, d] = sum_a charges[a] * pos[f, a, d].
        translational_dipole = np.einsum("a,fad->fd", charges, positions)

        spec = einstein_helfand_conductivity(
            translational_dipole,
            self.dt,
            volume,
            self.temperature,
            self.max_correlation_time,
            self.fit_start_frac,
            self.fit_end_frac,
        )
        return ConductivityResult(
            time=spec["lag_times"],
            msd=spec["msd"],
            sigma=spec["sigma"],
            slope=spec["slope"],
            fit_start=spec["fit_start"],
            fit_end=spec["fit_end"],
            meta={
                "dt": self.dt,
                "temperature": self.temperature,
                "n_frames": n_frames,
                "volume": volume,
            },
        )
