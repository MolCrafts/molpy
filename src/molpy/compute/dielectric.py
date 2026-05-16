"""Dielectric susceptibility Compute classes.

Thin glue layers bridging molpy Trajectory to molrs computational functions.
Zero NumPy physics computation in Python — all physics delegated to molrs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molrs.dielectric import (
    compute_current_density,
    einstein_helfand_spectrum,
    green_kubo_spectrum,
    static_dielectric_constant,
)
from molrs.signal import acf_fft, apply_window, frequency_grid

from .base import Compute
from .result import (
    ACFResult,
    DielectricResult,
    DielectricSusceptibilityResult,
    SpectralResult,
)

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory


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
        n_frames = sum(1 for _ in trajectory)
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")

        # Validate first frame
        frame0 = trajectory[0]
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

        # Extract and unwrap
        data = np.full((n_frames, n_atoms, n_dim), np.nan)
        for i, frame in enumerate(trajectory):
            for d, col in enumerate(self.columns):
                data[i, :, d] = frame["atoms"][col]
            if self.unwrap and i > 0:
                for a in range(n_atoms):
                    dr = np.array(
                        [data[i, a, d] - data[i - 1, a, d] for d in range(n_dim)]
                    )
                    unwrapped_dr = frame.box.diff_dr(dr)
                    for d in range(n_dim):
                        data[i, a, d] = data[i - 1, a, d] + unwrapped_dr[d]

        # Compute ACF per dimension, average, normalize
        max_lag = min(self.max_lag, n_frames - 1)
        acf_sum = np.zeros(max_lag + 1)
        for d in range(n_dim):
            col_data = data[:, :, d].mean(axis=1)  # (n_frames,) average over atoms
            acf_raw = acf_fft(col_data, max_lag)
            acf_sum += acf_raw
        acf_sum /= n_dim
        # Normalize: ACF[0] = 1.0
        if acf_sum[0] > 1e-30:
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

        # Windowed ACF becomes the spectrum (time→frequency FT happens
        # downstream in molrs.dielectric.{einstein_helfand,green_kubo}_spectrum)
        return SpectralResult(time=freq, frequency=freq, spectrum=windowed)


class DielectricSusceptibility(Compute["Trajectory", DielectricSusceptibilityResult]):
    """Compute dielectric susceptibility from MD trajectory.

    Extracts positions, velocities, and charges from Trajectory frames,
    delegates to molrs.dielectric for all physics computation, and assembles
    results into a DielectricSusceptibilityResult.
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

        # Extract positions and charges
        positions = np.zeros((n_frames, n_atoms, 3))
        charges = frame0["atoms"]["charge"].copy()

        for i, frame in enumerate(frames):
            for d, col in enumerate(["x", "y", "z"]):
                positions[i, :, d] = frame["atoms"][col]

        # Unwrap coordinates
        for i in range(1, n_frames):
            for a in range(n_atoms):
                dr = positions[i, a, :] - positions[i - 1, a, :]
                unwrapped_dr = frames[i - 1].box.diff_dr(dr)
                positions[i, a, :] = positions[i - 1, a, :] + unwrapped_dr

        # Compute dipole moment time series
        dipole_moments = np.zeros((n_frames, 3))
        for i in range(n_frames):
            for d in range(3):
                dipole_moments[i, d] = np.dot(charges, positions[i, :, d])

        # Compute current density
        current_density = compute_current_density(dipole_moments, self.dt, volume)

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
                eps_stat = static_dielectric_constant(
                    dipole_moments, volume, self.temperature, self.epsilon_inf
                )
                results["EH-full"] = DielectricResult(
                    time=spec["frequencies"],
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
                    time=spec["frequencies"],
                    frequency=spec["frequencies"],
                    epsilon_real=spec["epsilon_real"],
                    epsilon_imag=spec["epsilon_imag"],
                    epsilon_static=results.get(
                        "EH-full", DielectricResult()
                    ).epsilon_static,
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
