"""Dielectric susceptibility Compute classes.

Thin glue layers bridging molpy `Trajectory` to molrs computational
kernels. The Python side does only data extraction (positions, charges)
and vectorized NumPy assembly (dipole moment via `einsum`, minimum-image
unwrap); all correlators and spectral physics live in molrs:

* raw Computes: ``DebyeRelaxation``, ``GreenKuboConductivity``, ``DipoleRateCross``
* Fits: ``EinsteinHelfandSpectrum``, ``GreenKuboSpectrum``,
  ``DipoleRateCrossSpectrum``, ``DipoleAutocorrelationSpectrum``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molrs.compute.dielectric import Dielectric
from molrs.compute.fitting import LinearFit as _MolrsLinearFit
from molrs.compute.spectroscopy import (
    DipoleAutocorrelationSpectrum as _MolrsDipoleAutocorrelationSpectrum,
)
from molrs.compute.spectroscopy import (
    DipoleRateCrossSpectrum as _MolrsDipoleRateCrossSpectrum,
)
from molrs.compute.spectroscopy import (
    EinsteinHelfandSpectrum as _MolrsEinsteinHelfandSpectrum,
)
from molrs.compute.spectroscopy import GreenKuboSpectrum as _MolrsGreenKuboSpectrum
from molrs.compute.transport import DebyeRelaxation as _MolrsDebyeRelaxation
from molrs.compute.transport import DipoleRateCross as _MolrsDipoleRateCross
from molrs.compute.transport import EinsteinConductivity as _MolrsEinsteinConductivity
from molrs.compute.transport import GreenKuboConductivity as _MolrsGreenKuboConductivity
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


def _orth_mic_dr(dr: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """MIC for orthogonal PBC: ``dr - L * rint(dr/L)``."""
    return dr - lengths * np.rint(dr / lengths)


def _normalize_dielectric_routes(routes: list[str]) -> list[str]:
    """Map legacy aliases to physical route names."""
    alias = {
        "eq28": "dipole-rate-cross",
        "eq30": "dipole-autocorrelation",
        "dipole-rate-cross": "dipole-rate-cross",
        "dipole-autocorrelation": "dipole-autocorrelation",
        "green-kubo": "green-kubo",
        "einstein-helfand": "einstein-helfand",
    }
    out: list[str] = []
    for r in routes:
        key = r.strip().lower()
        if key not in alias:
            raise ValueError(
                f"unknown dielectric route {r!r}; expected one of "
                f"{sorted(set(alias.values()) | set(alias))}"
            )
        name = alias[key]
        if name not in out:
            out.append(name)
    return out


class DielectricSusceptibility(Compute):
    """Frequency-dependent dielectric susceptibility from an MD trajectory.

    Single-pass over ``trajectory``: online MIC unwrap of positions, accumulate
    only the dipole series ``M(t)`` (shape ``(n_frames, 3)``), then run
    Einstein–Helfand and/or Green–Kubo spectral routes in molrs. Does **not**
    store the full ``(n_frames, n_atoms, 3)`` coordinate tensor.

    Physics (molrs; LAMMPS *real* units on the kernels):

    * EH: ``DebyeRelaxation`` → ``EinsteinHelfandSpectrum``
    * GK: velocity current density when present, else FD ``Ṁ`` →
      ``GreenKuboConductivity`` → ``GreenKuboSpectrum``
    * ``dipole-rate-cross`` (alias ``eq28``): ``DipoleRateCross`` →
      ``DipoleRateCrossSpectrum``
    * ``dipole-autocorrelation`` (alias ``eq30``): ``DebyeRelaxation`` PACF →
      ``DipoleAutocorrelationSpectrum`` (``C(0)−iωĈ``; auto plateau)

    Performance notes (stream path):

    * Orthogonal NVT uses a pure-NumPy MIC (no per-frame Rust ``delta`` call).
    * Per-frame buffers for ``(n_atoms, 3)`` are recycled (no ``column_stack``).
    * Velocity columns are only read if a route needs current density
      (``green-kubo``).
    * Prefer :meth:`from_dipole_series` when ``M(t)`` / ``j(t)`` are already
      cached — skips trajectory IO entirely.

    Args:
        dt: Frame spacing in **ps** (caller-supplied; not inferred from files).
        temperature: Temperature in **K**.
        max_correlation_time: Longest ACF lag in **frames** (clamped to
            ``n_frames - 1``). Practical choice: ≤ ``n_frames / 10``.
        epsilon_inf: High-frequency permittivity (1.0 for non-polarizable FFs).
        window_type: ACF window for spectral fits — ``"hann"``,
            ``"blackman"``, ``"cosine_sq"``, or ``"none"`` (slides use none).
        routes: Subset of
            ``["einstein-helfand", "green-kubo", "dipole-rate-cross",
            "dipole-autocorrelation"]`` (aliases ``eq28`` / ``eq30`` accepted).
        volume: System volume in **Å³**. If ``None``, uses the mean frame
            volume (NVT/NVE-friendly).

    Inputs:
        Each frame's ``atoms`` block must already carry canonical columns
        ``x``, ``y``, ``z`` (**Å**) and ``charge`` (**e**). Frames must carry a
        non-free ``Box``. Charge assignment and file I/O are the caller's
        responsibility (via :mod:`molpy.io` + any prep script).
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
        self.routes = _normalize_dielectric_routes(
            routes or ["einstein-helfand", "green-kubo"]
        )
        self._volume = volume
        self.progress_every = int(config_kwargs.get("progress_every", 200_000))

    def _need_velocity_current(self) -> bool:
        return "green-kubo" in self.routes

    def from_dipole_series(
        self,
        dipole_moments: np.ndarray,
        *,
        volume: float,
        current_density: np.ndarray | None = None,
    ) -> DielectricSusceptibilityResult:
        """Run spectral routes from precomputed ``M(t)`` (and optional ``j(t)``).

        Use this when observables were reduced offline (or in a previous pass)
        so TRR is not re-read. ``dipole_moments`` shape ``(n_frames, 3)`` in
        e·Å; ``current_density`` shape ``(n_frames, 3)`` in e·Å⁻²·ps⁻¹ when
        providing velocity-based JACF.
        """
        M = np.ascontiguousarray(dipole_moments, dtype=np.float64)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError(f"dipole_moments must be (n, 3), got {M.shape}")
        jden = None
        if current_density is not None:
            jden = np.ascontiguousarray(current_density, dtype=np.float64)
            if jden.shape != M.shape:
                raise ValueError(
                    f"current_density shape {jden.shape} != dipole {M.shape}"
                )
        return self._spectra_from_series(
            M,
            jden,
            float(volume),
            use_velocity_current=jden is not None,
            stream_meta={},
        )

    def _spectra_from_series(
        self,
        dipole_moments: np.ndarray,
        jden_arr: np.ndarray | None,
        volume: float,
        *,
        use_velocity_current: bool,
        stream_meta: dict,
    ) -> DielectricSusceptibilityResult:
        n_frames = int(dipole_moments.shape[0])
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")
        max_lag = min(self.max_correlation_time, n_frames - 1)

        eps_stat = Dielectric.static_dielectric_constant(
            dipole_moments, volume, self.temperature, self.epsilon_inf
        )

        results: dict[str, DielectricResult] = {}
        meta_extra: dict = {
            "gk_current": (
                "velocity" if use_velocity_current else "finite_difference_Mdot"
            ),
            **stream_meta,
        }

        # Shared PACF raw once if either EH or dipole-autocorrelation is requested.
        need_pacf = ("einstein-helfand" in self.routes) or (
            "dipole-autocorrelation" in self.routes
        )
        debye_raw = (
            _MolrsDebyeRelaxation(volume, self.temperature, "tinfoil").compute(
                dipole_moments, self.dt, max_lag
            )
            if need_pacf
            else None
        )

        for route in self.routes:
            if route == "einstein-helfand":
                raw = debye_raw
                assert raw is not None
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
                if use_velocity_current and jden_arr is not None:
                    current_post = jden_arr
                else:
                    current_density = Dielectric.compute_current_density(
                        dipole_moments, self.dt, volume
                    )
                    current_post = np.ascontiguousarray(current_density[1:])
                raw = _MolrsGreenKuboConductivity().compute(
                    current_post, self.dt, max_lag
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

            if route == "dipole-rate-cross":
                # Raw C_ṀM (FD Ṁ + cartesian xcorr) + spectrum — both in molrs.
                meta_extra["dipole_rate"] = "finite_difference"
                raw = _MolrsDipoleRateCross().compute(dipole_moments, self.dt, max_lag)
                spec = _MolrsDipoleRateCrossSpectrum(
                    self.dt,
                    volume,
                    self.temperature,
                    self.epsilon_inf,
                    self.window_type,
                ).fit(raw["cross"])
                results["Eq28-full"] = DielectricResult(
                    frequency=spec["frequencies"],
                    epsilon_real=spec["eps_real"],
                    epsilon_imag=spec["eps_imag"],
                    epsilon_static=eps_stat,
                    epsilon_inf=self.epsilon_inf,
                    route="dipole-rate-cross",
                    component="full",
                )

            if route == "dipole-autocorrelation":
                # Reuses debye_raw PACF; spectrum uses C(0)−iωĈ + auto plateau.
                raw = debye_raw
                assert raw is not None
                pacf = raw["acf"]
                c0 = float(pacf[0])
                c_inf = float(pacf[-1])
                ratio = c_inf / c0 if abs(c0) > 0.0 else 0.0
                meta_extra["pacf_c0"] = c0
                meta_extra["pacf_c_inf"] = c_inf
                meta_extra["pacf_c_inf_over_c0"] = ratio
                meta_extra["pacf_subtract_plateau"] = abs(ratio) > 0.1
                spec = _MolrsDipoleAutocorrelationSpectrum(
                    self.dt,
                    volume,
                    self.temperature,
                    self.epsilon_inf,
                    self.window_type,
                    None,  # auto plateau
                ).fit(pacf)
                results["Eq30-full"] = DielectricResult(
                    frequency=spec["frequencies"],
                    epsilon_real=spec["eps_real"],
                    epsilon_imag=spec["eps_imag"],
                    epsilon_static=eps_stat,
                    epsilon_inf=self.epsilon_inf,
                    route="dipole-autocorrelation",
                    component="full",
                )

        return DielectricSusceptibilityResult(
            results=results,
            metadata={
                "dt": self.dt,
                "temperature": self.temperature,
                "n_frames": n_frames,
                "volume": volume,
                "max_correlation_time": max_lag,
                **meta_extra,
            },
        )

    def __call__(self, trajectory: Trajectory) -> DielectricSusceptibilityResult:
        import time as _time

        t_phase0 = _time.time()
        n_known: int | None = None
        n_attr = getattr(trajectory, "n_frames", None)
        if n_attr is not None:
            try:
                n_known = int(n_attr() if callable(n_attr) else n_attr)
            except Exception:
                n_known = None
        if n_known is not None:
            print(
                f"[DielectricSusceptibility] index/scan: n_frames={n_known} "
                f"({_time.time() - t_phase0:.1f}s)",
                flush=True,
            )

        want_j = self._need_velocity_current()
        charges: np.ndarray | None = None
        n_atoms = 0
        # Recycled SoA → AoS buffers (avoid per-frame column_stack allocations).
        pos = np.empty((0, 3), dtype=np.float64)
        vel = np.empty((0, 3), dtype=np.float64)
        # Recycled unwrap / wrap history (no pos.copy() per frame).
        prev_unwrapped_buf = np.empty((0, 3), dtype=np.float64)
        prev_wrapped_buf = np.empty((0, 3), dtype=np.float64)

        cap = int(n_known) if (n_known is not None and n_known > 0) else 4096
        dipole_moments = np.empty((cap, 3), dtype=np.float64)
        jden = np.empty((cap, 3), dtype=np.float64) if want_j else None

        use_velocity_current = False
        prev_unwrapped: np.ndarray | None = None
        prev_wrapped: np.ndarray | None = None
        prev_box = None
        box_obj: Box | None = None
        orth_lengths: np.ndarray | None = None  # fast MIC path
        n_frames = 0
        volume_sum = 0.0
        t_stream = _time.time()
        prog = max(self.progress_every, 1)

        def _ensure_cap(need: int) -> None:
            nonlocal dipole_moments, jden, cap
            if need <= cap:
                return
            new_cap = max(cap * 2, need)
            d2 = np.empty((new_cap, 3), dtype=np.float64)
            d2[:n_frames] = dipole_moments[:n_frames]
            dipole_moments = d2
            if jden is not None:
                j2 = np.empty((new_cap, 3), dtype=np.float64)
                j2[:n_frames] = jden[:n_frames]
                jden = j2
            cap = new_cap

        for frame in trajectory:
            if frame.box is None or frame.box.is_free:
                raise ValueError("Trajectory frames must have a non-free Box")
            atoms = frame["atoms"]

            vol_f = float(frame.box.volume())
            volume_sum += vol_f

            if charges is None:
                for col in ("x", "y", "z", "charge"):
                    if col not in atoms:
                        raise ValueError(f"Missing column '{col}' in atoms block")
                charges = np.ascontiguousarray(atoms["charge"], dtype=np.float64)
                n_atoms = int(charges.shape[0])
                pos = np.empty((n_atoms, 3), dtype=np.float64)
                prev_unwrapped_buf = np.empty((n_atoms, 3), dtype=np.float64)
                prev_wrapped_buf = np.empty((n_atoms, 3), dtype=np.float64)
                use_velocity_current = want_j and all(
                    c in atoms for c in ("vx", "vy", "vz")
                )
                if use_velocity_current:
                    vel = np.empty((n_atoms, 3), dtype=np.float64)
                else:
                    jden = None
                box_obj = Box.from_box(frame.box)
                # Orthogonal PBC → pure-NumPy MIC (avoids Rust call + zeros alloc/frame)
                try:
                    style = getattr(box_obj, "style", None)
                    is_orth = style is Box.Style.ORTHOGONAL or style == "orthogonal"
                except Exception:
                    is_orth = False
                if is_orth:
                    orth_lengths = np.asarray(
                        [box_obj.lx, box_obj.ly, box_obj.lz], dtype=np.float64
                    )

            # Fill recycled (n_atoms, 3) without column_stack
            pos[:, 0] = np.asarray(atoms["x"], dtype=np.float64)
            pos[:, 1] = np.asarray(atoms["y"], dtype=np.float64)
            pos[:, 2] = np.asarray(atoms["z"], dtype=np.float64)

            if prev_unwrapped is None:
                # First frame: unwrapped = wrapped (copy into recycled buffer).
                np.copyto(prev_unwrapped_buf, pos)
                unwrapped = prev_unwrapped_buf
            else:
                if box_obj is None or abs(vol_f - float(prev_box.volume())) > 1e-9:
                    box_obj = Box.from_box(prev_box)
                    orth_lengths = None
                    try:
                        style = getattr(box_obj, "style", None)
                        if style is Box.Style.ORTHOGONAL or style == "orthogonal":
                            orth_lengths = np.asarray(
                                [box_obj.lx, box_obj.ly, box_obj.lz], dtype=np.float64
                            )
                    except Exception:
                        pass
                # dr into a temporary view; update unwrapped in-place into buffer.
                dr = pos - prev_wrapped_buf
                if orth_lengths is not None:
                    prev_unwrapped_buf += _orth_mic_dr(dr, orth_lengths)
                else:
                    prev_unwrapped_buf += box_obj.diff_dr(dr)
                unwrapped = prev_unwrapped_buf

            # Save wrapped positions for next MIC step (recycled, no pos.copy()).
            np.copyto(prev_wrapped_buf, pos)
            prev_unwrapped = prev_unwrapped_buf
            prev_box = frame.box
            _ensure_cap(n_frames + 1)
            # M = q · R  (3-vector); BLAS gemv — stays in NumPy (already optimal).
            dipole_moments[n_frames] = charges @ unwrapped

            if use_velocity_current and jden is not None:
                vel[:, 0] = np.asarray(atoms["vx"], dtype=np.float64)
                vel[:, 1] = np.asarray(atoms["vy"], dtype=np.float64)
                vel[:, 2] = np.asarray(atoms["vz"], dtype=np.float64)
                jden[n_frames] = (charges @ vel) / vol_f

            n_frames += 1
            if n_frames % prog == 0:
                elapsed = _time.time() - t_stream
                rate = n_frames / max(elapsed, 1e-9)
                if n_known:
                    eta = (n_known - n_frames) / max(rate, 1e-9)
                    print(
                        f"[DielectricSusceptibility] stream {n_frames}/{n_known} "
                        f"({100 * n_frames / n_known:.1f}%) {rate:.0f} fr/s "
                        f"ETA {eta / 60:.1f} min",
                        flush=True,
                    )
                else:
                    print(
                        f"[DielectricSusceptibility] stream {n_frames} {rate:.0f} fr/s",
                        flush=True,
                    )

        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")
        if self._volume is not None:
            volume = float(self._volume)
        else:
            volume = volume_sum / n_frames

        dipole_moments = np.ascontiguousarray(dipole_moments[:n_frames])
        jden_arr = (
            np.ascontiguousarray(jden[:n_frames])
            if (use_velocity_current and jden is not None)
            else None
        )

        print(
            f"[DielectricSusceptibility] stream done: n_frames={n_frames} "
            f"in {(_time.time() - t_stream) / 60:.2f} min; "
            f"gk_current={'velocity' if use_velocity_current else 'Mdot'}; "
            f"mic={'orth-numpy' if orth_lengths is not None else 'box.diff_dr'}",
            flush=True,
        )
        return self._spectra_from_series(
            dipole_moments,
            jden_arr,
            volume,
            use_velocity_current=use_velocity_current,
            stream_meta={
                "mic_path": "orth-numpy" if orth_lengths is not None else "box.diff_dr"
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
