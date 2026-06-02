"""Result classes for compute operations.

This module defines result types returned by compute operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Result:
    """Base class for computation results.

    Subclasses should define specific fields for their result data.
    """

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TimeSeriesResult(Result):
    """Base class for time-series analysis results.

    Attributes:
        time: Time points for the analysis (in ps or frames)
    """

    time: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class MCDResult(TimeSeriesResult):
    """Results from Mean Displacement Correlation calculation.

    Attributes:
        time: Time lag values (in ps)
        correlations: Dictionary mapping tag names to correlation function arrays (MSD values).
            Each array has shape (n_time_lags,)
    """

    correlations: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class PMSDResult(TimeSeriesResult):
    """Results from Polarization Mean Square Displacement calculation.

    Attributes:
        time: Time lag values (in ps)
        pmsd: Polarization MSD values at each time lag, shape (n_time_lags,)
    """

    pmsd: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class OnsagerResult(TimeSeriesResult):
    """Results from an Onsager collective-displacement cross-correlation.

    Attributes:
        time: Time lag values (in ps), shape (n_time_lags,).
        correlations: Mapping from tag ``"i,j"`` to the cross-correlation
            ``L_ij(tau) = <DP_i(tau).DP_j(tau)>`` of the collective (summed)
            species displacements, shape (n_time_lags,), units A^2. The
            diagonal ``"i,i"`` is the collective MSD of species ``i``.
    """

    correlations: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class JACFResult(TimeSeriesResult):
    """Results from a Green-Kubo current-autocorrelation conductivity.

    Attributes:
        time: Time lag values (in ps), shape (n_time_lags,).
        jacf: Current autocorrelation ``C(tau) = <J(0).J(tau)>``,
            (e*A/ps)^2, shape (n_time_lags,).
        sigma_running: Running Green-Kubo conductivity integral
            ``sigma(tau) = 1/(3 V kB T) integral_0^tau C(t) dt`` (S/m),
            shape (n_time_lags,).
        sigma: DC ionic conductivity (S/m) — ``sigma_running`` at the final lag.
    """

    jacf: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    sigma_running: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    sigma: float = float("nan")


@dataclass
class PersistResult(TimeSeriesResult):
    """Results from a pair-survival (persistence) correlation.

    Attributes:
        time: Time lag values (in ps), shape (n_time_lags,).
        correlations: Mapping from tag ``"i,j:method:r0[,r1]"`` to the
            persistence correlation ``C(tau)`` (mean surviving partners per
            reference particle), shape (n_time_lags,). ``C(0)`` is the mean
            coordination number.
    """

    correlations: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class ACFResult(TimeSeriesResult):
    """Autocorrelation function result.

    Attributes:
        time: Time lag values (in ps)
        acf: Autocorrelation values at each time lag, shape (n_lags,)
        n_lags: Number of time lags
    """

    acf: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    n_lags: int = 0


@dataclass
class SpectralResult(Result):
    """Frequency-domain spectrum result.

    Attributes:
        frequency: Angular frequency grid omega, shape (n_freq,), units rad/ps.
        spectrum: Spectral density at each frequency, shape (n_freq,).
    """

    frequency: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    spectrum: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class DielectricResult(Result):
    """Single-route dielectric susceptibility result.

    Attributes:
        frequency: Angular frequency grid omega, shape (n_freq,), units rad/ps.
            Bin 0 is DC; bin 1 is Delta-omega = 2 * pi / (n_pad * dt).
        epsilon_real: Real part epsilon'(omega), shape (n_freq,), dimensionless.
        epsilon_imag: Loss spectrum epsilon''(omega), shape (n_freq,),
            dimensionless, positive sign convention.
        epsilon_static: Static dielectric constant epsilon(0), dimensionless.
            May be `nan` if the route does not provide a static estimate.
        epsilon_inf: High-frequency dielectric constant.
        route: Computation route ("einstein-helfand" or "green-kubo").
        component: System component ("full", "water", "ion").
        conductivity: Optional conductivity spectrum sigma(omega), shape (n_freq,).
    """

    frequency: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    epsilon_real: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    epsilon_imag: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    epsilon_static: float = float("nan")
    epsilon_inf: float = 1.0
    route: str = ""
    component: str = ""
    conductivity: NDArray[np.float64] | None = None

    def fit_debye(self) -> "DebyeFit":
        """Fit a single Debye relaxation to this spectrum (NumPy only).

        Uses the exact single-Debye identity
        ``epsilon''(omega) / (epsilon'(omega) - epsilon_inf) = omega * tau``:
        ``tau`` is the least-squares slope through the origin of that ratio
        versus ``omega`` over the low-frequency rising branch (up to the loss
        peak), with a loss-peak fallback ``tau = 1 / omega_peak``. The
        relaxation strength is the static limit
        ``delta_eps = epsilon(0) - epsilon_inf``.

        No SciPy: the estimator is closed-form linear regression. For broadened
        or skewed (Cole-Cole / Havriliak-Negami) line shapes do a nonlinear fit
        in your analysis script using :meth:`DebyeFit.epsilon` as the model.

        Returns:
            DebyeFit with tau (ps), delta_eps, eps_inf, eps_static, omega_peak.
        """
        w = np.asarray(self.frequency, dtype=np.float64)
        er = np.asarray(self.epsilon_real, dtype=np.float64)
        ei = np.asarray(self.epsilon_imag, dtype=np.float64)
        eps_inf = float(self.epsilon_inf)
        eps_static = (
            float(self.epsilon_static)
            if np.isfinite(self.epsilon_static)
            else float(er[0])
        )
        delta_eps = eps_static - eps_inf

        # Loss peak, skipping the DC bin (index 0).
        pk = int(np.argmax(ei[1:]) + 1) if ei.size > 2 else ei.size - 1
        omega_peak = float(w[pk]) if w.size > pk else float("nan")

        # Linear-ratio fit through the origin over the rising branch [1, pk].
        hi = max(pk + 1, 2)
        ws = w[1:hi]
        ers = er[1:hi]
        eis = ei[1:hi]
        good = (ws > 0.0) & (ers - eps_inf > 1e-12)
        tau = float("nan")
        if np.any(good):
            wg = ws[good]
            ratio = eis[good] / (ers[good] - eps_inf)  # = omega * tau
            denom = float(np.dot(wg, wg))
            if denom > 0.0:
                tau = float(np.dot(wg, ratio) / denom)
        if not np.isfinite(tau) or tau <= 0.0:
            tau = 1.0 / omega_peak if omega_peak > 0.0 else float("nan")

        return DebyeFit(
            tau=tau,
            delta_eps=delta_eps,
            eps_inf=eps_inf,
            eps_static=eps_static,
            omega_peak=omega_peak,
        )


@dataclass
class DebyeFit:
    """Single-Debye relaxation parameters fitted from a dielectric spectrum.

    Attributes:
        tau: Relaxation time (ps).
        delta_eps: Relaxation strength epsilon(0) - epsilon_inf (dimensionless).
        eps_inf: High-frequency permittivity used in the fit.
        eps_static: Static permittivity epsilon(0) (dimensionless).
        omega_peak: Angular frequency of the dielectric-loss peak (rad/ps).
    """

    tau: float = float("nan")
    delta_eps: float = float("nan")
    eps_inf: float = 1.0
    eps_static: float = float("nan")
    omega_peak: float = float("nan")

    def epsilon(self, omega: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        """Evaluate the fitted Debye model at angular frequencies ``omega``.

        Returns:
            ``(epsilon_real, epsilon_imag)`` under the positive-loss convention
            ``epsilon* = epsilon' - i epsilon''``.
        """
        omega = np.asarray(omega, dtype=np.float64)
        x = omega * self.tau
        denom = 1.0 + x * x
        eps_real = self.eps_inf + self.delta_eps / denom
        eps_imag = self.delta_eps * x / denom
        return eps_real, eps_imag


@dataclass
class ConductivityResult(TimeSeriesResult):
    """Einstein-Helfand ionic-conductivity result.

    Attributes:
        time: MSD lag times tau (ps), shape (n_lags,).
        msd: Collective MSD <|M_J(t+tau) - M_J(t)|^2> of the ionic
            translational dipole, (e*A)^2, shape (n_lags,).
        sigma: Static ionic conductivity sigma (S/m).
        slope: Fitted MSD slope over the diffusive window, (e*A)^2/ps.
        fit_start: First lag index used in the linear fit (inclusive).
        fit_end: Last lag index used in the linear fit (exclusive).
    """

    msd: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    sigma: float = float("nan")
    slope: float = float("nan")
    fit_start: int = 0
    fit_end: int = 0


@dataclass
class DielectricSusceptibilityResult(Result):
    """Aggregate dielectric susceptibility result.

    Attributes:
        results: Mapping from route-component key to DielectricResult
        metadata: Trajectory parameters and computation info
    """

    results: dict[str, DielectricResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize with nested DielectricResult recursion."""
        d = super().to_dict()
        d["results"] = {k: v.to_dict() for k, v in self.results.items()}
        return d
