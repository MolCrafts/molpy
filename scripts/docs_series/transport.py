"""Datasets for the transport pages: MSD and VACF.

Both observables come from the same ensemble of independent argon runs (one per
seed in ``run.SEEDS``), which is why the docs can show that the Einstein and
Green-Kubo routes to the diffusion coefficient agree. Lennard-Jones dynamics is
chaotic: a single 30 ps run of 500 atoms is one noisy draw, and its long-lag MSD
and fitted D move by tens of percent from seed to seed. So the figures show the
ensemble mean, and every diffusion coefficient is the mean over the seeds with
its sample standard deviation and the standard error of that mean.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import molpy as mp
from molpy.compute import (
    Correlator,
    CumulativeTrapezoid,
    LinearFit,
    Msd,
    PairPresence,
    Survival,
    TimeCorrelation,
    Vacf,
)
from molpy.core import NeighborQuery

from .lj import (
    ACCEL,
    ANGSTROM2_PER_FS_TO_CM2_PER_S,
    FS_TO_PS,
    KB,
    ArgonLj,
    Trajectory,
)
from .structure import write_json

#: Linear-response fitting window for the diffusive regime, fs.
FIT_START = 5000.0
FIT_END = 20000.0

#: VACF lags kept, in frames (2.5 ps at 10 fs).
VACF_MAX_LAG = 250


def _velocity_correlation(trajectory: Trajectory) -> TimeCorrelation:
    """<v(0).v(t)> of one run, averaged over atoms and time origins.

    This is the product ``Correlator`` with the mean kept, not the ``Vacf``
    surface: ``Vacf`` subtracts each velocity component's mean over the run,
    and over a finite run that mean is the atom's net displacement divided by
    the run length. Subtracting it removes about v_bar^2 * t from the running
    integral, which on these 30 ps runs lowers the Green-Kubo D by ~13 %.
    In equilibrium the true mean is zero, so the kept-mean curve is the
    Green-Kubo integrand.
    """
    correlator = Correlator("product", VACF_MAX_LAG)
    return correlator.compute(a=trajectory.velocities, dt=trajectory.dt)


def _vacf_surface(trajectory: Trajectory) -> TimeCorrelation:
    """The ``Vacf`` surface: per-component, mean-subtracted (for comparison)."""
    n_frames = trajectory.velocities.shape[0]
    velocities = trajectory.velocities.reshape(n_frames, -1)
    return Vacf(VACF_MAX_LAG).compute(velocities=velocities, dt=trajectory.dt)


def _green_kubo(correlation: TimeCorrelation, dimension: int = 3) -> np.ndarray:
    """Running Green-Kubo integral D(t) = (1/d) integral_0^t C(s) ds, A^2/fs."""
    integral = CumulativeTrapezoid().compute(correlation=correlation, channel=0)
    return integral.channel(0) / dimension


def _spread(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample standard deviation and standard error of the mean."""
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1))
    return {
        "mean": float(array.mean()),
        "std": std,
        "sem": std / np.sqrt(len(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summary(prefix: str, spread: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in spread.items()}


def mean_squared_displacement(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """Ensemble-mean MSD(tau) and the Einstein diffusion coefficient per seed."""
    max_lag = trajectories[0].unwrapped.shape[0] - 1
    results = [
        Msd(max_lag).compute(positions=t.unwrapped, dt=t.dt) for t in trajectories
    ]
    curves = np.array([r.channel(0) for r in results])
    lag = results[0].lag_times
    msd = curves.mean(axis=0)
    temperature = float(np.mean([t.temperature for t in trajectories]))

    # D from each run's own straight-line fit over FIT_START..FIT_END: the
    # spread of those fits is the uncertainty the docs quote. LinearFit takes
    # the window as fractions of the lag axis.
    fit = LinearFit(FIT_START / lag[-1], FIT_END / lag[-1])
    diffusion = [fit.compute(correlation=r, channel=0).slope / 6.0 for r in results]
    diffusion_cm2_per_s = [d * ANGSTROM2_PER_FS_TO_CM2_PER_S for d in diffusion]
    mean_diffusion = float(np.mean(diffusion))

    # Drop lag 0: MSD(0) = 0 cannot be shown on logarithmic axes, and the
    # figure is about the crossover between power laws. Points are then
    # subsampled on a log grid — no smoothing, just every computed value that
    # falls on the grid, so the decade-per-decade shape is preserved without
    # shipping 3000 near-duplicate points.
    picked = np.unique(np.round(np.geomspace(1, len(msd) - 1, 260)).astype(int))

    # Both asymptotes are predictions, not fitted curves drawn by hand:
    # ballistic uses <v^2> = 3 k_B T / m at the runs' mean temperature, and
    # diffusive uses the seed-mean D.
    mean_square_speed = 3.0 * KB * temperature / ArgonLj().mass * ACCEL
    rows: list[dict[str, float | str]] = []
    for i in picked:
        rows.append(
            {
                "t": round(float(lag[i]), 2),
                "msd": round(float(msd[i]), 5),
                # Short legend labels — long Unicode titles ellipsize under
                # the docs type scale (see molplot fence layout notes).
                "series": "MSD",
            }
        )
    for i in picked:
        tau = float(lag[i])
        ballistic = mean_square_speed * tau**2
        if 1e-4 <= ballistic <= 60.0:
            rows.append(
                {
                    "t": round(tau, 2),
                    "msd": round(ballistic, 5),
                    "series": "τ²",
                }
            )
        diffusive = 6.0 * mean_diffusion * tau
        if 1e-4 <= diffusive <= 60.0:
            rows.append(
                {
                    "t": round(tau, 2),
                    "msd": round(diffusive, 5),
                    "series": "6Dτ",
                }
            )
    write_json("msd/argon_msd.json", rows)

    # Slope of log MSD vs log tau: 2 while ballistic, 1 once diffusive.
    short = (lag > 0) & (lag <= 50.0)
    ballistic_slope = np.polyfit(np.log(lag[short]), np.log(msd[short]), 1)[0]
    first = 1  # tau = dt
    return {
        "n_seeds": float(len(trajectories)),
        "temperature_K": temperature,
        **_summary("D_cm2_per_s", _spread(diffusion_cm2_per_s)),
        "loglog_slope_short": float(ballistic_slope),
        "msd_at_dt": float(msd[first]),
        "ballistic_at_dt": float(mean_square_speed * lag[first] ** 2),
    }


def velocity_autocorrelation(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """Ensemble-mean normalized VACF and the Green-Kubo D per seed."""
    results = [_velocity_correlation(t) for t in trajectories]
    acfs = np.array([r.channel(0) for r in results])
    dt = trajectories[0].dt
    lag = results[0].lag_times
    acf = acfs.mean(axis=0)
    temperature = float(np.mean([t.temperature for t in trajectories]))

    write_json(
        "vacf/argon_vacf.json",
        [
            {"t": round(float(a), 2), "c": round(float(b / acf[0]), 5)}
            for a, b in zip(lag, acf)
        ],
    )

    # Running Green-Kubo integral of the ensemble-mean curve.
    mean_running = _green_kubo(TimeCorrelation(lag, acf[:, None]))
    write_json(
        "vacf/argon_running_diffusion.json",
        [
            {
                "t": round(float(a), 2),
                "D": round(float(b * ANGSTROM2_PER_FS_TO_CM2_PER_S), 8),
            }
            for a, b in zip(lag, mean_running)
        ],
    )

    diffusion_cm2_per_s = [
        float(_green_kubo(r)[-1] * ANGSTROM2_PER_FS_TO_CM2_PER_S) for r in results
    ]
    # The Vacf surface averages over the 3 components, so its integral is D.
    surface_cm2_per_s = [
        float(_green_kubo(_vacf_surface(t), 1)[-1] * ANGSTROM2_PER_FS_TO_CM2_PER_S)
        for t in trajectories
    ]
    # C(0) = <v.v> = 3 k_B T / m by equipartition.
    equipartition = 3.0 * KB * temperature / ArgonLj().mass * ACCEL
    minimum = int(np.argmin(acf))
    crossing = int(np.argmax(acf < 0.0))
    peak = int(np.argmax(mean_running))
    return {
        "n_seeds": float(len(trajectories)),
        "temperature_K": temperature,
        "C0": float(acf[0]),
        "C0_expected_3kT_m": float(equipartition),
        "zero_crossing_fs": float(lag[crossing]),
        "min_lag_fs": float(lag[minimum]),
        "min_normalized": float(acf[minimum] / acf[0]),
        "running_peak_fs": float(lag[peak]),
        "running_peak_cm2_per_s": float(
            mean_running[peak] * ANGSTROM2_PER_FS_TO_CM2_PER_S
        ),
        "running_at_400fs_cm2_per_s": float(
            mean_running[int(round(400.0 / dt))] * ANGSTROM2_PER_FS_TO_CM2_PER_S
        ),
        "running_at_1500fs_cm2_per_s": float(
            mean_running[int(round(1500.0 / dt))] * ANGSTROM2_PER_FS_TO_CM2_PER_S
        ),
        **_summary("D_cm2_per_s", _spread(diffusion_cm2_per_s)),
        "D_vacf_surface_cm2_per_s_mean": float(np.mean(surface_cm2_per_s)),
    }


def pair_survival(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """First-shell residence correlation for argon, from the first run.

    Pair presence is read from one directed neighbour query per frame (each
    atom against all atoms, so every pair is seen from both ends, as a
    coordination number counts it), then correlated. Normalised per reference
    atom, C(0) is the first-shell coordination number.
    """
    trajectory = trajectories[0]
    n_frames = 2000
    box = mp.Box.cube(trajectory.box_length)
    # r0 / r1 bracket the first minimum of g(r): a pair counts as bonded inside
    # r0 and is only considered broken once it leaves r1 (a Stillinger-Rahman
    # style buffer that stops rattling at the boundary from breaking pairs).
    r0, r1 = 5.4, 5.9
    tables = [
        NeighborQuery(box, xyz, r1).query(xyz) for xyz in trajectory.wrapped[:n_frames]
    ]
    presence = PairPresence(r0, True, r1=r1).compute(frames=tables)
    # Runs start only where a pair is inside r0 (r1 = r0), so the buffer
    # between r0 and r1 extends a run but never starts one.
    inner = PairPresence(r0, True).compute(frames=tables)
    if not np.array_equal(inner.pairs, presence.pairs):
        raise ValueError("inner and buffered presence disagree on their pairs")
    n_atoms = trajectory.wrapped.shape[1]
    rows: list[dict[str, float | str]] = []
    summary: dict[str, float] = {}
    for method in ("continuous", "intermittent"):
        result = Survival(
            method, 600, normalization="per_reference", origin="inner"
        ).compute(
            presence=presence.values,
            dt=trajectory.dt,
            inner=inner.values,
            n_reference=n_atoms,
        )
        correlation = result.channel(0)
        lag = result.lag_times
        rows.extend(
            {
                "t": round(float(a), 1),
                "c": round(float(b / correlation[0]), 5),
                "series": method,
            }
            for a, b in zip(lag[::4], correlation[::4])
        )
        summary[f"C0_{method}"] = float(correlation[0])
        normalized = correlation / correlation[0]
        # Log-linear tail fit. This is an EXTRAPOLATION unless the curve
        # actually reaches 1/e inside the window, so report whether it did;
        # the docs must not quote a lifetime that was never observed.
        slope = np.polyfit(lag[lag > 1000], np.log(normalized[lag > 1000]), 1)[0]
        summary[f"tau_ps_{method}"] = float(-1.0 / slope * FS_TO_PS)
        summary[f"reached_1_over_e_{method}"] = float(normalized[-1] < np.exp(-1.0))
    write_json("persist/argon_survival.json", rows)
    return summary
