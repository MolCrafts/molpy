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
from molpy.compute import Acf, Msd, pair_survival_tcf

from .lj import (
    ACCEL,
    ANGSTROM2_PER_FS_TO_CM2_PER_S,
    FS_TO_PS,
    KB,
    ArgonLJ,
    Trajectory,
)
from .structure import write_json

#: Linear-response fitting window for the diffusive regime, fs.
FIT_START = 5000.0
FIT_END = 20000.0

#: VACF lags kept, in frames (2.5 ps at 10 fs).
VACF_MAX_LAG = 250


def _unwrapped_frames(trajectory: Trajectory) -> list[mp.Frame]:
    """Frames carrying continuous coordinates — required for displacements."""
    frames = []
    for xyz in trajectory.unwrapped:
        frame = mp.Frame()
        frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        frame.box = mp.Box.cube(trajectory.box_length)
        frames.append(frame)
    return frames


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
    curves = np.array(
        [
            np.asarray(Msd(method="window").compute(_unwrapped_frames(t)).mean)
            for t in trajectories
        ]
    )
    dt = trajectories[0].dt
    lag = np.arange(curves.shape[1]) * dt
    msd = curves.mean(axis=0)
    temperature = float(np.mean([t.temperature for t in trajectories]))

    # D from each run's own straight-line fit over FIT_START..FIT_END: the
    # spread of those fits is the uncertainty the docs quote.
    window = (lag >= FIT_START) & (lag <= FIT_END)
    diffusion = [np.polyfit(lag[window], curve[window], 1)[0] / 6.0 for curve in curves]
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
    mean_square_speed = 3.0 * KB * temperature / ArgonLJ().mass * ACCEL
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
    acfs = np.array(
        [
            np.asarray(
                Acf()
                .compute(np.ascontiguousarray(t.velocities), max_lag=VACF_MAX_LAG)
                .acf
            )
            for t in trajectories
        ]
    )
    dt = trajectories[0].dt
    lag = np.arange(acfs.shape[1]) * dt
    acf = acfs.mean(axis=0)
    temperature = float(np.mean([t.temperature for t in trajectories]))

    write_json(
        "vacf/argon_vacf.json",
        [
            {"t": round(float(a), 2), "c": round(float(b / acf[0]), 5)}
            for a, b in zip(lag, acf)
        ],
    )

    # Running Green-Kubo integral: D(t) = 1/3 \int_0^t C(s) ds.
    def running(curve: np.ndarray) -> np.ndarray:
        steps = 0.5 * (curve[1:] + curve[:-1]) * np.diff(lag)
        return np.concatenate([[0.0], np.cumsum(steps)]) / 3.0

    mean_running = running(acf)
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
        float(running(curve)[-1] * ANGSTROM2_PER_FS_TO_CM2_PER_S) for curve in acfs
    ]
    equipartition = 3.0 * KB * temperature / ArgonLJ().mass * ACCEL
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
    }


def pair_survival(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """First-shell residence correlation for argon, from the first run.

    Slow: the kernel walks every (i, j) pair at every lag, so this is minutes,
    not seconds. It is the only dataset here that is not near-instant.
    """
    trajectory = trajectories[0]
    n_frames = 2000
    positions = np.ascontiguousarray(trajectory.wrapped[:n_frames])
    box = np.tile(np.array([[trajectory.box_length] * 3]), (n_frames, 1))
    # r0 / r1 bracket the first minimum of g(r): a pair counts as bonded inside
    # r0 and is only considered broken once it leaves r1 (a Stillinger-Rahman
    # style buffer that stops rattling at the boundary from breaking pairs).
    rows: list[dict[str, float | str]] = []
    summary: dict[str, float] = {}
    for method in ("continuous", "intermittent"):
        result = pair_survival_tcf(
            positions, positions, box, 5.4, 5.9, method, trajectory.dt, 600, True
        )
        correlation = np.asarray(result["correlation"])
        lag = np.asarray(result["lag_times"])
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
