"""Datasets for the transport pages: MSD and VACF.

Both observables come from the same argon trajectory, which is why the docs can
show that the Einstein and Green-Kubo routes to the diffusion coefficient agree.
"""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.compute import MSD, Acf, Persist

from .lj import KB, Trajectory
from .structure import write_json

#: Linear-response fitting window for the diffusive regime, fs.
FIT_START = 5000.0
FIT_END = 20000.0


def _unwrapped_frames(trajectory: Trajectory) -> list[mp.Frame]:
    """Frames carrying continuous coordinates — required for displacements."""
    frames = []
    for xyz in trajectory.unwrapped:
        frame = mp.Frame()
        frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        frame.box = mp.Box.cubic(trajectory.box_length)
        frames.append(frame)
    return frames


def mean_squared_displacement(trajectory: Trajectory) -> dict[str, float]:
    """MSD(tau) and the Einstein diffusion coefficient."""
    series = MSD(method="window")(_unwrapped_frames(trajectory))
    msd = np.asarray(series.mean)
    lag = np.arange(len(msd)) * trajectory.dt

    # Drop lag 0: MSD(0) = 0 cannot be shown on logarithmic axes, and the
    # figure is about the crossover between power laws. Points are then
    # subsampled on a log grid — no smoothing, just every computed value that
    # falls on the grid, so the decade-per-decade shape is preserved without
    # shipping 3000 near-duplicate points.
    picked = np.unique(np.round(np.geomspace(1, len(msd) - 1, 260)).astype(int))

    window = (lag >= FIT_START) & (lag <= FIT_END)
    slope, intercept = np.polyfit(lag[window], msd[window], 1)
    diffusion = slope / 6.0

    # Both asymptotes are predictions, not fitted curves drawn by hand:
    # ballistic uses <v^2> = 3 k_B T / m measured from the same run, and
    # diffusive uses the slope fitted over FIT_START..FIT_END.
    mean_square_speed = 3.0 * KB * trajectory.temperature / 39.948 * 4.184e-4
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
        diffusive = slope * tau
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

    return {
        "D_A2_per_fs": float(diffusion),
        "D_cm2_per_s": float(diffusion / 10.0),
        "msd_30ps": float(msd[-1]),
        "loglog_slope_short": float(ballistic_slope),
        "fit_intercept": float(intercept),
    }


def velocity_autocorrelation(trajectory: Trajectory) -> dict[str, float]:
    """Normalized VACF and the Green-Kubo diffusion coefficient."""
    velocities = np.ascontiguousarray(trajectory.velocities)
    result = Acf().compute(velocities, max_lag=250)
    acf = np.asarray(result.acf)
    lag = np.arange(len(acf)) * trajectory.dt

    write_json(
        "vacf/argon_vacf.json",
        [
            {"t": round(float(a), 2), "c": round(float(b / acf[0]), 5)}
            for a, b in zip(lag, acf)
        ],
    )

    # Running Green-Kubo integral: D(t) = 1/3 \int_0^t C(s) ds.
    running = (
        np.concatenate([[0.0], np.cumsum(0.5 * (acf[1:] + acf[:-1]) * np.diff(lag))])
        / 3.0
    )
    write_json(
        "vacf/argon_running_diffusion.json",
        [
            {"t": round(float(a), 2), "D": round(float(b / 10.0), 8)}
            for a, b in zip(lag, running)
        ],
    )

    equipartition = 3.0 * KB * trajectory.temperature / 39.948 * 4.184e-4
    minimum = int(np.argmin(acf))
    return {
        "C0": float(acf[0]),
        "C0_expected_3kT_m": float(equipartition),
        "min_lag_fs": float(lag[minimum]),
        "min_normalized": float(acf[minimum] / acf[0]),
        "D_cm2_per_s": float(running[-1] / 10.0),
    }


def pair_survival(trajectory: Trajectory) -> dict[str, float]:
    """First-shell residence correlation for argon.

    Slow: the kernel walks every (i, j) pair at every lag, so this is minutes,
    not seconds. It is the only dataset here that is not near-instant.
    """
    n_frames = 2000
    positions = np.ascontiguousarray(trajectory.wrapped[:n_frames])
    box = np.tile(np.array([[trajectory.box_length] * 3]), (n_frames, 1))
    # r0 / r1 bracket the first minimum of g(r): a pair counts as bonded inside
    # r0 and is only considered broken once it leaves r1 (a Stillinger-Rahman
    # style buffer that stops rattling at the boundary from breaking pairs).
    rows: list[dict[str, float | str]] = []
    summary: dict[str, float] = {}
    for method in ("continuous", "intermittent"):
        result = Persist.pair_survival_tcf(
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
        summary[f"tau_ps_{method}"] = float(-1.0 / slope / 1000.0)
        summary[f"reached_1_over_e_{method}"] = float(normalized[-1] < np.exp(-1.0))
    write_json("persist/argon_survival.json", rows)
    return summary
