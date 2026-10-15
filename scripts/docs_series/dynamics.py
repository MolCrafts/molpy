"""Datasets for the dynamics and tessellation pages.

Van Hove, Voronoi, and the vibrational density of states, all from the same
argon trajectory as the rest of the figures.
"""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.compute import (
    Axis,
    RadicalVoronoi,
    ReorientationAcf,
    Spectrum,
    Vacf,
    VanHove,
)

from .lj import FS_TO_PS, FS_TO_S, SPEED_OF_LIGHT_CM_PER_S, Trajectory
from .structure import write_json

#: VACF lags transformed into the VDOS, in frames.
VDOS_MAX_LAG = 400


def van_hove(trajectory: Trajectory) -> dict[str, float]:
    """Self part of G(r, t) at a spread of lags."""
    positions = trajectory.unwrapped[:1500]
    box = mp.Box.cube(trajectory.box_length)
    lags = [10, 50, 200, 600]
    result = VanHove(Axis(0.0, 12.0, 120), lags, stride=20).compute(
        positions=positions, boxes=[box] * len(positions)
    )

    r = np.asarray(result.r_centers)
    g_self = np.asarray(result.g_self)
    rows: list[dict[str, float | str]] = []
    for index, lag in enumerate(lags):
        picoseconds = lag * trajectory.dt * FS_TO_PS
        # Compact legend labels so four lags fit on one bottom row.
        label = f"{picoseconds:g} ps"
        rows.extend(
            {"r": round(float(a), 3), "g": round(float(b), 5), "lag": label}
            for a, b in zip(r, g_self[index])
        )
    write_json("van_hove/argon_self.json", rows)

    # The self part is a probability density in 4*pi*r^2 dr; its first moment
    # is the mean displacement, which must grow with lag.
    means = [float(np.sum(r * g_self[i]) / np.sum(g_self[i])) for i in range(len(lags))]
    return {f"mean_r_lag{lag}": m for lag, m in zip(lags, means)}


def voronoi_volumes(trajectory: Trajectory) -> dict[str, float]:
    """Distribution of Voronoi cell volumes in the liquid."""
    box = mp.Box.cube(trajectory.box_length)
    volumes: list[np.ndarray] = []
    for xyz in trajectory.wrapped[::100]:
        cells = RadicalVoronoi().compute(
            positions=xyz, simbox=box, radii=np.zeros(len(xyz))
        )
        volumes.append(np.asarray(cells.volumes))
    stacked = np.concatenate(volumes)

    counts, edges = np.histogram(stacked, bins=50, range=(30.0, 70.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    write_json(
        "voronoi/argon_volumes.json",
        [
            {"v": round(float(c), 3), "p": round(float(v), 5)}
            for c, v in zip(centers, counts)
        ],
    )
    n_atoms = trajectory.wrapped.shape[1]
    return {
        "mean_volume": float(stacked.mean()),
        "expected_V_over_N": float(trajectory.box_length**3 / n_atoms),
        "std_volume": float(stacked.std()),
        "tiling_error": float(
            abs(volumes[0].sum() - trajectory.box_length**3) / trajectory.box_length**3
        ),
    }


def vibrational_dos(trajectory: Trajectory) -> dict[str, float]:
    """VDOS of liquid argon: the Fourier transform of the VACF.

    The VACF is cosine-squared windowed and zero-padded to four times its
    length; ``Spectrum`` reports angular frequencies in rad/fs, converted
    here to wavenumbers.
    """
    n_frames = trajectory.velocities.shape[0]
    vacf = Vacf(VDOS_MAX_LAG).compute(
        velocities=trajectory.velocities.reshape(n_frames, -1), dt=trajectory.dt
    )
    spectrum = Spectrum("cosine_sq", 4).compute(
        signal=vacf.channel(0), dt=vacf.lag_times[1]
    )
    frequency = spectrum.frequencies / (2.0 * np.pi * FS_TO_S * SPEED_OF_LIGHT_CM_PER_S)
    intensity = spectrum.real

    keep = frequency <= 200.0
    normalized = intensity[keep] / intensity[keep].max()
    write_json(
        "spectra/argon_vdos.json",
        [
            {"nu": round(float(a), 2), "I": round(float(b), 5)}
            for a, b in zip(frequency[keep], normalized)
        ],
    )
    peak = int(np.argmax(intensity[keep]))
    correlation_time = VDOS_MAX_LAG * trajectory.dt * FS_TO_S
    return {
        "peak_cm1": float(frequency[keep][peak]),
        "zero_frequency_weight": float(normalized[0]),
        "resolution_cm1": float(1.0 / (SPEED_OF_LIGHT_CM_PER_S * correlation_time)),
    }


def debye_reference(_trajectory: Trajectory) -> dict[str, float]:
    """The closed-form Debye spectrum, evaluated rather than typed by hand.

    This is the only figure in the docs that is not a measurement: it is the
    analytic Debye form, plotted so a reader can recognise its shape. Writing it
    from the formula keeps it reproducible and keeps the parameters visible.
    """
    eps_zero, eps_inf, tau = 54.0, 1.0, 6.5
    omega = np.geomspace(1e-3, 10.0, 120)
    response = eps_inf + (eps_zero - eps_inf) / (1.0 + 1j * omega * tau)

    rows: list[dict[str, float | str]] = []
    for w, value in zip(omega, response):
        rows.append(
            {
                "omega": float(f"{w:.5g}"),
                "eps": round(float(value.real), 4),
                "part": "ε′",
            }
        )
        rows.append(
            {
                "omega": float(f"{w:.5g}"),
                "eps": round(float(-value.imag), 4),
                "part": "ε″",
            }
        )
    write_json("dielectric/debye_reference.json", rows)
    loss = -response.imag
    peak = int(np.argmax(loss))
    return {
        "eps_zero": eps_zero,
        "eps_inf": eps_inf,
        "tau": tau,
        "loss_peak_omega": float(omega[peak]),
        "omega_tau_at_peak": float(omega[peak] * tau),
    }


def rotational_relaxation(_trajectory: Trajectory) -> dict[str, float]:
    """C1 and C2 for rods undergoing rotational diffusion.

    Rotational diffusion has an exact answer, C_l(t) = exp(-l(l+1) D_r t), so
    tau_1 / tau_2 must be 3 whatever D_r is. That ratio is the validation.
    """
    rng = np.random.default_rng(0)
    n_rods, n_frames, angular_step = 300, 400, 0.06
    axis = rng.normal(size=(n_rods, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)

    axes = []
    for _ in range(n_frames):
        kick = rng.normal(0.0, angular_step, size=(n_rods, 3))
        kick -= (kick * axis).sum(axis=1, keepdims=True) * axis  # keep it a rotation
        axis = axis + kick
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        axes.append(axis)
    vectors = np.array(axes)

    # dt = 1: lag times in frames.
    c1_curve = ReorientationAcf(1, 200).compute(vectors=vectors, dt=1.0)
    c2_curve = ReorientationAcf(2, 200).compute(vectors=vectors, dt=1.0)
    lag = c1_curve.lag_times
    c1, c2 = c1_curve.channel(0), c2_curve.channel(0)

    rows: list[dict[str, float | str]] = []
    for label, curve in (("C₁(t)", c1), ("C₂(t)", c2)):
        rows.extend(
            {"t": round(float(a), 1), "c": round(float(b), 5), "order": label}
            for a, b in zip(lag, curve)
        )
    write_json("reorientation/rod_legendre.json", rows)

    def relaxation_time(curve: np.ndarray) -> float:
        keep = (curve > 0.05) & (lag > 0)
        return float(-1.0 / np.polyfit(lag[keep], np.log(curve[keep]), 1)[0])

    tau1, tau2 = relaxation_time(c1), relaxation_time(c2)
    return {"tau1_frames": tau1, "tau2_frames": tau2, "ratio": tau1 / tau2}
