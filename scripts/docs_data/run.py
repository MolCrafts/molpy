"""Produce (and cache) the reference argon trajectory the docs figures use.

One run serves every figure: structure needs decorrelated configurations,
MSD needs a long continuous path, VACF needs dense time resolution. Sampling
every step of a 30 ps constant-energy run covers all three.

The cache lives under ``.cache/`` (gitignored). Only the small derived JSON
files under ``docs/data/`` are committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .lj import LennardJonesMD, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / ".cache" / "docs_data"
DOCS_DATA = REPO_ROOT / "docs" / "data"

#: State point: liquid argon just above the triple point (Rahman 1964).
TEMPERATURE = 85.0
MASS_DENSITY = 1.374
N_ATOMS = 500

#: 10 fs is a conventional argon timestep; 30 ps reaches the diffusive regime.
TIMESTEP = 10.0
PRODUCTION_STEPS = 3000


def argon_trajectory(*, refresh: bool = False) -> Trajectory:
    """Return the reference trajectory, running the MD only when needed."""
    cache_file = CACHE / f"argon_{N_ATOMS}_{PRODUCTION_STEPS}.npz"
    if cache_file.is_file() and not refresh:
        stored = np.load(cache_file)
        return Trajectory(
            wrapped=stored["wrapped"],
            unwrapped=stored["unwrapped"],
            velocities=stored["velocities"],
            box_length=float(stored["box_length"]),
            dt=float(stored["dt"]),
            temperature=float(stored["temperature"]),
            energy_drift=float(stored["energy_drift"]),
        )

    md = LennardJonesMD(n_atoms=N_ATOMS, mass_density=MASS_DENSITY, seed=0)
    # Melt the FCC starting lattice before cooling: at the triple point a
    # perfect crystal can stay metastable and the "liquid" would be a solid.
    md.thermalize(300.0)
    md.equilibrate(300.0, steps=400, dt=TIMESTEP)
    md.equilibrate(TEMPERATURE, steps=800, dt=TIMESTEP)
    trajectory = md.sample(steps=PRODUCTION_STEPS, stride=1, dt=TIMESTEP)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        wrapped=trajectory.wrapped,
        unwrapped=trajectory.unwrapped,
        velocities=trajectory.velocities,
        box_length=trajectory.box_length,
        dt=trajectory.dt,
        temperature=trajectory.temperature,
        energy_drift=trajectory.energy_drift,
    )
    return trajectory
