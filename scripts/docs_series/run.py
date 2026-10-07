"""Run the argon trajectories the docs figures use.

One run serves the structural figures: structure needs decorrelated
configurations, VACF needs dense time resolution, and sampling every step of a
30 ps constant-energy run covers both. The transport coefficients are not taken
from one run: Lennard-Jones dynamics is chaotic, so one 30 ps trajectory of 500
atoms is one draw of a noisy estimator (its long-lag MSD and its fitted D move
by tens of percent from seed to seed). The transport pages average over an
ensemble of independent runs, one per seed in :data:`SEEDS`, and quote the
spread.

Every run is a pure function of its seed: nothing is read from or written to a
cache on disk, so a regenerated figure is never built from a stale trajectory.
Only the small derived JSON files under ``docs/series/`` are committed.
"""

from __future__ import annotations

import functools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .lj import LennardJonesMd, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SERIES = REPO_ROOT / "docs" / "series"

#: State point: liquid argon just above the triple point (Rahman 1964).
TEMPERATURE = 85.0
MASS_DENSITY = 1.374
N_ATOMS = 500

#: 10 fs is a conventional argon timestep; 30 ps reaches the diffusive regime.
TIMESTEP = 10.0
PRODUCTION_STEPS = 3000

#: The seed of the single reference run the structural figures use.
REFERENCE_SEED = 0

#: The independent runs the transport coefficients are averaged over.
SEEDS = tuple(range(8))


@functools.cache
def argon_trajectory(seed: int = REFERENCE_SEED) -> Trajectory:
    """Melt, cool and sample one argon run; the seed draws its velocities."""
    md = LennardJonesMd(n_atoms=N_ATOMS, mass_density=MASS_DENSITY, seed=seed)
    # Melt the FCC starting lattice before cooling: at the triple point a
    # perfect crystal can stay metastable and the "liquid" would be a solid.
    md.thermalize(300.0)
    md.equilibrate(300.0, steps=400, dt=TIMESTEP)
    md.equilibrate(TEMPERATURE, steps=800, dt=TIMESTEP)
    return md.sample(steps=PRODUCTION_STEPS, stride=1, dt=TIMESTEP)


@functools.cache
def argon_ensemble() -> tuple[Trajectory, ...]:
    """One run per seed in :data:`SEEDS`, in seed order, run in parallel."""
    workers = min(len(SEEDS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(argon_trajectory, SEEDS))
