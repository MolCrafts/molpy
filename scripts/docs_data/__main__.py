"""Regenerate the datasets behind the documentation figures.

    python -m docs_data            # all datasets
    python -m docs_data structure  # one group

Run from the repository root with ``scripts/`` on the path::

    PYTHONPATH=scripts python -m docs_data
"""

from __future__ import annotations

import sys

from . import aggregate, angles, dynamics, order, structure, transport
from .run import argon_trajectory

GROUPS = {
    "structure": (
        structure.radial_distribution,
        structure.neighbor_cost,
        structure.local_density,
        structure.structure_factor,
    ),
    "transport": (
        transport.mean_squared_displacement,
        transport.velocity_autocorrelation,
        transport.pair_survival,
    ),
    "order": (order.steinhardt_contrast, order.bond_order_diagram),
    "angles": (angles.solid_angle_jacobian,),
    "aggregate": (
        aggregate.percolation,
        aggregate.chain_gyration,
        aggregate.descriptor_map,
    ),
    "dynamics": (
        dynamics.van_hove,
        dynamics.voronoi_volumes,
        dynamics.vibrational_dos,
        dynamics.debye_reference,
        dynamics.rotational_relaxation,
    ),
}


def main(argv: list[str]) -> int:
    requested = argv[1:] or list(GROUPS)
    unknown = [name for name in requested if name not in GROUPS]
    if unknown:
        print(f"unknown group(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(GROUPS)}", file=sys.stderr)
        return 2

    trajectory = argon_trajectory()
    print(
        f"argon: {trajectory.wrapped.shape[0]} frames, "
        f"T = {trajectory.temperature:.1f} K, "
        f"energy drift = {trajectory.energy_drift:.1e}"
    )
    for name in requested:
        for builder in GROUPS[name]:
            summary = builder(trajectory)
            reported = ", ".join(f"{k}={v:.4g}" for k, v in summary.items())
            print(f"  {builder.__name__}: {reported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
