"""Datasets for the order pages.

The Steinhardt parameters of an ideal FCC lattice are exactly known
(q4 = 0.19094, q6 = 0.57452), which makes the crystal half of this figure a
validation of the kernel as well as a teaching contrast.
"""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.compute import NeighborList, Steinhardt

from .lj import Trajectory
from .structure import _frames, write_json

#: Ideal-lattice reference values, Steinhardt et al. (1983).
FCC_REFERENCE = {"q4": 0.190941, "q6": 0.574524}


def _fcc_frame(cells: int = 5, lattice: float = 5.26) -> mp.Frame:
    basis = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    )
    xyz = np.array(
        [
            (np.array([i, j, k]) + b) * lattice
            for i in range(cells)
            for j in range(cells)
            for k in range(cells)
            for b in basis
        ]
    )
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cubic(cells * lattice)
    return frame


def steinhardt_contrast(trajectory: Trajectory) -> dict[str, float]:
    """q6 distributions for a perfect FCC crystal and for liquid argon."""
    crystal = _fcc_frame()
    crystal_q = np.asarray(
        Steinhardt(l=[4, 6])([crystal], [NeighborList(cutoff=4.5)(crystal)])[0]["ql"]
    )

    frames = _frames(trajectory, stride=50)
    # 5.4 A is the first minimum of g(r): the defensible "first shell" cutoff.
    liquid = Steinhardt(l=[4, 6])(frames, [NeighborList(cutoff=5.4)(f) for f in frames])
    liquid_q4 = np.concatenate([np.asarray(r["ql"])[0] for r in liquid])
    liquid_q6 = np.concatenate([np.asarray(r["ql"])[1] for r in liquid])

    rows: list[dict[str, float | str]] = []
    for label, values in (
        ("FCC", crystal_q[1]),
        ("liquid", liquid_q6),
    ):
        counts, edges = np.histogram(values, bins=60, range=(0.0, 0.7), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        rows.extend(
            {"q6": round(float(c), 4), "p": round(float(v), 3), "phase": label}
            for c, v in zip(centers, counts)
        )
    write_json("order/steinhardt_q6.json", rows)

    return {
        "fcc_q4": float(crystal_q[0].mean()),
        "fcc_q6": float(crystal_q[1].mean()),
        "fcc_q4_reference": FCC_REFERENCE["q4"],
        "fcc_q6_reference": FCC_REFERENCE["q6"],
        "liquid_q4": float(liquid_q4.mean()),
        "liquid_q6": float(liquid_q6.mean()),
        "liquid_q6_std": float(liquid_q6.std()),
    }


def bond_order_diagram(trajectory: Trajectory) -> dict[str, float]:
    """Bond directions on the (theta, phi) sphere: FCC against the liquid.

    A perfect lattice puts every bond into a handful of discrete directions, so
    the map is emitted as the occupied cells only — that sparsity *is* the
    result, and a dense grid of zeros would hide it.
    """
    from molpy.compute import BondOrder

    crystal = _fcc_frame(cells=4)
    counts, _, theta_edges, phi_edges = BondOrder(n_theta=36, n_phi=72)(
        [crystal], [NeighborList(cutoff=4.5)(crystal)]
    )[0]
    counts = np.asarray(counts)
    theta = 0.5 * (np.asarray(theta_edges)[:-1] + np.asarray(theta_edges)[1:])
    phi = 0.5 * (np.asarray(phi_edges)[:-1] + np.asarray(phi_edges)[1:])

    rows: list[dict[str, float | str]] = []
    for i, j in np.argwhere(counts > 0):
        rows.append(
            {
                "theta": round(float(np.degrees(theta[i])), 2),
                "phi": round(float(np.degrees(phi[j])), 2),
                "n": int(counts[i, j]),
            }
        )
    write_json("environment/fcc_bond_order.json", rows)

    frames = _frames(trajectory, stride=1000)
    liquid = np.sum(
        [
            np.asarray(r[0])
            for r in BondOrder(n_theta=36, n_phi=72)(
                frames, [NeighborList(cutoff=5.4)(f) for f in frames]
            )
        ],
        axis=0,
    )
    return {
        "fcc_bonds": float(counts.sum()),
        "fcc_occupied": float((counts > 0).sum()),
        "n_cells": float(counts.size),
        "liquid_occupied": float((liquid > 0).sum()),
    }
