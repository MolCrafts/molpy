"""Dataset for the Distribution page: the sin(theta) solid-angle Jacobian.

Three atoms whose two bond directions are independent and isotropic have an
angle distributed as sin(theta)/2 — not uniformly. That is pure geometry, it has
nothing to do with chemistry, and it is the first thing to divide out of any
measured angular distribution. Measuring it on constructed random directions
gives a figure whose correct answer is known in advance.
"""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.compute import AngleDistribution

from .lj import Trajectory
from .structure import write_json


def solid_angle_jacobian(_trajectory: Trajectory) -> dict[str, float]:
    """Angle distribution of isotropic random triplets, raw and sin-corrected."""
    rng = np.random.default_rng(0)
    n_triplets = 200_000

    def unit_vectors(count: int) -> np.ndarray:
        v = rng.normal(size=(count, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    middle = np.full((n_triplets, 3), 500.0)
    first = middle + unit_vectors(n_triplets) * rng.uniform(1.0, 3.0, (n_triplets, 1))
    third = middle + unit_vectors(n_triplets) * rng.uniform(1.0, 3.0, (n_triplets, 1))

    xyz = np.empty((3 * n_triplets, 3))
    xyz[0::3], xyz[1::3], xyz[2::3] = first, middle, third
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cubic(1000.0)
    index = np.arange(3 * n_triplets, dtype=np.uint32).reshape(n_triplets, 3)
    frame["angles"] = {
        "atomi": index[:, 0],
        "atomj": index[:, 1],
        "atomk": index[:, 2],
    }

    # The kernel bins radians; the constructor's 0..180 default is a degrees
    # range bolted onto it, so the range must be given explicitly as 0..pi.
    result = AngleDistribution(n_bins=90, min=0.0, max=float(np.pi))([frame])
    centers = np.asarray(result.bin_centers)
    density = np.asarray(result.density)
    corrected = np.asarray(result.density_sin_corrected)

    degrees = np.degrees(centers)
    rows: list[dict[str, float | str]] = []
    for angle, raw, flat in zip(degrees, density, corrected):
        rows.append(
            {
                "theta": round(float(angle), 2),
                "p": round(float(raw), 5),
                "series": "measured",
            }
        )
        rows.append(
            {
                "theta": round(float(angle), 2),
                "p": round(float(flat), 5),
                "series": "corr.",
            }
        )
    for angle, value in zip(degrees, np.sin(centers) / 2.0):
        rows.append(
            {
                "theta": round(float(angle), 2),
                "p": round(float(value), 5),
                "series": "½sinθ",
            }
        )
    write_json("distribution/solid_angle.json", rows)

    interior = slice(3, -3)
    return {
        "peak_degrees": float(degrees[int(np.argmax(density))]),
        "max_deviation_from_sin": float(
            np.max(np.abs(density[interior] / (np.sin(centers) / 2.0)[interior] - 1.0))
        ),
        "corrected_flatness": float(
            corrected[interior].std() / corrected[interior].mean()
        ),
    }
