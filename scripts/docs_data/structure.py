"""Datasets for the structural pages: RDF, NeighborList, Density, Diffraction.

Every number written here comes from running the documented MolPy compute on
the argon trajectory from :mod:`docs_data.run`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import molpy as mp
from molpy.compute import (
    LocalDensity,
    NeighborList,
    RDF,
    StaticStructureFactorDebye,
)

from .lj import Trajectory
from .run import DOCS_DATA

#: Structural averages do not need every step, but more (partly correlated)
#: configurations still shrink the noise in the g(r) tail.
STRUCTURE_STRIDE = 5


def write_json(relative: str, rows: list[dict[str, float | str]]) -> Path:
    """Write ``rows`` to ``docs/data/<relative>`` and return the path."""
    path = DOCS_DATA / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def _frames(trajectory: Trajectory, stride: int = STRUCTURE_STRIDE) -> list[mp.Frame]:
    """Wrap stored coordinates as MolPy frames with a periodic box."""
    frames = []
    for xyz in trajectory.wrapped[::stride]:
        frame = mp.Frame()
        frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        frame.box = mp.Box.cubic(trajectory.box_length)
        frames.append(frame)
    return frames


def radial_distribution(trajectory: Trajectory) -> dict[str, float]:
    """g(r) and the running coordination number n(R) for liquid argon."""
    frames = _frames(trajectory)
    # Stay just inside L/2: the last bins of a minimum-image histogram are
    # distorted by the corners of the periodic cell.
    r_max = np.floor(trajectory.box_length / 2.0)
    neighbors = [NeighborList(cutoff=r_max)(frame) for frame in frames]
    result = RDF(n_bins=int(r_max / 0.05), r_max=r_max)(frames, neighbors)

    r = np.asarray(result.bin_centers)
    g = np.asarray(result.rdf)
    write_json(
        "rdf/argon_gr.json",
        [{"r": round(float(a), 4), "g": round(float(b), 4)} for a, b in zip(r, g)],
    )

    density = frames[0]["atoms"].nrows / trajectory.box_length**3
    integrand = 4.0 * np.pi * density * r**2 * g
    coordination = np.concatenate([[0.0], np.cumsum(np.diff(r) * integrand[:-1])])
    write_json(
        "rdf/argon_coordination.json",
        [
            {"r": round(float(a), 4), "n": round(float(b), 4)}
            for a, b in zip(r, coordination)
        ],
    )

    peak = int(np.argmax(g))
    window = slice(peak, peak + int(2.0 / 0.05))
    minimum = peak + int(np.argmin(g[window]))
    return {
        "peak_r": float(r[peak]),
        "peak_g": float(g[peak]),
        "min_r": float(r[minimum]),
        "min_g": float(g[minimum]),
        "coordination_at_min": float(coordination[minimum]),
        "tail_g": float(g[-20:].mean()),
        "density": float(density),
    }


def neighbor_cost(trajectory: Trajectory) -> dict[str, float]:
    """Neighbours per particle against cutoff, and the ideal-gas estimate.

    Plotted as *neighbours* rather than *pairs*: the list stores each pair once,
    so ``2 * n_pairs / N`` is the quantity that (4/3) pi rho r^3 predicts.
    """
    frame = _frames(trajectory, stride=len(trajectory.wrapped))[0]
    n_atoms = frame["atoms"].nrows
    density = n_atoms / trajectory.box_length**3

    rows: list[dict[str, float | str]] = []
    ratios: list[float] = []
    for cutoff in np.arange(3.0, trajectory.box_length / 2.0 + 0.01, 0.5):
        measured = 2.0 * NeighborList(cutoff=float(cutoff))(frame).n_pairs / n_atoms
        ideal = density * 4.0 / 3.0 * np.pi * cutoff**3
        ratios.append(float(measured / ideal))
        rows.append(
            {
                "cutoff": round(float(cutoff), 2),
                "neighbours": round(float(measured), 3),
                "series": "measured",
            }
        )
        rows.append(
            {
                "cutoff": round(float(cutoff), 2),
                "neighbours": round(float(ideal), 3),
                "series": "ideal gas",
            }
        )
    write_json("neighborlist/pair_scaling.json", rows)
    return {
        "density": float(density),
        "n_atoms": float(n_atoms),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "ratio_last": ratios[-1],
    }


def local_density(trajectory: Trajectory) -> dict[str, float]:
    """Distribution of per-particle local density at two probe radii."""
    frames = _frames(trajectory)
    bulk = frames[0]["atoms"].nrows / trajectory.box_length**3

    rows: list[dict[str, float | str]] = []
    summary: dict[str, float] = {"bulk": float(bulk)}
    for probe in (4.0, 8.0):
        neighbors = [NeighborList(cutoff=probe)(frame) for frame in frames]
        per_frame = LocalDensity(r_max=probe)(frames, neighbors)
        values = np.concatenate([np.asarray(density) for _, density in per_frame])
        counts, edges = np.histogram(values, bins=40, range=(0.0, 0.045), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        label = f"r={probe:.0f} Å"
        rows.extend(
            {
                "density": round(float(c), 5),
                "p": round(float(v), 2),
                "probe": label,
            }
            for c, v in zip(centers, counts)
        )
        summary[f"mean_{probe:.0f}"] = float(values.mean())
        summary[f"std_{probe:.0f}"] = float(values.std())
    write_json("density/local_histogram.json", rows)
    return summary


def structure_factor(trajectory: Trajectory) -> dict[str, float]:
    """S(k) from the Debye equation on decorrelated configurations."""
    frames = _frames(trajectory, stride=60)
    # Start above the forward-scattering region: the Debye sum keeps its self
    # terms, so S(k -> 0) runs away to N and is not the compressibility limit.
    k = np.linspace(1.0, 8.0, 180)
    per_frame = StaticStructureFactorDebye(k)(frames)
    s_k = np.mean([np.asarray(s) for _, s, _ in per_frame], axis=0)
    write_json(
        "diffraction/argon_sk.json",
        [{"k": round(float(a), 4), "S": round(float(b), 4)} for a, b in zip(k, s_k)],
    )
    peak = int(np.argmax(s_k))
    return {"peak_k": float(k[peak]), "peak_S": float(s_k[peak])}
