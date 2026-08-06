"""Datasets for the aggregate pages: Cluster, Shape, Decomposition.

The cluster dataset exists to show that the neighbour cutoff *is* the definition
of an aggregate: a 0.4 A change across the width of the first g(r) peak turns
fifty small clusters into one percolating network.
"""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.compute import (
    CenterOfMass,
    Cluster,
    DescriptorRow,
    KMeans,
    NeighborList,
    Pca,
    RadiusOfGyration,
    Steinhardt,
)

from .lj import Trajectory
from .order import _fcc_frame
from .structure import _frames, write_json


def percolation(trajectory: Trajectory) -> dict[str, float]:
    """Cluster count and largest-cluster fraction against the cutoff."""
    frames = _frames(trajectory, stride=600)
    n_atoms = frames[0]["atoms"].nrows

    rows: list[dict[str, float | str]] = []
    transition: list[float] = []
    for cutoff in np.arange(3.2, 5.61, 0.05):
        counts, fractions = [], []
        for frame in frames:
            result = Cluster(min_cluster_size=2)(
                [frame], [NeighborList(cutoff=float(cutoff))(frame)]
            )[0]
            sizes = np.asarray(result.cluster_sizes)
            counts.append(result.num_clusters)
            fractions.append((sizes.max() if len(sizes) else 0) / n_atoms)
        fraction = float(np.mean(fractions))
        rows.append(
            {
                "cutoff": round(float(cutoff), 3),
                "value": round(float(np.mean(counts)), 2),
                "series": "N_clusters",
            }
        )
        rows.append(
            {
                "cutoff": round(float(cutoff), 3),
                "value": round(fraction * 100.0, 2),
                "series": "largest %",
            }
        )
        if 0.05 < fraction < 0.95:
            transition.append(float(cutoff))
    write_json("cluster/argon_percolation.json", rows)
    return {
        "transition_low": min(transition) if transition else float("nan"),
        "transition_high": max(transition) if transition else float("nan"),
    }


def chain_gyration(_trajectory: Trajectory) -> dict[str, float]:
    """Radius of gyration against chain length for ideal random-walk chains.

    The exact result for a freely jointed chain is Rg^2 = N b^2 / 6, so this
    figure has an analytic curve to be checked against rather than fitted to.
    """
    rng = np.random.default_rng(0)
    bond = 1.0
    n_chains = 200
    lengths = [10, 20, 40, 80, 160, 320]

    rows: list[dict[str, float | str]] = []
    ratios: list[float] = []
    for n_beads in lengths:
        measured = []
        for _ in range(n_chains):
            steps = rng.normal(0.0, bond / np.sqrt(3.0), size=(n_beads, 3))
            positions = np.cumsum(steps, axis=0)
            positions -= positions.mean(axis=0)
            positions += 200.0
            frame = mp.Frame()
            frame["atoms"] = {
                "x": positions[:, 0],
                "y": positions[:, 1],
                "z": positions[:, 2],
            }
            frame.box = mp.Box.cubic(400.0)
            clusters = Cluster(min_cluster_size=5)(
                [frame], [NeighborList(cutoff=2.5)(frame)]
            )
            masses = np.full(n_beads, 1.0)
            com = CenterOfMass(masses)([frame], clusters)
            radii = np.asarray(RadiusOfGyration(masses)([frame], clusters, com)[0])
            if radii.size:
                measured.append(float(radii.max()))
        squared = np.asarray(measured) ** 2
        # The formula predicts <Rg^2>, so the comparable measurement is
        # sqrt(mean(Rg^2)) -- NOT mean(Rg). By Jensen's inequality the latter
        # sits a few percent lower, which is a real effect and not an error.
        root_mean_square = float(np.sqrt(squared.mean()))
        mean_rg = float(np.mean(measured))
        analytic = float(np.sqrt(n_beads * bond**2 / 6.0))
        ratios.append(root_mean_square / analytic)
        rows.append(
            {"n": n_beads, "rg": round(root_mean_square, 4), "series": "√⟨R²⟩"}
        )
        rows.append({"n": n_beads, "rg": round(mean_rg, 4), "series": "⟨R⟩"})
        rows.append({"n": n_beads, "rg": round(analytic, 4), "series": "exact"})
    write_json("shape/ideal_chain_rg.json", rows)

    picked = [r for r in rows if r["series"] == "√⟨R²⟩"]
    log_n = np.log([r["n"] for r in picked])
    log_rg = np.log([r["rg"] for r in picked])
    return {
        "flory_exponent": float(np.polyfit(log_n, log_rg, 1)[0]),
        "rms_ratio_min": float(np.min(ratios)),
        "rms_ratio_max": float(np.max(ratios)),
    }


def descriptor_map(trajectory: Trajectory) -> dict[str, float]:
    """PCA + k-means on order descriptors, crystal against liquid."""
    crystal = _fcc_frame()
    liquid_frames = _frames(trajectory, stride=750)

    def descriptors(frame: mp.Frame, cutoff: float) -> np.ndarray:
        nlist = NeighborList(cutoff=cutoff)(frame)
        ql = np.asarray(Steinhardt(l=[4, 6])([frame], [nlist])[0]["ql"])
        coordination = np.full(ql.shape[1], 2.0 * nlist.n_pairs / ql.shape[1])
        return np.column_stack([ql[0], ql[1], coordination])

    crystal_rows = descriptors(crystal, 4.5)
    liquid_rows = np.vstack([descriptors(f, 5.4) for f in liquid_frames])
    # Subsample so the two phases contribute equally to the principal axes.
    rng = np.random.default_rng(0)
    take = min(len(crystal_rows), len(liquid_rows), 400)
    crystal_rows = crystal_rows[rng.choice(len(crystal_rows), take, replace=False)]
    liquid_rows = liquid_rows[rng.choice(len(liquid_rows), take, replace=False)]

    matrix = np.vstack([crystal_rows, liquid_rows])
    truth = np.array([0] * take + [1] * take)
    # Standardize: PCA on raw columns would be dominated by coordination number.
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

    projected = Pca()([DescriptorRow(row) for row in matrix])
    coords = np.asarray(projected.coords)
    labels = np.asarray(KMeans(k=2, max_iter=100, seed=0)(projected).labels)

    rows: list[dict[str, float | str]] = [
        {
            "pc1": round(float(coords[i, 0]), 4),
            "pc2": round(float(coords[i, 1]), 4),
            "phase": "FCC" if truth[i] == 0 else "liquid",
        }
        for i in range(len(coords))
    ]
    write_json("decomposition/phase_map.json", rows)

    agreement = max((labels == truth).mean(), (labels != truth).mean())
    return {
        "variance_pc1": float(np.asarray(projected.variance)[0]),
        "variance_pc2": float(np.asarray(projected.variance)[1]),
        "kmeans_agreement": float(agreement),
    }
