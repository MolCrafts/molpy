"""Shared fixtures for the molpy core benchmark suite.

These benches measure molpy's public ``core/`` surface — the thin Python facade
over the molrs Rust kernels (``Box``, ``Atomistic``, ``Frame``). They live under
``benchmarks/`` (not ``tests/``) so the normal ``pytest tests/`` run does not
pick them up; ``pytest benchmarks/`` (and the bench.yml workflow) runs them.

Run::

    pip install -e ".[dev]"          # pulls in pytest-benchmark
    pytest benchmarks/ --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs

import molpy as mp

# Regression sizing: one small representative size — these benches guard against
# a perf/behaviour regression, not measure peak throughput (see benchmarks/README
# / molrs REG_N=1000). Bump locally if you need a scaling sweep.
SIZES: list[int] = [1_000]
SIZE_IDS: list[str] = ["reg-1k"]

BOX_LEN: float = 10.0


@pytest.fixture(params=SIZES, ids=SIZE_IDS)
def n(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def points(n: int) -> np.ndarray:
    """N points spanning ``[-L, 2L]`` per axis so wrap/fractional has work to do."""
    rng = np.random.default_rng(0)
    return (rng.random((n, 3), dtype=np.float64) * 3.0 - 1.0) * BOX_LEN


def make_chain(n: int) -> "mp.Atomistic":
    """Linear carbon chain of ``n`` atoms (``n-1`` bonds)."""
    mol = mp.Atomistic()
    atoms = [mol.def_atom(element="C") for _ in range(n)]
    for i in range(n - 1):
        mol.def_bond(atoms[i], atoms[i + 1])
    return mol


# --------------------------------------------------------------------------- #
# molpy.compute regression-benchmark fixtures                                  #
# --------------------------------------------------------------------------- #
# These are shared by the ``benchmarks/compute/`` tier. Regression sizing: a
# SMALL fixed point cloud (~600 atoms) and a few trajectory frames — enough to
# exercise every ``molpy.compute`` kernel and catch a perf or structural
# regression, NOT to measure peak throughput. pytest-benchmark auto-calibrates
# rounds, so small inputs keep the whole suite fast.

CMP_N_ATOMS: int = 600
CMP_BOX_LEN: float = 12.0
CMP_CUTOFF: float = 3.0


def random_frame(
    n: int = CMP_N_ATOMS, box_len: float = CMP_BOX_LEN, seed: int = 0
) -> "molrs.Frame":
    """A cubic periodic frame of ``n`` uniformly random atoms."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = molrs.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.simbox = mp.Box.cubic(box_len)
    return frame


@pytest.fixture
def cmp_frame() -> "molrs.Frame":
    """One 600-atom periodic frame."""
    return random_frame()


@pytest.fixture
def cmp_frames() -> list["molrs.Frame"]:
    """Three independent 600-atom periodic frames."""
    return [random_frame(seed=i) for i in range(3)]


@pytest.fixture
def cmp_nlist(cmp_frame: "molrs.Frame"):
    """Neighbor list over ``cmp_frame`` at the shared cutoff."""
    from molpy.compute import NeighborList

    return NeighborList(cutoff=CMP_CUTOFF)(cmp_frame)


@pytest.fixture
def cmp_frames_nlists(cmp_frames: list["molrs.Frame"]):
    """``(frames, nlists)`` for the multi-frame accumulating ops (RDF, ...)."""
    from molpy.compute import NeighborList

    nl = NeighborList(cutoff=CMP_CUTOFF)
    return cmp_frames, [nl(f) for f in cmp_frames]


def _drift_trajectory(
    n_frames: int = 13,
    box_len: float = 10.0,
    velocity: float = 1.0,
    with_velocities: bool = False,
):
    """Cation (type 1) drifting +velocity/frame in x (wrapped); anion (type 2) fixed."""
    from molpy.core.trajectory import Trajectory

    frames = []
    for i in range(n_frames):
        xc = (i * velocity) % box_len
        cols = {
            "x": np.array([xc, 0.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        if with_velocities:
            cols["vx"] = np.array([velocity, 0.0])
            cols["vy"] = np.array([0.0, 0.0])
            cols["vz"] = np.array([0.0, 0.0])
        frame = molrs.Frame()
        frame["atoms"] = cols
        frame.simbox = mp.Box.cubic(box_len)
        frames.append(frame)
    return Trajectory(frames)


@pytest.fixture
def drift_traj():
    """Two-species drift Trajectory for MCD / PMSD / Onsager."""
    return _drift_trajectory(velocity=1.0)


@pytest.fixture
def current_traj():
    """Two-species Trajectory carrying velocities for JACF."""
    return _drift_trajectory(velocity=1.0, with_velocities=True)


@pytest.fixture
def pair_traj():
    """A permanently bonded cation-anion pair Trajectory for Persist."""
    from molpy.core.trajectory import Trajectory

    frames = []
    for _ in range(6):
        frame = molrs.Frame()
        frame["atoms"] = {
            "x": np.array([0.0, 0.5]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        frame.simbox = mp.Box.cubic(100.0)
        frames.append(frame)
    return Trajectory(frames)


@pytest.fixture
def pos_traj() -> list["molrs.Frame"]:
    """A few frames of ~300 drifting particles for MSD / VanHove / reorientation."""
    rng = np.random.default_rng(3)
    n, box_len = 300, 30.0
    base = rng.uniform(0.0, box_len, size=(n, 3))
    frames = []
    for i in range(8):
        xyz = base + i * 0.2
        frame = molrs.Frame()
        frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        frame.simbox = mp.Box.cubic(box_len)
        frames.append(frame)
    return frames


@pytest.fixture
def charge_traj() -> list["molrs.Frame"]:
    """20 frames of 8 charged atoms for the dielectric-susceptibility route."""
    rng = np.random.default_rng(5)
    n, box_len = 8, 10.0
    frames = []
    for i in range(20):
        xyz = rng.random((n, 3)) + i * 0.1
        frame = molrs.Frame()
        frame["atoms"] = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "charge": np.ones(n) * 0.5,
        }
        frame.simbox = mp.Box.cubic(box_len)
        frames.append(frame)
    return frames


@pytest.fixture
def ion_traj() -> list["molrs.Frame"]:
    """A drifting +/- ion pair over 40 frames for IonicConductivity."""
    frames = []
    for i in range(40):
        frame = molrs.Frame()
        frame["atoms"] = {
            "x": np.array([1.0 + 0.01 * i, 5.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "charge": np.array([1.0, -1.0]),
        }
        frame.simbox = mp.Box.cubic(30.0)
        frames.append(frame)
    return frames


@pytest.fixture
def raw_acf() -> np.ndarray:
    """A small raw autocorrelation curve (1-D) for the spectra transforms."""
    from molpy.compute import compute_acf

    rng = np.random.default_rng(7)
    velocities = rng.standard_normal((40, 100, 3))
    return compute_acf(velocities, cache_size=32)


@pytest.fixture
def voronoi_inputs():
    """``(positions, radii, box)`` for the radical-Voronoi tessellation."""
    rng = np.random.default_rng(1)
    n = 300
    positions = rng.uniform(0.0, CMP_BOX_LEN, size=(n, 3))
    radii = np.full(n, 1.0)
    return positions, radii, mp.Box.cubic(CMP_BOX_LEN)


@pytest.fixture
def descriptor_rows():
    """200 eight-dimensional descriptor rows for the PCA / k-means ML ops."""
    from molpy.compute import DescriptorRow

    rng = np.random.default_rng(9)
    return [DescriptorRow(rng.random(8)) for _ in range(200)]
