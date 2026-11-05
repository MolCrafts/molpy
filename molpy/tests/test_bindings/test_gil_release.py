"""Long native work releases the GIL, so other Python threads keep running.

A molab workflow runs independent tasks on worker threads; a minimisation that
held the GIL for its whole run serialised them (a 2 s task took 63 s beside a
30 s one).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import molpy as mp


def _alkane(n_carbons: int) -> tuple[object, object]:
    mol = mp.io.smiles.read_smiles("C" * n_carbons).to_atomistic()
    mol, _ = mp.conformer.Conformer(seed=3).generate(mol)
    typifier = mp.ff.typifier.Mmff94sTypifier()
    frame = typifier.typify(mol).to_frame()
    rng = np.random.default_rng(0)
    for axis in ("x", "y", "z"):  # start away from the minimum so it works for a while
        frame["atoms"][axis] = np.asarray(frame["atoms"][axis]) + rng.normal(
            0.0, 0.3, frame["atoms"].n_rows
        )
    return mp.ff.PotentialCompiler(typifier.forcefield()).compile(frame), frame


def _runs_beside_python(work) -> None:
    """``work`` runs on a worker thread while this one keeps ticking.

    Holding the GIL freezes this thread for the whole native call; releasing
    it only lets the OS deschedule it for a few milliseconds. So the witness
    is the longest stall between ticks, not the tick rate: a parallel native
    kernel may take most of the CPU on a small CI runner and slow the
    counter without holding the GIL.
    """
    done = threading.Event()
    failure: list[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as exc:  # re-raised on the main thread below
            failure.append(exc)
        done.set()

    worker = threading.Thread(target=run)
    t0 = time.perf_counter()
    worker.start()
    last, longest_stall = t0, 0.0
    while not done.is_set():
        now = time.perf_counter()
        longest_stall = max(longest_stall, now - last)
        last = now
    elapsed = time.perf_counter() - t0
    worker.join()
    if failure:
        raise failure[0]
    if elapsed < 0.05:
        # A machine fast enough to finish inside OS scheduling noise cannot
        # tell a held GIL from a released one; that is not a failure.
        pytest.skip(
            f"native call took {elapsed * 1e3:.1f} ms: too fast to measure here"
        )
    assert longest_stall < 0.5 * elapsed, (longest_stall, elapsed)


def _lattice(n_side: int, spacing: float) -> np.ndarray:
    grid = np.arange(n_side) * spacing
    return (
        np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T.copy()
    )


def test_minimize_lets_other_python_threads_run() -> None:
    potentials, frame = _alkane(40)
    _runs_beside_python(
        lambda: mp.optimize.Lbfgs(potentials, fmax=1e-4, max_steps=20000).minimize(
            frame
        )
    )


def test_a_native_md_run_lets_other_python_threads_run() -> None:
    n_side, spacing = 8, 1.12
    pos = _lattice(n_side, spacing)
    atoms = mp.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d]))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(len(pos)))
    atoms.insert("mass", np.ones(len(pos)))
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame.box = mp.core.Box(mp.core.Cell.lengths([n_side * spacing] * 3))

    def run() -> None:  # an MD stays on the thread that made it
        sim = mp.md.MD("real")
        sim.set_integrator(mp.md.VelocityVerlet(0.001))
        sim.set_neighbor(mp.core.NeighborList(3.0), skin=0.5)
        sim.set_evaluator(mp.ff.potential.PairLjCut(1.0, 1.0, 2.5))
        sim.run(frame, 200)

    _runs_beside_python(run)


def test_a_compute_lets_other_python_threads_run() -> None:
    points = np.random.default_rng(0).uniform(0.0, 30.0, (1500, 3))
    debye = mp.compute.StaticStructureFactorDebye(list(np.linspace(0.5, 5.0, 40)))
    _runs_beside_python(lambda: debye.compute(positions=points))


def test_a_neighbor_list_build_lets_other_python_threads_run() -> None:
    points = np.random.default_rng(1).uniform(0.0, 60.0, (300_000, 3))
    simbox = mp.core.Box(mp.core.Cell.lengths([60.0, 60.0, 60.0]))
    nl = mp.core.NeighborList(2.5)
    _runs_beside_python(lambda: nl.build(points, simbox))


def test_reading_and_writing_let_other_python_threads_run() -> None:
    pos = _lattice(80, 1.5)
    atoms = mp.core.Block()
    atoms.insert("element", ["Ar"] * len(pos))
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d]))
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    sink = bytearray()
    _runs_beside_python(lambda: mp.io.write_xyz(sink, frame))
    text = bytes(sink)
    _runs_beside_python(lambda: mp.io.read_xyz(text))


def test_typing_lets_other_python_threads_run() -> None:
    mol = mp.io.smiles.read_smiles("C" * 240).to_atomistic()
    mol = mp.perceive.add_hydrogens(mol)
    typifier = mp.ff.typifier.OplsAaTypifier()
    _runs_beside_python(lambda: typifier.typify(mol))
