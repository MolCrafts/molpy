"""Long native work releases the GIL, so other Python threads keep running.

A molab workflow runs independent tasks on worker threads; a minimisation that
held the GIL for its whole run serialised them (a 2 s task took 63 s beside a
30 s one).
"""

from __future__ import annotations

import threading
import time

import numpy as np

import molpy as mp


def _loop_rate(seconds: float) -> float:
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        n += 1
    return n / seconds


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


def test_minimize_lets_other_python_threads_run() -> None:
    potentials, frame = _alkane(40)
    free = _loop_rate(0.3)
    done = threading.Event()

    def work() -> None:
        mp.optimize.Lbfgs(potentials, fmax=1e-4, max_steps=20000).minimize(frame)
        done.set()

    worker = threading.Thread(target=work)
    t0 = time.perf_counter()
    worker.start()
    ticks = 0
    while not done.is_set():
        ticks += 1
    elapsed = time.perf_counter() - t0
    worker.join()
    assert elapsed > 0.05, "the minimisation finished too fast to measure"
    assert ticks / elapsed > 0.25 * free, (ticks / elapsed, free)
