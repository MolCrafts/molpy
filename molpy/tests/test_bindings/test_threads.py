"""Frame, Block and FrameMeta cross Python threads.

A worker pool (``concurrent.futures``, ``asyncio.to_thread``) builds a value on
one thread and hands it to another, which reads, writes, pickles and finally
drops it. Every step here runs on a thread other than the one that built the
object.
"""

import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import molpy as mp
import numpy as np
import pytest
from molpy.core import Block, Frame, MetaValue

N_ROWS = 64


@pytest.fixture
def unraisable():
    """Collect exceptions raised where Python cannot propagate them — a
    destructor running on the wrong thread reports there."""
    seen = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda record: seen.append(record.exc_value)
    yield seen
    sys.unraisablehook = previous


def on_worker(fn, *args):
    """Run ``fn`` on a fresh worker thread and return (or raise) its result."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args).result()


def build_frame() -> Frame:
    atoms = Block()
    atoms.insert("x", np.arange(N_ROWS, dtype=np.float64))
    atoms.insert("element", ["C"] * N_ROWS)
    frame = Frame()
    frame["atoms"] = atoms
    frame.meta = {"source": MetaValue("string", "worker")}
    frame.box = mp.core.Box(mp.core.Cell.lengths([10.0, 10.0, 10.0]))
    return frame


def test_a_frame_built_on_a_worker_is_read_written_pickled_and_dropped_on_another(
    unraisable,
):
    frame = on_worker(build_frame)

    assert frame["atoms"]["x"].tolist() == list(range(N_ROWS))
    assert frame.meta["source"] == "worker"
    frame["atoms"]["x"] = np.zeros(N_ROWS)
    frame.meta["step"] = MetaValue("u64", 3)
    restored = pickle.loads(pickle.dumps(frame))
    assert restored["atoms"]["x"].tolist() == [0.0] * N_ROWS
    assert restored.meta["step"] == 3
    assert restored.box is not None

    del frame, restored
    assert unraisable == []


def test_a_frame_built_here_is_written_and_dropped_on_a_worker(unraisable):
    frame = build_frame()

    def mutate_and_drop(f):
        f["atoms"]["x"] = np.full(N_ROWS, 2.0)
        f["bonds"] = Block({"i": np.array([0], dtype=np.uint64)})
        return f["atoms"]["x"].sum()

    assert on_worker(mutate_and_drop, frame) == 2.0 * N_ROWS
    # the worker's writes reached the frame this thread holds
    assert frame["atoms"]["x"].tolist() == [2.0] * N_ROWS
    assert frame["bonds"]["i"].tolist() == [0]

    on_worker(lambda f: pickle.loads(pickle.dumps(f)), frame)
    handoff = [frame]
    del frame
    on_worker(handoff.clear)
    assert unraisable == []


def test_a_block_and_frame_meta_cross_threads(unraisable):
    block = on_worker(lambda: Block({"q": np.array([1.0, -1.0])}))
    meta = on_worker(lambda: build_frame().meta)

    block["q"] = np.array([0.5, -0.5])
    assert block["q"].tolist() == [0.5, -0.5]
    assert pickle.loads(pickle.dumps(block))["q"].tolist() == [0.5, -0.5]
    assert meta["source"] == "worker"
    meta["source"] = MetaValue("string", "main")
    assert meta["source"] == "main"

    held = [block, meta]
    del block, meta
    on_worker(held.clear)
    assert unraisable == []


def test_two_threads_read_one_frame_concurrently(unraisable):
    frame = build_frame()
    expected = float(np.arange(N_ROWS).sum())
    barrier = threading.Barrier(2)

    def read_many(_):
        barrier.wait()
        sums = set()
        for _ in range(500):
            sums.add(float(frame["atoms"]["x"].sum()))
            assert frame["atoms"].n_rows == N_ROWS
            assert frame.box is not None
        return sums

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(read_many, range(2)))
    assert results == [{expected}, {expected}]
    assert unraisable == []


def test_a_numpy_view_taken_on_a_worker_outlives_the_frame(unraisable):
    def view_then_drop():
        frame = build_frame()
        view = frame["atoms"]["x"]
        del frame
        return view

    view = on_worker(view_then_drop)
    assert view.tolist() == list(range(N_ROWS))

    frame = build_frame()
    view = on_worker(lambda: frame["atoms"]["x"])
    # A column write replaces the column; the view keeps the array it was taken
    # over and the frame shows the new values — no write is lost.
    frame["atoms"]["x"] = np.full(N_ROWS, 7.0)
    assert view.tolist() == list(range(N_ROWS))
    assert frame["atoms"]["x"].tolist() == [7.0] * N_ROWS

    held = [view]
    del view
    on_worker(held.clear)
    assert unraisable == []
