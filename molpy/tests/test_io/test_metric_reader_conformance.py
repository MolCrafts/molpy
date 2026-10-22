"""Every metric reader honours a viewer's request the same way.

A viewer resumes a source by handing back ``since``, thins it with ``stride``
and bounds it with ``limit``, and it does so without knowing which reader is
on the other end. So the request must mean one thing across formats:

* ``since`` — the number of records of the reader's *raw* stream (before any
  filter or stride) to skip;
* ``stride`` — keep every *stride*-th of the remaining samples **per series**;
* ``limit`` — stop after this many records are yielded.

These tests hold every reader to that, against a reference written here from
the contract (independent of molpy's implementation), and check that
equivalent data in different formats comes back identical.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import molpy as mp

STEPS = list(range(0, 1100, 100))  # 11 samples, split over two LAMMPS runs
TEMPS = [300.0 + i for i in range(len(STEPS))]
PRESS = [1.0 + i / 10 for i in range(len(STEPS))]
TIMES = [s * 0.002 for s in STEPS]


class Request:
    """The sampling policy a viewer hands down. Matched by shape, not import."""

    def __init__(self, *, since=0, stride=1, limit=None, metric_type=None, keys=()):
        self.since = since
        self.stride = stride
        self.limit = limit
        self.metric_type = metric_type
        self.keys = keys

    def __repr__(self) -> str:
        return (
            f"Request(since={self.since}, stride={self.stride}, "
            f"limit={self.limit}, keys={self.keys})"
        )


REQUESTS = [
    Request(since=3, stride=2, limit=4),
    Request(since=5, stride=3, limit=100),
    Request(since=1, stride=4, limit=2),
    Request(since=7, stride=1, limit=5),
    Request(since=4, stride=2, limit=3, keys=("lammps/Press", "mrec/time")),
    Request(since=10_000, stride=2, limit=5),
]


def reference(
    request: Request, records: Iterable[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """The contract, spelled out once and independently of molpy."""
    if request.limit is not None and request.limit <= 0:
        return
    per_series: dict[Any, int] = {}
    kept = 0
    for index, record in enumerate(records):
        if index < request.since:
            continue
        if request.keys and record.get("k") not in request.keys:
            continue
        if request.metric_type is not None and record.get("t") != request.metric_type:
            continue
        position = per_series.get(record.get("k"), 0)
        per_series[record.get("k")] = position + 1
        if position % request.stride:
            continue
        yield record
        kept += 1
        if request.limit is not None and kept >= request.limit:
            return


def _thermo(rows: range) -> str:
    body = "".join(f"  {STEPS[i]:>6d}  {TEMPS[i]:.2f}  {PRESS[i]:.2f}\n" for i in rows)
    return (
        "Per MPI rank memory allocation (min/avg/max) = 3.5 | 3.5 | 3.5 Mbytes\n"
        "   Step   Temp   Press\n"
        f"{body}"
        "Loop time of 1.0 on 1 procs for 100 steps with 10 atoms\n"
    )


def write_lammps(tmp_path: Path) -> Path:
    path = tmp_path / "log.lammps"
    path.write_text(
        "LAMMPS (2 Aug 2023)\n" + _thermo(range(6)) + _thermo(range(6, 11)),
        encoding="utf-8",
    )
    return path


def write_mrec(tmp_path: Path) -> Path:
    frame = mp.Frame()
    frame["atoms"] = {"x": [0.0], "y": [0.0], "z": [0.0]}
    path = tmp_path / "run.mrec"
    mp.io.write_mrec(
        path,
        trajectory=mp.Trajectory(
            [frame] * len(STEPS), step=np.array(STEPS), time=np.array(TIMES)
        ),
    )
    return path


def write_mrec_zip(tmp_path: Path) -> Path:
    return Path(mp.io.mrec.pack_mrec_zip(str(write_mrec(tmp_path))))


def write_jsonl_mirror(tmp_path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """The same records a reader yields, as molexp would have logged them."""
    path = tmp_path / "metrics.mlp.jsonl"
    path.write_text(
        "".join(
            json.dumps({"t": r["t"], "k": r["k"], "s": r["s"], "v": r["v"]}) + "\n"
            for r in records
        ),
        encoding="utf-8",
    )
    return path


def write_jsonl(tmp_path: Path) -> Path:
    return write_jsonl_mirror(
        tmp_path,
        (
            {"t": "scalar", "k": key, "s": float(step), "v": value}
            for step, temp, press in zip(STEPS, TEMPS, PRESS, strict=True)
            for key, value in (("lammps/Temp", temp), ("lammps/Press", press))
        ),
    )


READERS = {
    "lammps_log": (mp.io.lammps.LammpsLogMetricReader(), write_lammps),
    "mrec": (mp.io.mrec.MrecMetricReader(), write_mrec),
    "mrec_zip": (mp.io.mrec.MrecMetricReader(), write_mrec_zip),
    "mlp_jsonl": (mp.io.mlp_jsonl.MlpJsonlMetricReader(), write_jsonl),
}


def plotted(records: Iterable[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
    """What a chart draws from a record."""
    return [(r["k"], r["s"], r["v"]) for r in records]


@pytest.mark.parametrize("name", list(READERS))
@pytest.mark.parametrize("request_", REQUESTS, ids=repr)
def test_every_reader_honours_the_request_by_the_contract(
    tmp_path: Path, name: str, request_: Request
):
    reader, write = READERS[name]
    path = write(tmp_path)
    assert reader.sniff(path)
    everything = list(reader.read(path, source="src"))
    assert len(everything) == 22
    got = list(reader.read(path, source="src", request=request_))
    assert got == list(reference(request_, everything))


@pytest.mark.parametrize("name", list(READERS))
def test_resuming_with_since_never_repeats_or_drops_a_record(tmp_path: Path, name):
    """Paging with ``limit`` and resuming at the raw index rebuilds the stream."""
    reader, write = READERS[name]
    path = write(tmp_path)
    everything = list(reader.read(path, source="src"))
    pages: list[dict[str, Any]] = []
    for since in range(0, len(everything), 5):
        pages.extend(
            reader.read(path, source="src", request=Request(since=since, limit=5))
        )
    assert pages == everything


@pytest.mark.parametrize("request_", REQUESTS, ids=repr)
@pytest.mark.parametrize("name", ["lammps_log", "mrec"])
def test_equivalent_data_reads_the_same_in_every_format(
    tmp_path: Path, name: str, request_: Request
):
    """A run logged by molexp and the same run read from its solver's file
    must chart identically under the same request."""
    reader, write = READERS[name]
    native = write(tmp_path)
    mirror = write_jsonl_mirror(tmp_path, reader.read(native, source="src"))
    from_native = reader.read(native, source="src", request=request_)
    from_mirror = mp.io.mlp_jsonl.MlpJsonlMetricReader().read(
        mirror, source="src", request=request_
    )
    assert plotted(from_native) == plotted(from_mirror)


def test_since_is_a_raw_index_not_a_strided_one(tmp_path: Path):
    """``since=3`` skips three raw records — a row and a half of a two-column
    table — whatever the stride; the stride then counts per series from there."""
    path = write_lammps(tmp_path)
    records = mp.io.lammps.LammpsLogMetricReader().read(
        path, source="src", request=Request(since=3, stride=2, limit=4)
    )
    assert plotted(records) == [
        ("lammps/Press", 100.0, PRESS[1]),
        ("lammps/Temp", 200.0, TEMPS[2]),
        ("lammps/Press", 300.0, PRESS[3]),
        ("lammps/Temp", 400.0, TEMPS[4]),
    ]


def test_every_reader_puts_the_simulation_step_in_s(tmp_path: Path):
    for name, (reader, write) in READERS.items():
        directory = tmp_path / name
        directory.mkdir()
        steps = sorted({r["s"] for r in reader.read(write(directory), source="src")})
        assert steps == [float(s) for s in STEPS], name


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_yields_nothing(tmp_path: Path, limit: int):
    for name, (reader, write) in READERS.items():
        directory = tmp_path / name
        directory.mkdir()
        path = write(directory)
        assert list(reader.read(path, source="src", request=Request(limit=limit))) == []
