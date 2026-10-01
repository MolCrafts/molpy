"""Metric readers molpy publishes for any molcrafts viewer.

molpy owns these formats, so molpy is what knows how to turn them into
plottable series. A viewer (molplot, through molexp) never parses anything and
never imports molpy: it looks up a reader by format and asks it for records.

Registration is declarative, at install time::

    [project.entry-points."molcrafts.metric_readers"]
    lammps_log = "molpy.integrations.metric_readers:LammpsLogReader"
    mrec       = "molpy.integrations.metric_readers:MrecReader"
    mlp_jsonl  = "molpy.integrations.metric_readers:MlpJsonlReader"

Nothing here imports molexp. The reader contract is a structural Protocol —
``format`` / ``sniff`` / ``read`` plus the optional ``patterns`` and
``tailable`` hints — so matching it is a matter of shape, not of dependency.
That is what keeps the arrow pointing one way: a format's owner publishes a
capability, and the platform consumes it.

**The viewer sets the sampling policy.** ``read`` receives a request carrying
``stride`` — how many samples of each series to skip — and each reader pushes
it as far down as its parser allows. For a LAMMPS thermo table that is a numpy
slice, so a 160k-row table costs a slice rather than 160k dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

# LAMMPS writes this immediately before every thermo header row; the banner
# alone is not enough, because a log whose run never reached a thermo section
# has nothing to plot.
_LAMMPS_MARKER = b"Per MPI rank memory allocation"
_LAMMPS_BANNER = b"LAMMPS ("
_PROBE_BYTES = 64 * 1024
_LOG_SUFFIXES = frozenset({".log", ".lammps", ".out", ".txt"})


class ReadRequest(Protocol):
    """What the viewer asked for. Supplied by the platform, matched by shape."""

    since: int
    stride: int
    limit: int | None
    metric_type: str | None
    keys: tuple[str, ...]


def _wanted(request: ReadRequest | None, record: dict[str, Any]) -> bool:
    if request is None:
        return True
    if request.metric_type is not None and record.get("t") != request.metric_type:
        return False
    return not (request.keys and record.get("k") not in request.keys)


def _stride_of(request: ReadRequest | None) -> int:
    return 1 if request is None else max(1, request.stride)


def _limit_of(request: ReadRequest | None) -> int | None:
    return None if request is None else request.limit


def _since_of(request: ReadRequest | None) -> int:
    return 0 if request is None else max(0, request.since)


def _head(path: Path, size: int = _PROBE_BYTES) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


class LammpsLogReader:
    """``log.lammps`` thermo tables, parsed by molpy's LAMMPS log reader.

    **Thermo rows carry no wall-clock.** LAMMPS records simulation steps, not
    timestamps, so each record is tagged ``wall_time_source: "ingest"`` and
    carries its step in ``s``. Synthesizing per-row times from ``Loop time``
    would be fabricated data.
    """

    format = "lammps_log"
    patterns = (
        "**/log.lammps",
        "**/*.lammps",
        "**/lammps*.log",
        "**/lammps*.out",
        "**/log.*.lammps",
    )
    tailable = False

    def sniff(self, path: Path) -> bool:
        if not path.is_file() or path.suffix not in _LOG_SUFFIXES:
            return False
        head = _head(path)
        if not head:
            return False
        if _LAMMPS_MARKER in head:
            return True
        if _LAMMPS_BANNER not in head[:512]:
            return False
        # Banner present but the marker sits past the probe window — scan on
        # in bounded chunks rather than loading the whole file.
        with path.open("rb") as handle:
            handle.seek(len(head))
            return _LAMMPS_MARKER in handle.read(4 * 1024 * 1024)

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        from molpy.io import read_lammps_log

        stride = _stride_of(request)
        limit = _limit_of(request)
        skip = _since_of(request)
        parsed = read_lammps_log(path)
        emitted = 0
        seen = 0

        for run_index, run in enumerate(parsed.runs):
            thermo = run.thermo
            if thermo is None or not thermo.columns:
                continue
            columns = thermo.columns
            step_at = columns.index("Step") if "Step" in columns else None
            # Push the sampling into the array: the viewer asked for every
            # Nth *sample*, and a row is one sample of every column at once.
            rows = thermo.rows[::stride] if stride > 1 else thermo.rows
            for row in rows:
                step = float(row[step_at]) if step_at is not None else None
                for position, column in enumerate(columns):
                    if position == step_at:
                        continue
                    record = {
                        "t": "scalar",
                        "k": f"lammps/{column}",
                        "v": float(row[position]),
                        "tags": {
                            "run_index": run_index,
                            "wall_time_source": "ingest",
                            "source": source,
                        },
                    }
                    if step is not None:
                        record["s"] = step
                    seen += 1
                    if seen <= skip or not _wanted(request, record):
                        continue
                    yield record
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


class MrecReader:
    """``*.mrec`` records — the per-frame series a trajectory carries.

    A record store is not a metrics file: what it has that a chart can draw is
    the frame index against simulation ``step`` and ``time``. Those are
    emitted and nothing else is invented.
    """

    format = "mrec"
    patterns = ("**/*.mrec", "**/*.mrec/")
    tailable = False

    def sniff(self, path: Path) -> bool:
        if path.suffix != ".mrec":
            return False
        if path.is_dir():
            return True
        return path.is_file()

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        from molpy.io import read_mrec_trajectory

        stride = _stride_of(request)
        limit = _limit_of(request)
        skip = _since_of(request)
        trajectory = read_mrec_trajectory(path)

        series: dict[str, Any] = {}
        for name in ("step", "time"):
            values = getattr(trajectory, name)
            if values is not None and len(values):
                series[name] = values

        emitted = 0
        seen = 0
        for name, values in series.items():
            sampled = values[::stride] if stride > 1 else values
            for index, value in enumerate(sampled):
                record = {
                    "t": "scalar",
                    "k": f"mrec/{name}",
                    "s": float(index * stride),
                    "v": float(value),
                    "tags": {"source": source},
                }
                seen += 1
                if seen <= skip or not _wanted(request, record):
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


class MlpJsonlReader:
    """``*.mlp.jsonl`` — records that already are metric records.

    It is the format molexp writes while a run is in flight, which makes it
    the one that grows underneath a live chart. That is why it is *tailable*;
    it is not otherwise privileged, and it is read through the same seam as
    every other format.
    """

    format = "mlp_jsonl"
    patterns = ("**/*.mlp.jsonl",)
    tailable = True

    def sniff(self, path: Path) -> bool:
        return path.is_file() and path.name.endswith(".mlp.jsonl")

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        stride = _stride_of(request)
        limit = _limit_of(request)
        skip = _since_of(request)
        # A JSONL file is one sample per line and the series are interleaved,
        # so striding is counted per series — striding the file would land on
        # the same key every time and drop the others.
        seen_per_series: dict[Any, int] = {}
        emitted = 0
        index = 0

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or "k" not in record:
                    continue
                index += 1
                if index <= skip or not _wanted(request, record):
                    continue
                if stride > 1:
                    key = record.get("k")
                    position = seen_per_series.get(key, 0)
                    seen_per_series[key] = position + 1
                    if position % stride:
                        continue
                tags = record.get("tags")
                if isinstance(tags, dict):
                    tags.setdefault("source", source)
                else:
                    record["tags"] = {"source": source}
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


__all__ = ["LammpsLogReader", "MlpJsonlReader", "MrecReader"]
