"""LAMMPS logs — :mod:`molrs.io.log`, mirrored by identity — and molpy's metric readers.

Every native name is the molrs object (``mp.io.log.LammpsLog is
molrs.io.log.LammpsLog``); the logs themselves are read by
``mp.io.read_lammps_log`` / ``read_lammps_log_str``.

molpy adds the metric readers of two text formats a run writes, published in
the ``molcrafts.metric_readers`` entry-point group for any molcrafts viewer:

* :class:`LammpsLogMetricReader` — ``log.lammps`` thermo tables;
* :class:`MlpJsonlMetricReader` — ``*.mlp.jsonl``, records that already are
  metric records.

Registration is declarative, at install time::

    [project.entry-points."molcrafts.metric_readers"]
    lammps_log = "molpy.io.log:LammpsLogMetricReader"
    mlp_jsonl  = "molpy.io.log:MlpJsonlMetricReader"

No reader parses a format itself, and nothing here imports a viewer.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING

from molrs.io.log import *  # noqa: F403
from molrs.io.log import __all__ as _native

from . import _metric

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from ._metric import ReadRequest

# LAMMPS writes this immediately before every thermo header row; the banner
# alone is not enough, because a log whose run never reached a thermo section
# has nothing to plot.
_LAMMPS_MARKER = b"Per MPI rank memory allocation"
_LAMMPS_BANNER = b"LAMMPS ("
_LOG_SUFFIXES = frozenset({".log", ".lammps", ".out", ".txt"})


class LammpsLogMetricReader:
    """``log.lammps`` thermo tables, parsed by ``molrs.io.read_lammps_log``.

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
        head = _metric.head(path)
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
        from molrs.io import read_lammps_log

        stride = _metric.stride_of(request)
        limit = _metric.limit_of(request)
        skip = _metric.since_of(request)
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
                    if seen <= skip or not _metric.wanted(request, record):
                        continue
                    yield record
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


class MlpJsonlMetricReader:
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
        stride = _metric.stride_of(request)
        limit = _metric.limit_of(request)
        skip = _metric.since_of(request)
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
                    record = _json.loads(stripped)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or "k" not in record:
                    continue
                index += 1
                if index <= skip or not _metric.wanted(request, record):
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


__all__ = [*_native, "LammpsLogMetricReader", "MlpJsonlMetricReader"]
