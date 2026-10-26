"""LAMMPS's files — ``molrs::io::lammps``.

The one-frame doors are functions of :mod:`molpy.io`: the data file
(:func:`~molpy.io.read_lammps_data`, :func:`~molpy.io.write_lammps_data`),
molecule templates (:func:`~molpy.io.read_lammps_molecule`,
:func:`~molpy.io.read_lammps_molecule_json` and their writers), one dump
snapshot (:func:`~molpy.io.read_lammps_dump`,
:func:`~molpy.io.write_lammps_dump`), ``dump local``
(:func:`~molpy.io.write_lammps_dump_local`), ``fix bond/react`` file sets
(:func:`~molpy.io.write_lammps_bond_react_map`,
:func:`~molpy.io.write_lammps_bond_react_system`) and logs
(:func:`~molpy.io.read_lammps_log`).

This module holds the family's classes: :class:`LammpsDumpReader` /
:class:`LammpsDumpWriter`, every snapshot of a dump;
:class:`LammpsForcefieldReader` / :class:`LammpsForcefieldWriter`, the
force-field include (``*.ff``), a data file's ``* Coeffs`` sections and the
``fix cmap`` file; :class:`BondReactTemplate`, one reaction's pre/post
template pair; the log records :func:`~molpy.io.read_lammps_log` hands out —
:class:`LammpsLog` and its runs, thermo tables, warnings, and performance and
timing summaries; :func:`is_lammps_log`, which tells a log by its first
run before it is read; and :func:`infer_lammps_atom_style`, the
``atom_style`` :func:`~molpy.io.write_lammps_data` writes a frame in when
none is named, for the input script's ``atom_style`` line.

It also holds :class:`LammpsLogMetricReader`, the metric reader of
``log.lammps`` thermo tables, published in the ``molcrafts.metric_readers``
entry-point group for any molcrafts viewer::

    [project.entry-points."molcrafts.metric_readers"]
    lammps_log = "molpy.io.lammps:LammpsLogMetricReader"

It parses nothing itself, and nothing here imports a viewer.
"""

from __future__ import annotations

from molpy._core import (
    BondReactTemplate,
    LammpsCpuUse,
    LammpsDumpReader,
    LammpsDumpWriter,
    LammpsForcefieldReader,
    LammpsForcefieldWriter,
    LammpsLoadBalance,
    LammpsLog,
    LammpsLogHeader,
    LammpsLoopTime,
    LammpsMemoryUsage,
    LammpsNeighborStatistics,
    LammpsPerformance,
    LammpsRun,
    LammpsThermo,
    LammpsTimingBreakdown,
    LammpsTimingRow,
    LammpsWarning,
    infer_lammps_atom_style,
    is_lammps_log,
)

from typing import TYPE_CHECKING
from molpy.io import metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from molpy.io.metric_reader import ReadRequest
_LAMMPS_BANNER = b"LAMMPS ("
_SCAN_BYTES = 4 * 1024 * 1024
_LOG_SUFFIXES = frozenset({".log", ".lammps", ".out", ".txt"})


class LammpsLogMetricReader:
    """``log.lammps`` thermo tables, parsed by ``molpy.io.read_lammps_log``.

    **Thermo rows carry no wall-clock.** LAMMPS records simulation steps, not
    timestamps, so each record is tagged ``wall_time_source: "ingest"`` and
    carries its step in ``s`` (the ``Step`` column; a table without one gets
    the row's ordinal across every run). Synthesizing per-row times from
    ``Loop time`` would be fabricated data.
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
        head = metric_reader.head(path)
        if not head:
            return False
        if is_lammps_log(head.decode("utf-8", errors="replace")):
            return True
        if _LAMMPS_BANNER not in head[:512]:
            return False
        # Banner present but the run starts past the probe window: read on, a
        # bounded window rather than the whole file.
        text = metric_reader.head(path, len(head) + _SCAN_BYTES)
        return is_lammps_log(text.decode("utf-8", errors="replace"))

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the thermo records *request* selects (see :mod:`molpy.io.metric_reader`).

        The raw stream is row-major: for each run, for each thermo row, one
        record per non-``Step`` column, so ``since`` counts those records.
        """
        from molpy.io import read_lammps_log

        parsed = read_lammps_log(path)
        yield from metric_reader.apply_request(request, _thermo_records(parsed, source))


def _thermo_records(parsed: LammpsLog, source: str) -> Iterator[dict[str, Any]]:
    """Every thermo row of every run, one record per column, ``s`` the step."""
    ordinal = 0
    for run_index, run in enumerate(parsed.runs):
        thermo = run.thermo
        if thermo is None or not thermo.columns:
            continue
        columns = thermo.columns
        step_at = columns.index("Step") if "Step" in columns else None
        series = [
            (position, f"lammps/{column}")
            for position, column in enumerate(columns)
            if position != step_at
        ]
        tags = {"run_index": run_index, "wall_time_source": "ingest", "source": source}
        for row in thermo.rows.tolist():
            step = float(row[step_at]) if step_at is not None else float(ordinal)
            ordinal += 1
            for position, key in series:
                yield {
                    "t": "scalar",
                    "k": key,
                    "s": step,
                    "v": float(row[position]),
                    "tags": dict(tags),
                }


__all__ = [
    "BondReactTemplate",
    "LammpsCpuUse",
    "LammpsDumpReader",
    "LammpsDumpWriter",
    "LammpsForcefieldReader",
    "LammpsForcefieldWriter",
    "LammpsLoadBalance",
    "LammpsLog",
    "LammpsLogHeader",
    "LammpsLoopTime",
    "LammpsMemoryUsage",
    "LammpsNeighborStatistics",
    "LammpsPerformance",
    "LammpsRun",
    "LammpsThermo",
    "LammpsTimingBreakdown",
    "LammpsTimingRow",
    "LammpsWarning",
    "infer_lammps_atom_style",
    "is_lammps_log",
    "LammpsLogMetricReader",
]
