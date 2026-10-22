"""LAMMPS's files — :mod:`molrs.io.lammps`, mirrored by identity — and molpy's log metric reader.

Every native name is the molrs object (``mp.io.lammps.LammpsLog is
molrs.io.lammps.LammpsLog``): the lazy dump reader ``LammpsDumpReader``, the
``fix bond/react`` template ``BondReactTemplate`` and the log records
(``LammpsLog`` and its runs, thermo tables, warnings and timing summaries).
The files themselves are read and written by the functions of :mod:`molpy.io`
(``mp.io.read_lammps_data``, ``read_lammps_dump_trajectory``, ``read_lammps_log``,
``write_lammps_bond_react_system``, …).

molpy adds :class:`LammpsLogMetricReader`, the metric reader of ``log.lammps``
thermo tables, published in the ``molcrafts.metric_readers`` entry-point group
for any molcrafts viewer::

    [project.entry-points."molcrafts.metric_readers"]
    lammps_log = "molpy.io.lammps:LammpsLogMetricReader"

It parses nothing itself, and nothing here imports a viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molrs.io.lammps import *  # noqa: F403
from molrs.io.lammps import LammpsLog, is_lammps_log
from molrs.io.lammps import __all__ as _native

from molpy.io import metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from molpy.io.metric_reader import ReadRequest

# Whether a text is a LAMMPS run is molrs's call (``is_lammps_log``: the
# ``Per MPI rank memory allocation`` line that opens every thermo section, so a
# log whose run never reached a thermo table is not one). The banner only
# decides whether a file is worth reading past the probe window.
_LAMMPS_BANNER = b"LAMMPS ("
_SCAN_BYTES = 4 * 1024 * 1024
_LOG_SUFFIXES = frozenset({".log", ".lammps", ".out", ".txt"})


class LammpsLogMetricReader:
    """``log.lammps`` thermo tables, parsed by ``molrs.io.read_lammps_log``.

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
        """Yield the thermo records *request* selects (see :mod:`._metric_reader`).

        The raw stream is row-major: for each run, for each thermo row, one
        record per non-``Step`` column, so ``since`` counts those records.
        """
        from molrs.io import read_lammps_log

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


__all__ = [*_native, "LammpsLogMetricReader"]
