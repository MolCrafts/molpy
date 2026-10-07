"""LAMMPS's files — :mod:`molrs.io.lammps`, mirrored by identity — and molpy's log metric reader.

Every native name is the molrs object (``mp.io.lammps.LammpsLog is
molrs.io.lammps.LammpsLog``): the lazy dump reader ``LammpsDumpReader``, the
``fix bond/react`` template ``BondReactTemplate`` and the log records
(``LammpsLog`` and its runs, thermo tables, warnings and timing summaries).
The files themselves are read and written by the functions of :mod:`molpy.io`
(``mp.io.read_lammps_data``, ``read_lammps_trajectory``, ``read_lammps_log``,
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
from molrs.io.lammps import __all__ as _native

from . import _metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from ._metric_reader import ReadRequest

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
        head = _metric_reader.head(path)
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

        stride = _metric_reader.stride_of(request)
        limit = _metric_reader.limit_of(request)
        skip = _metric_reader.since_of(request)
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
                    if seen <= skip or not _metric_reader.wanted(request, record):
                        continue
                    yield record
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


__all__ = [*_native, "LammpsLogMetricReader"]
