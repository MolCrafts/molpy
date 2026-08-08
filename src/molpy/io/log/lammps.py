"""LAMMPS log file parser.

Parsing logic lives in ``molrs`` (Rust). This module is a thin public façade:
it keeps the nested dataclass API used by callers and hydrates molrs payloads
into those types (including a NumPy structured array for thermo columns).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from molrs.io import read_lammps_log as _rs_read_lammps_log

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class LAMMPSLogHeader:
    """Header text before the first parsed LAMMPS run block."""

    lines: tuple[str, ...]

    @property
    def raw_text(self) -> str:
        """Header lines joined by newlines."""
        return "\n".join(self.lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {"lines": list(self.lines), "raw_text": self.raw_text}


@dataclass(frozen=True, slots=True)
class LAMMPSMemoryUsage:
    """``Per MPI rank memory allocation`` line."""

    minimum: float
    average: float
    maximum: float
    units: str
    raw_line: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSThermo:
    """LAMMPS thermo table with dynamic columns."""

    columns: tuple[str, ...]
    data: np.ndarray
    raw_lines: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        """Number of thermo rows."""
        return int(self.data.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "columns": list(self.columns),
            "rows": _array_rows(self.data),
            "raw_lines": list(self.raw_lines),
        }


@dataclass(frozen=True, slots=True)
class LAMMPSLoopTime:
    """``Loop time`` summary line."""

    seconds: float
    procs: int
    steps: int | None
    atoms: int | None
    raw_line: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSPerformance:
    """LAMMPS ``Performance`` summary line."""

    ns_per_day: float
    hours_per_ns: float
    timesteps_per_second: float
    atom_steps_per_second: float | None
    atom_steps_units: str | None
    raw_line: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSCPUUse:
    """``% CPU use`` summary line."""

    percent: float
    MPI_tasks: int
    OMP_threads: int | None
    raw_line: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSTimingRow:
    """One row from a LAMMPS timing breakdown table."""

    section: str
    min_time: float
    avg_time: float
    max_time: float
    percent_varavg: float
    percent_total: float
    raw_line: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSTimingBreakdown:
    """``MPI task timing breakdown`` or thread timing table."""

    title: str
    rows: tuple[LAMMPSTimingRow, ...]
    raw_lines: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSLoadBalance:
    """LAMMPS load-balance statistic plus optional histogram."""

    name: str
    average: float
    maximum: float
    minimum: float
    histogram: tuple[int, ...]
    raw_lines: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSNeighborStatistics:
    """Neighbor-list statistics emitted after a run."""

    total_neighbors: int | None
    ave_neighs_per_atom: float | None
    ave_special_neighs_per_atom: float | None
    neighbor_list_builds: int | None
    dangerous_builds: int | None
    raw_lines: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSWarning:
    """A warning line from the LAMMPS log."""

    message: str
    raw_line: str
    line_number: int | None = None
    run_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(frozen=True, slots=True)
class LAMMPSRun:
    """One LAMMPS run output block."""

    index: int
    setup_log: tuple[str, ...]
    memory: LAMMPSMemoryUsage | None
    thermo: LAMMPSThermo | None
    loop_time: LAMMPSLoopTime | None
    performance: LAMMPSPerformance | None
    CPU_use: LAMMPSCPUUse | None
    MPI_task_timing: LAMMPSTimingBreakdown | None
    thread_timing: LAMMPSTimingBreakdown | None
    load_balance: tuple[LAMMPSLoadBalance, ...]
    neighbor_statistics: LAMMPSNeighborStatistics | None
    warnings: tuple[LAMMPSWarning, ...]
    unparsed_log: tuple[str, ...]
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return _dataclass_to_dict(self)


@dataclass(slots=True, init=False)
class LAMMPSLog:
    """Parsed LAMMPS log with one structured entry per run.

    Args:
        file: Path to a LAMMPS log file.
        style: Thermo style. Only ``"default"`` is currently parsed.

    """

    path: Path
    version: str | None
    header: LAMMPSLogHeader
    runs: tuple[LAMMPSRun, ...]
    total_wall_time: str | None
    warnings: tuple[LAMMPSWarning, ...]
    raw_text: str
    style: str

    def __init__(self, file: PathLike, style: str = "default"):
        self.path = Path(file)
        self.version = None
        self.header = LAMMPSLogHeader(lines=())
        self.runs = ()
        self.total_wall_time = None
        self.warnings = ()
        self.raw_text = ""
        self.style = style

    def read(self) -> "LAMMPSLog":
        """Read and parse the log file. Returns ``self`` for chaining."""
        payload = _rs_read_lammps_log(str(self.path), self.style)
        parsed = _hydrate_log(payload, path=self.path, style=self.style)
        self.version = parsed.version
        self.header = parsed.header
        self.runs = parsed.runs
        self.total_wall_time = parsed.total_wall_time
        self.warnings = parsed.warnings
        self.raw_text = parsed.raw_text
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "path": str(self.path),
            "version": self.version,
            "header": self.header.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
            "total_wall_time": self.total_wall_time,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "raw_text": self.raw_text,
        }


def read_LAMMPS_log(file: PathLike) -> LAMMPSLog:
    """Read a LAMMPS log file into a nested dataclass result.

    Args:
        file: Path to the LAMMPS log file.

    Returns:
        Parsed ``LAMMPSLog`` object.
    """
    return LAMMPSLog(file).read()


def _hydrate_log(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    style: str | None = None,
) -> LAMMPSLog:
    """Build a :class:`LAMMPSLog` from a molrs nested-dict payload."""
    log = LAMMPSLog(path if path is not None else payload.get("path", "<string>"))
    log.style = style if style is not None else str(payload.get("style", "default"))
    log.version = payload.get("version")
    header = payload.get("header") or {}
    log.header = LAMMPSLogHeader(lines=tuple(header.get("lines") or ()))
    log.runs = tuple(_hydrate_run(run) for run in payload.get("runs") or ())
    log.total_wall_time = payload.get("total_wall_time")
    log.warnings = tuple(
        _hydrate_warning(warning) for warning in payload.get("warnings") or ()
    )
    log.raw_text = payload.get("raw_text") or ""
    return log


def _hydrate_run(payload: dict[str, Any]) -> LAMMPSRun:
    return LAMMPSRun(
        index=int(payload.get("index", 0)),
        setup_log=tuple(payload.get("setup_log") or ()),
        memory=_hydrate_memory(payload.get("memory")),
        thermo=_hydrate_thermo(payload.get("thermo")),
        loop_time=_hydrate_loop_time(payload.get("loop_time")),
        performance=_hydrate_performance(payload.get("performance")),
        CPU_use=_hydrate_cpu_use(payload.get("CPU_use")),
        MPI_task_timing=_hydrate_timing_breakdown(payload.get("MPI_task_timing")),
        thread_timing=_hydrate_timing_breakdown(payload.get("thread_timing")),
        load_balance=tuple(
            _hydrate_load_balance(item) for item in payload.get("load_balance") or ()
        ),
        neighbor_statistics=_hydrate_neighbor_statistics(
            payload.get("neighbor_statistics")
        ),
        warnings=tuple(
            _hydrate_warning(warning) for warning in payload.get("warnings") or ()
        ),
        unparsed_log=tuple(payload.get("unparsed_log") or ()),
        raw_text=payload.get("raw_text") or "",
    )


def _hydrate_memory(payload: dict[str, Any] | None) -> LAMMPSMemoryUsage | None:
    if not payload:
        return None
    return LAMMPSMemoryUsage(
        minimum=float(payload["minimum"]),
        average=float(payload["average"]),
        maximum=float(payload["maximum"]),
        units=str(payload["units"]),
        raw_line=str(payload["raw_line"]),
    )


def _hydrate_thermo(payload: dict[str, Any] | None) -> LAMMPSThermo | None:
    if not payload:
        return None
    columns = tuple(payload.get("columns") or ())
    rows = payload.get("rows") or []
    if not columns or not rows:
        return None
    dtype = np.dtype({"names": columns, "formats": ["f8"] * len(columns)})
    data = np.array([tuple(row) for row in rows], dtype=dtype)
    return LAMMPSThermo(
        columns=columns,
        data=data,
        raw_lines=tuple(payload.get("raw_lines") or ()),
    )


def _hydrate_loop_time(payload: dict[str, Any] | None) -> LAMMPSLoopTime | None:
    if not payload:
        return None
    return LAMMPSLoopTime(
        seconds=float(payload["seconds"]),
        procs=int(payload["procs"]),
        steps=_int_or_none(payload.get("steps")),
        atoms=_int_or_none(payload.get("atoms")),
        raw_line=str(payload["raw_line"]),
    )


def _hydrate_performance(payload: dict[str, Any] | None) -> LAMMPSPerformance | None:
    if not payload:
        return None
    return LAMMPSPerformance(
        ns_per_day=float(payload["ns_per_day"]),
        hours_per_ns=float(payload["hours_per_ns"]),
        timesteps_per_second=float(payload["timesteps_per_second"]),
        atom_steps_per_second=_float_or_none(payload.get("atom_steps_per_second")),
        atom_steps_units=payload.get("atom_steps_units"),
        raw_line=str(payload["raw_line"]),
    )


def _hydrate_cpu_use(payload: dict[str, Any] | None) -> LAMMPSCPUUse | None:
    if not payload:
        return None
    return LAMMPSCPUUse(
        percent=float(payload["percent"]),
        MPI_tasks=int(payload["MPI_tasks"]),
        OMP_threads=_int_or_none(payload.get("OMP_threads")),
        raw_line=str(payload["raw_line"]),
    )


def _hydrate_timing_breakdown(
    payload: dict[str, Any] | None,
) -> LAMMPSTimingBreakdown | None:
    if not payload:
        return None
    rows = tuple(
        LAMMPSTimingRow(
            section=str(row["section"]),
            min_time=float(row["min_time"]),
            avg_time=float(row["avg_time"]),
            max_time=float(row["max_time"]),
            percent_varavg=float(row["percent_varavg"]),
            percent_total=float(row["percent_total"]),
            raw_line=str(row["raw_line"]),
        )
        for row in payload.get("rows") or ()
    )
    return LAMMPSTimingBreakdown(
        title=str(payload.get("title") or ""),
        rows=rows,
        raw_lines=tuple(payload.get("raw_lines") or ()),
    )


def _hydrate_load_balance(payload: dict[str, Any]) -> LAMMPSLoadBalance:
    return LAMMPSLoadBalance(
        name=str(payload["name"]),
        average=float(payload["average"]),
        maximum=float(payload["maximum"]),
        minimum=float(payload["minimum"]),
        histogram=tuple(int(v) for v in payload.get("histogram") or ()),
        raw_lines=tuple(payload.get("raw_lines") or ()),
    )


def _hydrate_neighbor_statistics(
    payload: dict[str, Any] | None,
) -> LAMMPSNeighborStatistics | None:
    if not payload:
        return None
    return LAMMPSNeighborStatistics(
        total_neighbors=_int_or_none(payload.get("total_neighbors")),
        ave_neighs_per_atom=_float_or_none(payload.get("ave_neighs_per_atom")),
        ave_special_neighs_per_atom=_float_or_none(
            payload.get("ave_special_neighs_per_atom")
        ),
        neighbor_list_builds=_int_or_none(payload.get("neighbor_list_builds")),
        dangerous_builds=_int_or_none(payload.get("dangerous_builds")),
        raw_lines=tuple(payload.get("raw_lines") or ()),
    )


def _hydrate_warning(payload: dict[str, Any]) -> LAMMPSWarning:
    return LAMMPSWarning(
        message=str(payload.get("message") or ""),
        raw_line=str(payload.get("raw_line") or ""),
        line_number=_int_or_none(payload.get("line_number")),
        run_index=_int_or_none(payload.get("run_index")),
    )


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def _array_rows(array: np.ndarray) -> list[list[float]]:
    if array.dtype.names:
        return [[float(value) for value in record] for record in array.tolist()]
    return np.asarray(array).astype(float).tolist()


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {field.name: _jsonify(getattr(obj, field.name)) for field in fields(obj)}


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _array_rows(value)
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return _dataclass_to_dict(value)
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "LAMMPSCPUUse",
    "LAMMPSLoadBalance",
    "LAMMPSLog",
    "LAMMPSLogHeader",
    "LAMMPSLoopTime",
    "LAMMPSMemoryUsage",
    "LAMMPSNeighborStatistics",
    "LAMMPSPerformance",
    "LAMMPSRun",
    "LAMMPSThermo",
    "LAMMPSTimingBreakdown",
    "LAMMPSTimingRow",
    "LAMMPSWarning",
    "read_LAMMPS_log",
]
