"""LAMMPS log file parser.

This module parses the standard LAMMPS run output structure documented in
``Run_output.html``: thermo tables, loop timing, performance summaries,
CPU/MPI timing, load-balance statistics, neighbor statistics, and warnings.
Unrecognized lines are preserved so callers can still inspect information that
does not yet have a structured representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

PathLike = str | Path

_MEMORY_RE = re.compile(
    r"^Per MPI rank memory allocation \(min/avg/max\) = "
    r"(?P<minimum>\S+) \| (?P<average>\S+) \| (?P<maximum>\S+) (?P<units>\S+)"
)
_LOOP_TIME_RE = re.compile(
    r"^Loop time of (?P<seconds>\S+) on (?P<procs>\d+) procs"
    r"(?: for (?P<steps>\d+) steps with (?P<atoms>\d+) atoms)?"
)
_PERFORMANCE_RE = re.compile(
    r"^Performance:\s+"
    r"(?P<ns_per_day>\S+) ns/day,\s+"
    r"(?P<hours_per_ns>\S+) hours/ns,\s+"
    r"(?P<timesteps_per_second>\S+) timesteps/s"
    r"(?:,\s+(?P<atom_steps_per_second>\S+) (?P<atom_steps_units>\S+))?"
)
_CPU_USE_RE = re.compile(
    r"^(?P<percent>\S+)% CPU use with (?P<MPI_tasks>\d+) MPI tasks"
    r"(?: x (?P<OMP_threads>\d+) OpenMP threads)?"
)
_LOAD_BALANCE_RE = re.compile(
    r"^(?P<name>Nlocal|Nghost|Neighs):\s+"
    r"(?P<average>\S+) ave (?P<maximum>\S+) max (?P<minimum>\S+) min"
)
_WARNING_RE = re.compile(r"^WARNING:\s*(?P<message>.*)")


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
        text = self.path.read_text()
        parsed = _parse_LAMMPS_log_text(self.path, text, self.style)
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


def _parse_LAMMPS_log_text(path: Path, text: str, style: str) -> LAMMPSLog:
    lines = text.splitlines()
    run_ranges = _find_run_ranges(lines)
    header_end = run_ranges[0][0] if run_ranges else len(lines)
    total_wall_time = _parse_total_wall_time(lines)
    if total_wall_time is not None and not run_ranges:
        header_end = min(header_end, _total_wall_time_index(lines))

    log = LAMMPSLog(path)
    log.version = _first_line(text)
    log.header = LAMMPSLogHeader(lines=tuple(lines[:header_end]))
    log.runs = tuple(
        _parse_run(index, lines[start:end], start, style)
        for index, (start, end) in enumerate(run_ranges)
    )
    log.total_wall_time = total_wall_time
    log.warnings = tuple(_collect_warnings(lines, 0, None))
    log.raw_text = text
    return log


def _find_run_ranges(lines: list[str]) -> list[tuple[int, int]]:
    starts = [idx for idx, line in enumerate(lines) if _MEMORY_RE.match(line.strip())]
    ranges: list[tuple[int, int]] = []
    wall_idx = _total_wall_time_index(lines)
    for idx, start in enumerate(starts):
        next_start = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        end = min(next_start, wall_idx) if wall_idx >= 0 else next_start
        ranges.append((start, end))
    return ranges


def _parse_run(
    index: int,
    lines: list[str],
    line_offset: int,
    style: str,
) -> LAMMPSRun:
    consumed: set[int] = set()
    memory = _parse_memory(lines, consumed)
    loop_idx = _first_index(lines, lambda line: _LOOP_TIME_RE.match(line.strip()))
    thermo = _parse_thermo(lines, consumed, loop_idx, style)
    loop_time = _parse_loop_time(lines, consumed)
    performance = _parse_performance(lines, consumed)
    CPU_use = _parse_CPU_use(lines, consumed)
    MPI_task_timing = _parse_timing_breakdown(
        lines, consumed, "MPI task timing breakdown"
    )
    thread_timing = _parse_timing_breakdown(lines, consumed, "Thread timings breakdown")
    if thread_timing is None:
        thread_timing = _parse_timing_breakdown(lines, consumed, "Thread timing")
    load_balance = tuple(_parse_load_balance(lines, consumed))
    neighbor_statistics = _parse_neighbor_statistics(lines, consumed)
    warnings = tuple(_collect_warnings(lines, line_offset, index))
    for idx, line in enumerate(lines):
        if _WARNING_RE.match(line.strip()):
            consumed.add(idx)

    setup_log = tuple(
        line
        for idx, line in enumerate(lines[: loop_idx if loop_idx >= 0 else 0])
        if idx not in consumed and line.strip()
    )
    unparsed_log = tuple(
        line
        for idx, line in enumerate(lines)
        if idx not in consumed and line.strip() and line not in setup_log
    )

    return LAMMPSRun(
        index=index,
        setup_log=setup_log,
        memory=memory,
        thermo=thermo,
        loop_time=loop_time,
        performance=performance,
        CPU_use=CPU_use,
        MPI_task_timing=MPI_task_timing,
        thread_timing=thread_timing,
        load_balance=load_balance,
        neighbor_statistics=neighbor_statistics,
        warnings=warnings,
        unparsed_log=unparsed_log,
        raw_text="\n".join(lines),
    )


def _parse_memory(lines: list[str], consumed: set[int]) -> LAMMPSMemoryUsage | None:
    for idx, line in enumerate(lines):
        match = _MEMORY_RE.match(line.strip())
        if not match:
            continue
        consumed.add(idx)
        return LAMMPSMemoryUsage(
            minimum=float(match["minimum"]),
            average=float(match["average"]),
            maximum=float(match["maximum"]),
            units=match["units"],
            raw_line=line,
        )
    return None


def _parse_thermo(
    lines: list[str],
    consumed: set[int],
    loop_idx: int,
    style: str,
) -> LAMMPSThermo | None:
    if style != "default" or loop_idx < 0:
        return None

    start = _first_index(lines, lambda line: _MEMORY_RE.match(line.strip()))
    if start < 0:
        start = -1
    body_indices = [idx for idx in range(start + 1, loop_idx) if lines[idx].strip()]
    if len(body_indices) < 2:
        return None

    header_idx = body_indices[0]
    columns = tuple(lines[header_idx].split())
    data_indices: list[int] = []
    for idx in body_indices[1:]:
        parts = lines[idx].split()
        if len(parts) != len(columns) or not all(_is_float(part) for part in parts):
            break
        data_indices.append(idx)

    if not columns or not data_indices:
        return None

    dtype = np.dtype({"names": columns, "formats": ["f8"] * len(columns)})
    try:
        data = np.loadtxt(
            [lines[idx] for idx in data_indices],
            dtype=dtype,
            ndmin=1,
        )
    except ValueError:
        return None

    consumed.add(header_idx)
    consumed.update(data_indices)
    raw_lines = tuple(lines[idx] for idx in [header_idx, *data_indices])
    return LAMMPSThermo(columns=columns, data=data, raw_lines=raw_lines)


def _parse_loop_time(lines: list[str], consumed: set[int]) -> LAMMPSLoopTime | None:
    for idx, line in enumerate(lines):
        match = _LOOP_TIME_RE.match(line.strip())
        if not match:
            continue
        consumed.add(idx)
        return LAMMPSLoopTime(
            seconds=float(match["seconds"]),
            procs=int(match["procs"]),
            steps=_int_or_none(match["steps"]),
            atoms=_int_or_none(match["atoms"]),
            raw_line=line,
        )
    return None


def _parse_performance(
    lines: list[str],
    consumed: set[int],
) -> LAMMPSPerformance | None:
    for idx, line in enumerate(lines):
        match = _PERFORMANCE_RE.match(line.strip())
        if not match:
            continue
        consumed.add(idx)
        return LAMMPSPerformance(
            ns_per_day=float(match["ns_per_day"]),
            hours_per_ns=float(match["hours_per_ns"]),
            timesteps_per_second=float(match["timesteps_per_second"]),
            atom_steps_per_second=_float_or_none(match["atom_steps_per_second"]),
            atom_steps_units=match["atom_steps_units"],
            raw_line=line,
        )
    return None


def _parse_CPU_use(lines: list[str], consumed: set[int]) -> LAMMPSCPUUse | None:
    for idx, line in enumerate(lines):
        match = _CPU_USE_RE.match(line.strip())
        if not match:
            continue
        consumed.add(idx)
        return LAMMPSCPUUse(
            percent=float(match["percent"]),
            MPI_tasks=int(match["MPI_tasks"]),
            OMP_threads=_int_or_none(match["OMP_threads"]),
            raw_line=line,
        )
    return None


def _parse_timing_breakdown(
    lines: list[str],
    consumed: set[int],
    title_prefix: str,
) -> LAMMPSTimingBreakdown | None:
    start = _first_index(lines, lambda line: line.strip().startswith(title_prefix))
    if start < 0:
        return None

    raw_indices = [start]
    rows: list[LAMMPSTimingRow] = []
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            break
        row = _parse_timing_row(line)
        if row is not None:
            rows.append(row)
            raw_indices.append(idx)
            continue
        if "|" in line or set(stripped) <= {"-"} or stripped.startswith("Section"):
            raw_indices.append(idx)
            continue
        if rows:
            break

    if not rows:
        return None
    consumed.update(raw_indices)
    return LAMMPSTimingBreakdown(
        title=lines[start].strip().rstrip(":"),
        rows=tuple(rows),
        raw_lines=tuple(lines[idx] for idx in raw_indices),
    )


def _parse_timing_row(line: str) -> LAMMPSTimingRow | None:
    if "|" not in line:
        return None
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 6 or not all(_is_float(part) for part in parts[1:]):
        return None
    return LAMMPSTimingRow(
        section=parts[0],
        min_time=float(parts[1]),
        avg_time=float(parts[2]),
        max_time=float(parts[3]),
        percent_varavg=float(parts[4]),
        percent_total=float(parts[5]),
        raw_line=line,
    )


def _parse_load_balance(
    lines: list[str],
    consumed: set[int],
) -> list[LAMMPSLoadBalance]:
    entries: list[LAMMPSLoadBalance] = []
    idx = 0
    while idx < len(lines):
        match = _LOAD_BALANCE_RE.match(lines[idx].strip())
        if not match:
            idx += 1
            continue

        raw_indices = [idx]
        histogram: tuple[int, ...] = ()
        next_idx = idx + 1
        if next_idx < len(lines) and lines[next_idx].strip().startswith("Histogram:"):
            histogram = tuple(int(value) for value in lines[next_idx].split()[1:])
            raw_indices.append(next_idx)
        consumed.update(raw_indices)
        entries.append(
            LAMMPSLoadBalance(
                name=match["name"],
                average=float(match["average"]),
                maximum=float(match["maximum"]),
                minimum=float(match["minimum"]),
                histogram=histogram,
                raw_lines=tuple(lines[line_idx] for line_idx in raw_indices),
            )
        )
        idx = raw_indices[-1] + 1
    return entries


def _parse_neighbor_statistics(
    lines: list[str],
    consumed: set[int],
) -> LAMMPSNeighborStatistics | None:
    keys = {
        "Total # of neighbors": ("total_neighbors", int),
        "Ave neighs/atom": ("ave_neighs_per_atom", float),
        "Ave special neighs/atom": ("ave_special_neighs_per_atom", float),
        "Neighbor list builds": ("neighbor_list_builds", int),
        "Dangerous builds": ("dangerous_builds", int),
    }
    values: dict[str, Any] = {
        "total_neighbors": None,
        "ave_neighs_per_atom": None,
        "ave_special_neighs_per_atom": None,
        "neighbor_list_builds": None,
        "dangerous_builds": None,
    }
    raw_indices: list[int] = []
    for idx, line in enumerate(lines):
        if "=" not in line:
            continue
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        if key not in keys:
            continue
        field_name, converter = keys[key]
        values[field_name] = (
            converter(float(raw_value)) if converter is int else float(raw_value)
        )
        raw_indices.append(idx)

    if not raw_indices:
        return None
    consumed.update(raw_indices)
    return LAMMPSNeighborStatistics(
        total_neighbors=values["total_neighbors"],
        ave_neighs_per_atom=values["ave_neighs_per_atom"],
        ave_special_neighs_per_atom=values["ave_special_neighs_per_atom"],
        neighbor_list_builds=values["neighbor_list_builds"],
        dangerous_builds=values["dangerous_builds"],
        raw_lines=tuple(lines[idx] for idx in raw_indices),
    )


def _collect_warnings(
    lines: list[str],
    line_offset: int,
    run_index: int | None,
) -> list[LAMMPSWarning]:
    warnings: list[LAMMPSWarning] = []
    for idx, line in enumerate(lines):
        match = _WARNING_RE.match(line.strip())
        if match:
            warnings.append(
                LAMMPSWarning(
                    message=match["message"],
                    raw_line=line,
                    line_number=line_offset + idx + 1,
                    run_index=run_index,
                )
            )
    return warnings


def _parse_total_wall_time(lines: list[str]) -> str | None:
    idx = _total_wall_time_index(lines)
    if idx < 0:
        return None
    return lines[idx].split(":", 1)[1].strip() if ":" in lines[idx] else lines[idx]


def _total_wall_time_index(lines: list[str]) -> int:
    return _first_index(lines, lambda line: line.strip().startswith("Total wall time:"))


def _first_index(lines: list[str], predicate) -> int:
    for idx, line in enumerate(lines):
        if predicate(line):
            return idx
    return -1


def _first_line(text: str) -> str | None:
    line = text.splitlines()[0] if text.splitlines() else None
    return line


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value is not None else None


def _int_or_none(value: str | None) -> int | None:
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
