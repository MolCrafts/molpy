"""Tests for ``molpy.io.LAMMPSLog`` parsing."""

import json
from pathlib import Path

import pytest

from molpy.io import LAMMPSLog, read_LAMMPS_log


def _write_log(tmp_path: Path, text: str) -> Path:
    log_file = tmp_path / "log.lammps"
    log_file.write_text(text.strip() + "\n")
    return log_file


def test_lammps_log_reads_default_thermo(TEST_DATA_DIR: Path) -> None:
    """Default thermo style: header + numeric rows up to ``Loop time``."""
    log = LAMMPSLog(TEST_DATA_DIR / "lammps-log" / "thermo_style_default.log")
    log.read()

    assert len(log.runs) == 1
    stage = log.runs[0].thermo.data
    columns = stage.dtype.names
    assert "Step" in columns
    assert "Temp" in columns
    assert stage["Step"][0] == 0
    assert stage.shape[0] >= 2  # at least two thermo rows


def test_lammps_log_to_dict_is_json_friendly(TEST_DATA_DIR: Path) -> None:
    """``to_dict`` returns plain Python types suitable for HTTP transport."""
    log = LAMMPSLog(TEST_DATA_DIR / "lammps-log" / "thermo_style_default.log").read()
    payload = log.to_dict()

    assert len(payload["runs"]) == 1
    stage = payload["runs"][0]["thermo"]
    assert isinstance(stage["columns"], list)
    assert isinstance(stage["rows"], list)
    assert all(isinstance(row, list) for row in stage["rows"])
    assert all(isinstance(value, float) for value in stage["rows"][0])


def test_lammps_log_handles_missing_file(tmp_path: Path) -> None:
    """Missing files raise ``FileNotFoundError`` rather than silent no-op."""
    log = LAMMPSLog(tmp_path / "nope.log")
    with pytest.raises(FileNotFoundError):
        log.read()


def test_lammps_log_handles_no_thermo_block(tmp_path: Path) -> None:
    """A log without a ``Per MPI rank`` block reports zero stages."""
    log_file = tmp_path / "empty.log"
    log_file.write_text("LAMMPS (1 Jan 2026)\n# nothing happened\n")
    log = LAMMPSLog(log_file).read()

    assert log.runs == ()
    assert log.version.startswith("LAMMPS")


def test_lammps_log_parses_two_stages(tmp_path: Path) -> None:
    """Two successive ``Per MPI rank ... Loop time`` blocks → two stages."""
    text = (
        "LAMMPS (1 Jan 2026)\n"
        "...\n"
        "Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes\n"
        "Step Temp PotEng\n"
        "0 300.0 -1000.0\n"
        "10 305.0 -1010.0\n"
        "Loop time of 0.1 on 1 procs\n"
        "...\n"
        "Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes\n"
        "Step Temp PotEng\n"
        "20 310.0 -1020.0\n"
        "30 315.0 -1030.0\n"
        "Loop time of 0.1 on 1 procs\n"
    )
    log_file = tmp_path / "two_stages.log"
    log_file.write_text(text)
    log = LAMMPSLog(log_file).read()

    assert len(log.runs) == 2
    assert log.runs[0].thermo.data["Step"][0] == 0
    assert log.runs[1].thermo.data["Step"][0] == 20


def test_read_LAMMPS_log_returns_nested_run_structure(tmp_path: Path) -> None:
    """New API follows LAMMPS run output sections."""
    log_file = _write_log(
        tmp_path,
        """
LAMMPS (1 Jan 2026)
using 1 OpenMP thread(s) per MPI task
Per MPI rank memory allocation (min/avg/max) = 4.5 | 4.75 | 5.0 Mbytes
Step Temp PotEng E_pair
0 300.0 -1000.0 -900.0
10 305.0 -1010.0 -910.0
Loop time of 0.2 on 2 procs for 10 steps with 100 atoms
Performance: 432.0 ns/day, 0.056 hours/ns, 50.0 timesteps/s, 5.0 katom-step/s
98.5% CPU use with 2 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.010 | 0.011 | 0.012 | 0.0 | 55.0
Neigh   | 0.002 | 0.003 | 0.004 | 0.0 | 15.0

Thread timings breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.005 | 0.006 | 0.007 | 0.0 | 30.0

Nlocal:    50 ave 55 max 45 min
Histogram: 1 0 0 0 0 0 0 0 0 1
Nghost:    10 ave 12 max 8 min
Histogram: 0 1 0 0 0 0 0 0 1 0
Neighs:    150 ave 160 max 140 min
Histogram: 0 0 1 0 0 0 0 1 0 0
Total # of neighbors = 300
Ave neighs/atom = 3.0
Ave special neighs/atom = 2.0
Neighbor list builds = 1
Dangerous builds = 0
WARNING: test warning
Total wall time: 0:00:01
""",
    )

    log = read_LAMMPS_log(log_file)

    assert log.version == "LAMMPS (1 Jan 2026)"
    assert log.total_wall_time == "0:00:01"
    assert len(log.runs) == 1

    run = log.runs[0]
    assert run.memory.average == 4.75
    assert run.thermo.columns == ("Step", "Temp", "PotEng", "E_pair")
    assert run.thermo.data["Step"].tolist() == [0.0, 10.0]
    assert run.loop_time.procs == 2
    assert run.loop_time.steps == 10
    assert run.loop_time.atoms == 100
    assert run.performance.atom_steps_units == "katom-step/s"
    assert run.CPU_use.MPI_tasks == 2
    assert run.CPU_use.OMP_threads == 1
    assert run.MPI_task_timing.rows[0].section == "Pair"
    assert run.thread_timing.rows[0].percent_total == 30.0
    assert run.load_balance[0].name == "Nlocal"
    assert run.load_balance[0].histogram == (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    assert run.neighbor_statistics.total_neighbors == 300
    assert run.neighbor_statistics.dangerous_builds == 0
    assert run.warnings[0].message == "test warning"
    assert "Step Temp PotEng E_pair" in run.raw_text


def test_read_LAMMPS_log_to_dict_is_json_friendly(tmp_path: Path) -> None:
    """Nested dataclasses serialize without NumPy objects."""
    log_file = _write_log(
        tmp_path,
        """
LAMMPS (1 Jan 2026)
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Step Temp
0 300
Loop time of 0.1 on 1 procs for 0 steps with 1 atoms
""",
    )
    payload = read_LAMMPS_log(log_file).to_dict()

    json.dumps(payload)
    assert payload["runs"][0]["thermo"]["columns"] == ["Step", "Temp"]
    assert payload["runs"][0]["CPU_use"] is None
    assert payload["runs"][0]["thermo"]["rows"] == [[0.0, 300.0]]
