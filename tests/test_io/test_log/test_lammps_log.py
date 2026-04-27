"""Tests for ``molpy.io.LAMMPSLog`` thermo extraction."""

from pathlib import Path

import pytest

from molpy.io import LAMMPSLog


def test_lammps_log_reads_default_thermo(TEST_DATA_DIR: Path) -> None:
    """Default thermo style: header + numeric rows up to ``Loop time``."""
    log = LAMMPSLog(TEST_DATA_DIR / "lammps-log" / "thermo_style_default.log")
    log.read()

    assert log["n_stages"] == 1
    stage = log["stages"][0]
    columns = stage.dtype.names
    assert "Step" in columns
    assert "Temp" in columns
    assert stage["Step"][0] == 0
    assert stage.shape[0] >= 2  # at least two thermo rows


def test_lammps_log_to_dict_is_json_friendly(TEST_DATA_DIR: Path) -> None:
    """``to_dict`` returns plain Python types suitable for HTTP transport."""
    log = LAMMPSLog(TEST_DATA_DIR / "lammps-log" / "thermo_style_default.log").read()
    payload = log.to_dict()

    assert payload["n_stages"] == 1
    assert isinstance(payload["stages"], list)
    stage = payload["stages"][0]
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

    assert log["n_stages"] == 0
    assert log["stages"] == []
    assert log["version"].startswith("LAMMPS")


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

    assert log["n_stages"] == 2
    assert log["stages"][0]["Step"][0] == 0
    assert log["stages"][1]["Step"][0] == 20
