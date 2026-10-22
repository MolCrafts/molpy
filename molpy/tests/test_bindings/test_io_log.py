"""FFI seam of the structured LAMMPS log: classes, dtypes and the dict form.

The parser's numbers are proven in the Rust unit tests; this file checks
what crosses the boundary.
"""

from __future__ import annotations

import json

import molpy as mp
import numpy as np

_LOG = """\
LAMMPS (1 Jan 2026)
units real
run 10
Per MPI rank memory allocation (min/avg/max) = 3.1 | 3.2 | 3.3 Mbytes
   Step          Temp          PotEng
         0   300            -1.5
        10   298.5          -1.7
Loop time of 0.5 on 2 procs for 10 steps with 100 atoms
Performance: 1.728 ns/day, 13.889 hours/ns, 20.000 timesteps/s, 2.000 katom-step/s
WARNING: test warning (src/x.cpp:1)
Total wall time: 0:00:01
"""


def test_parse_returns_the_structured_classes():
    log = mp.io.read_lammps_log(mp.io.Text(_LOG))
    assert isinstance(log, mp.io.lammps.LammpsLog)
    assert log.version == "LAMMPS (1 Jan 2026)"
    assert len(log) == len(log.runs) == 1
    run = log.runs[0]
    assert isinstance(run, mp.io.lammps.LammpsRun)
    assert isinstance(run.thermo, mp.io.lammps.LammpsThermo)
    assert isinstance(run.loop_time, mp.io.lammps.LammpsLoopTime)
    assert isinstance(run.performance, mp.io.lammps.LammpsPerformance)
    assert isinstance(run.memory, mp.io.lammps.LammpsMemoryUsage)
    assert isinstance(log.header, mp.io.lammps.LammpsLogHeader)


def test_thermo_columns_cross_as_float64_arrays():
    thermo = mp.io.read_lammps_log(mp.io.Text(_LOG)).runs[0].thermo
    assert thermo.columns == ["Step", "Temp", "PotEng"]
    assert thermo.rows.dtype == np.float64
    assert thermo.rows.shape == (2, 3)
    assert thermo["Step"].tolist() == [0.0, 10.0]
    assert "PotEng" in thermo
    assert "Nope" not in thermo
    assert len(thermo) == thermo.n_rows == 2


def test_scalars_and_optionals_cross_as_python_types():
    log = mp.io.read_lammps_log(mp.io.Text(_LOG))
    run = log.runs[0]
    assert run.loop_time.procs == 2 and run.loop_time.steps == 10
    assert run.loop_time.atoms == 100
    assert run.performance.atom_steps_units == "katom-step/s"
    assert run.cpu_use is None
    assert run.load_balance == []
    # The log lists every warning; the run lists the ones raised inside it.
    assert len(log.warnings) == 1 and len(run.warnings) == 1
    assert isinstance(run.warnings[0], mp.io.lammps.LammpsWarning)
    assert run.warnings[0].message == log.warnings[0].message
    assert run.warnings[0].message.startswith("test warning")


def test_to_dict_is_json_friendly():
    payload = mp.io.read_lammps_log(mp.io.Text(_LOG)).to_dict()
    json.dumps(payload)
    assert payload["runs"][0]["thermo"]["columns"] == ["Step", "Temp", "PotEng"]
    assert payload["runs"][0]["thermo"]["rows"][0] == [0.0, 300.0, -1.5]


def test_missing_file_is_file_not_found(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        mp.io.read_lammps_log(tmp_path / "nope.log")


def test_a_path_and_its_content_read_alike(tmp_path):
    path = tmp_path / "log.lammps"
    path.write_text(_LOG, encoding="utf-8")
    from_path = mp.io.read_lammps_log(path)
    from_text = mp.io.read_lammps_log(mp.io.Text(_LOG))
    from_bytes = mp.io.read_lammps_log(_LOG.encode())
    assert from_path.path == str(path)
    assert from_text.path == from_bytes.path == ""
    assert (
        from_path.runs[0].thermo.rows.tolist() == from_text.runs[0].thermo.rows.tolist()
    )
    assert from_bytes.to_dict()["runs"] == from_text.to_dict()["runs"]


def test_is_lammps_log_keys_on_the_run_the_reader_parses():
    assert mp.io.lammps.is_lammps_log(_LOG)
    banner_only = "LAMMPS (1 Jan 2026)\nERROR: Unknown command\n"
    assert not mp.io.lammps.is_lammps_log(banner_only)
    assert len(mp.io.read_lammps_log(mp.io.Text(banner_only)).runs) == 0
