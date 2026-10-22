"""The AmberTools programs: settings from molpy's configuration, logged runs, one failure."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import molpy as mp
from molpy.tool.ambertools.programs import (
    AmberProgram,
    Antechamber,
    Parmchk2,
    Prepgen,
    Tleap,
    run_step,
)


_PROGRAMS = [
    (Antechamber, "antechamber"),
    (Parmchk2, "parmchk2"),
    (Prepgen, "prepgen"),
    (Tleap, "tleap"),
]


def _done(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(("cls", "tool"), _PROGRAMS)
def test_package_defaults(cls: type[AmberProgram], tool: str) -> None:
    program = cls()
    assert program.executable == tool
    assert program.key == f"ambertools.{tool}"
    assert program.workdir is None
    assert program.launcher == []
    assert program.env_vars == {}
    assert program.timeout is None
    assert program.environment.is_system
    assert program.settings.tool == f"ambertools.{tool}"
    assert program.settings.sources["executable"] == (
        "defaults",
        f"ambertools.{tool}.executable",
    )


@pytest.mark.parametrize(("cls", "tool"), _PROGRAMS)
def test_each_reads_its_own_table(cls: type[AmberProgram], tool: str) -> None:
    config = mp.config.load_config(
        {
            "ambertools": {
                tool: {
                    "executable": f"/opt/amber/bin/{tool}",
                    "env": "/opt/amber",
                    "env_manager": "venv",
                    "env_vars": {"AMBERHOME": "/opt/amber"},
                    "timeout": 30,
                }
            }
        }
    )
    program = cls(config=config)
    assert program.executable == f"/opt/amber/bin/{tool}"
    assert program.environment.env == Path("/opt/amber")
    assert program.environment.env_manager == "venv"
    assert program.env_vars == {"AMBERHOME": "/opt/amber"}
    assert program.timeout == 30.0


def test_the_ambertools_table_reaches_every_program() -> None:
    config = mp.config.load_config(
        {"ambertools": {"env": "AmberTools25", "env_manager": "conda"}}
    )
    for cls, tool in _PROGRAMS:
        program = cls(config=config)
        assert program.environment.env == "AmberTools25"
        assert program.settings.sources["env"] == ("run", "ambertools.env")


def test_workdir_is_a_path(tmp_path: Path) -> None:
    assert Tleap(str(tmp_path)).workdir == tmp_path


def test_run_builds_the_command_in_the_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    config = mp.config.load_config(
        {"ambertools": {"tleap": {"env_vars": {"A": "1"}, "timeout": 12}}}
    )
    program = Tleap(workdir, config=config)
    with patch("subprocess.run", return_value=_done()) as run:
        program.run(["-f", "x.in"])
    assert run.call_args.args[0] == ["tleap", "-f", "x.in"]
    assert run.call_args.kwargs["cwd"] == workdir
    assert run.call_args.kwargs["env"]["A"] == "1"
    assert run.call_args.kwargs["timeout"] == 12.0
    assert workdir.is_dir()


def test_conda_wraps_the_command() -> None:
    config = mp.config.load_config(
        {
            "conda": {"executable": "/opt/conda/bin/conda"},
            "ambertools": {"env": "AmberTools25", "env_manager": "conda"},
        }
    )
    with patch("subprocess.run", return_value=_done()) as run:
        Antechamber(config=config).run(["-h"])
    assert run.call_args.args[0] == [
        "/opt/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "AmberTools25",
        "antechamber",
        "-h",
    ]


def test_incomplete_environment_raises() -> None:
    config = mp.config.load_config({"ambertools": {"tleap": {"env": "only-env"}}})
    with pytest.raises(ValueError, match="incomplete"):
        Tleap(config=config)


def test_check_executable_names_the_setting_to_fix() -> None:
    config = mp.config.load_config(
        {"ambertools": {"tleap": {"executable": "no_such_tleap_x"}}}
    )
    with pytest.raises(
        mp.tool.ToolError, match="ambertools.tleap.executable"
    ) as raised:
        Tleap(config=config).check_executable()
    assert raised.value.tool == "ambertools.tleap"
    assert raised.value.returncode is None


def test_repr() -> None:
    text = repr(Tleap(Path("w")))
    assert "<Tleap(" in text
    assert "tleap" in text


class TestFailureContract:
    """Every failure is a ToolError carrying the tool, exit code and output tail."""

    def test_a_non_zero_exit_raises_by_default(self, tmp_path) -> None:
        stderr = "".join(f"line {i}\n" for i in range(30))
        with (
            patch("subprocess.run", return_value=_done(4, stderr=stderr)),
            pytest.raises(mp.tool.ToolError) as raised,
        ):
            Tleap(tmp_path).run(["-f", "x.in"])
        error = raised.value
        assert isinstance(error, RuntimeError)
        assert (error.tool, error.returncode) == ("ambertools.tleap", 4)
        assert error.tail.splitlines() == [f"line {i}" for i in range(10, 30)]
        assert "exited with status 4" in str(error)

    def test_stdout_is_the_tail_when_stderr_is_empty(self, tmp_path) -> None:
        with (
            patch("subprocess.run", return_value=_done(1, stdout="FATAL: no unit")),
            pytest.raises(mp.tool.ToolError, match="FATAL: no unit"),
        ):
            Tleap(tmp_path).run([])

    def test_check_false_returns_the_process(self, tmp_path) -> None:
        with patch("subprocess.run", return_value=_done(1)):
            assert Tleap(tmp_path).run([], check=False).returncode == 1

    def test_a_missing_program_is_a_tool_error(self, tmp_path) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("tleap")),
            pytest.raises(mp.tool.ToolError, match="could not start") as raised,
        ):
            Tleap(tmp_path).run([])
        assert raised.value.returncode is None

    def test_a_timeout_is_a_tool_error(self, tmp_path) -> None:
        config = mp.config.load_config({"ambertools": {"timeout": 1}})
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("t", 1)),
            pytest.raises(mp.tool.ToolError, match="timed out after 1 s"),
        ):
            Tleap(tmp_path, config=config).run([])


class TestRunStep:
    def test_a_missing_executable_refuses_to_start(self, tmp_path) -> None:
        with (
            patch.object(mp.tool.Tool, "resolve_executable", return_value=None),
            pytest.raises(mp.tool.ToolError, match="not found"),
        ):
            run_step(Tleap(tmp_path), tmp_path / "out", lambda: _done())

    def test_a_missing_output_is_a_tool_error(self, tmp_path, molpy_records) -> None:
        program = Tleap(tmp_path)
        with (
            patch.object(mp.tool.Tool, "resolve_executable", return_value="/bin/tleap"),
            pytest.raises(mp.tool.ToolError, match="out.prmtop unwritten") as raised,
        ):
            run_step(program, tmp_path / "out.prmtop", lambda: _done(stdout="warn"))
        assert raised.value.returncode == 0
        assert raised.value.tail == "warn"
        (record,) = (r for r in molpy_records if r.message == "step output missing")
        assert record.logger_name == "molpy.ambertools.tleap"
        assert record.extra["output"] == str(tmp_path / "out.prmtop")


class TestLogging:
    def test_a_run_logs_start_and_finish(self, tmp_path, molpy_records) -> None:
        with patch("subprocess.run", return_value=_done()):
            Parmchk2(tmp_path).run(["-i", "a.mol2"])
        started, finished = (
            r for r in molpy_records if r.logger_name.startswith("molpy.ambertools")
        )
        assert (
            started.logger_name == finished.logger_name == "molpy.ambertools.parmchk2"
        )
        assert started.message == "process started"
        assert finished.message == "process finished"
        assert finished.extra["command"] == ["parmchk2", "-i", "a.mol2"]
        assert finished.extra["cwd"] == str(tmp_path)
        assert finished.extra["returncode"] == 0
        assert finished.extra["elapsed_s"] >= 0.0

    @pytest.mark.parametrize(("cls", "tool"), _PROGRAMS)
    def test_each_logs_under_its_namespace(
        self, cls, tool, tmp_path, molpy_records
    ) -> None:
        with patch("subprocess.run", return_value=_done(1)):
            cls(tmp_path).run([], check=False)
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert failed.logger_name == f"molpy.ambertools.{tool}"
        assert str(failed.level) == "ERROR"
        assert failed.extra["returncode"] == 1

    def test_a_timeout_is_logged(self, tmp_path, molpy_records) -> None:
        config = mp.config.load_config({"ambertools": {"timeout": 1}})
        expired = subprocess.TimeoutExpired("tleap", 1)
        with (
            patch("subprocess.run", side_effect=expired),
            pytest.raises(mp.tool.ToolError),
        ):
            Tleap(tmp_path, config=config).run([])
        (record,) = (r for r in molpy_records if r.message == "process timed out")
        assert record.extra["timeout_s"] == 1.0
