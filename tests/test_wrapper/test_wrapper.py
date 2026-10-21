"""The Wrapper base: settings from molpy's configuration, logged runs, one failure."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from molpy.config import load_config
from molpy.wrapper import (
    AntechamberWrapper,
    Parmchk2Wrapper,
    PrepgenWrapper,
    SanderWrapper,
    TleapWrapper,
    Tool,
    ToolError,
    Wrapper,
    run_step,
)

_WRAPPERS = [
    (AntechamberWrapper, "antechamber"),
    (Parmchk2Wrapper, "parmchk2"),
    (PrepgenWrapper, "prepgen"),
    (SanderWrapper, "sander"),
    (TleapWrapper, "tleap"),
]


def _done(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(("cls", "tool"), _WRAPPERS)
def test_package_defaults(cls: type[Wrapper], tool: str) -> None:
    wrapper = cls()
    assert wrapper.executable == tool
    assert wrapper.key == f"wrapper.{tool}"
    assert wrapper.workdir is None
    assert wrapper.launcher == []
    assert wrapper.env_vars == {}
    assert wrapper.timeout is None
    assert wrapper.environment.is_system
    assert wrapper.settings.tool == f"wrapper.{tool}"
    assert wrapper.settings.sources["executable"] == (
        "defaults",
        f"wrapper.{tool}.executable",
    )


@pytest.mark.parametrize(("cls", "tool"), _WRAPPERS)
def test_each_wrapper_reads_its_own_table(cls: type[Wrapper], tool: str) -> None:
    config = load_config(
        {
            "wrapper": {
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
    wrapper = cls(config=config)
    assert wrapper.executable == f"/opt/amber/bin/{tool}"
    assert wrapper.environment.env == Path("/opt/amber")
    assert wrapper.environment.env_manager == "venv"
    assert wrapper.env_vars == {"AMBERHOME": "/opt/amber"}
    assert wrapper.timeout == 30.0


def test_the_wrapper_table_reaches_every_ambertools_program() -> None:
    config = load_config({"wrapper": {"env": "AmberTools25", "env_manager": "conda"}})
    for cls, tool in _WRAPPERS:
        wrapper = cls(config=config)
        assert wrapper.environment.env == "AmberTools25"
        assert wrapper.settings.sources["env"] == ("run", "wrapper.env")


class TestThirdPartyWrapper:
    """A Wrapper subclass outside molpy resolves its settings like a built-in."""

    class GmxWrapper(Wrapper):
        tool = "gmx"

        def _default_executable(self) -> str:
            return "gmx"

    def test_it_reads_its_own_table(self) -> None:
        config = load_config(
            {"wrapper": {"timeout": 5, "gmx": {"executable": "gmx_mpi"}}}
        )
        wrapper = self.GmxWrapper(config=config)
        assert wrapper.executable == "gmx_mpi"
        assert wrapper.timeout == 5.0
        assert wrapper.settings.sources["executable"] == (
            "run",
            "wrapper.gmx.executable",
        )

    def test_without_a_table_it_takes_its_default(self) -> None:
        wrapper = self.GmxWrapper()
        assert wrapper.executable == "gmx"
        assert "executable" not in wrapper.settings.sources

    def test_without_a_default_it_names_the_setting(self) -> None:
        class Bare(Wrapper):
            tool = "bare"

        with pytest.raises(ValueError, match="wrapper.bare.executable"):
            Bare()


def test_workdir_is_a_path(tmp_path: Path) -> None:
    assert TleapWrapper(str(tmp_path)).workdir == tmp_path


def test_run_builds_the_command_in_the_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    config = load_config(
        {"wrapper": {"tleap": {"env_vars": {"A": "1"}, "timeout": 12}}}
    )
    wrapper = TleapWrapper(workdir, config=config)
    with patch("subprocess.run", return_value=_done()) as run:
        wrapper.run(["-f", "x.in"])
    assert run.call_args.args[0] == ["tleap", "-f", "x.in"]
    assert run.call_args.kwargs["cwd"] == workdir
    assert run.call_args.kwargs["env"]["A"] == "1"
    assert run.call_args.kwargs["timeout"] == 12.0
    assert workdir.is_dir()


def test_conda_wraps_the_command() -> None:
    config = load_config(
        {
            "conda": {"executable": "/opt/conda/bin/conda"},
            "wrapper": {"env": "AmberTools25", "env_manager": "conda"},
        }
    )
    with patch("subprocess.run", return_value=_done()) as run:
        AntechamberWrapper(config=config).run(["-h"])
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
    config = load_config({"wrapper": {"tleap": {"env": "only-env"}}})
    with pytest.raises(ValueError, match="incomplete"):
        TleapWrapper(config=config)


def test_check_executable_names_the_setting_to_fix() -> None:
    config = load_config({"wrapper": {"tleap": {"executable": "no_such_tleap_x"}}})
    with pytest.raises(ToolError, match="wrapper.tleap.executable") as raised:
        TleapWrapper(config=config).check_executable()
    assert raised.value.tool == "wrapper.tleap"
    assert raised.value.returncode is None


def test_repr() -> None:
    text = repr(TleapWrapper(Path("w")))
    assert "TleapWrapper" in text
    assert "tleap" in text


class TestFailureContract:
    """Every failure is a ToolError carrying the tool, exit code and output tail."""

    def test_a_non_zero_exit_raises_by_default(self, tmp_path) -> None:
        stderr = "".join(f"line {i}\n" for i in range(30))
        with (
            patch("subprocess.run", return_value=_done(4, stderr=stderr)),
            pytest.raises(ToolError) as raised,
        ):
            TleapWrapper(tmp_path).run(["-f", "x.in"])
        error = raised.value
        assert isinstance(error, RuntimeError)
        assert (error.tool, error.returncode) == ("wrapper.tleap", 4)
        assert error.tail.splitlines() == [f"line {i}" for i in range(10, 30)]
        assert "exited with status 4" in str(error)

    def test_stdout_is_the_tail_when_stderr_is_empty(self, tmp_path) -> None:
        with (
            patch("subprocess.run", return_value=_done(1, stdout="FATAL: no unit")),
            pytest.raises(ToolError, match="FATAL: no unit"),
        ):
            TleapWrapper(tmp_path).run([])

    def test_check_false_returns_the_process(self, tmp_path) -> None:
        with patch("subprocess.run", return_value=_done(1)):
            assert TleapWrapper(tmp_path).run([], check=False).returncode == 1

    def test_a_missing_program_is_a_tool_error(self, tmp_path) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("tleap")),
            pytest.raises(ToolError, match="could not start") as raised,
        ):
            TleapWrapper(tmp_path).run([])
        assert raised.value.returncode is None

    def test_a_timeout_is_a_tool_error(self, tmp_path) -> None:
        config = load_config({"wrapper": {"timeout": 1}})
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("t", 1)),
            pytest.raises(ToolError, match="timed out after 1 s"),
        ):
            TleapWrapper(tmp_path, config=config).run([])


class TestRunStep:
    def test_a_missing_executable_refuses_to_start(self, tmp_path) -> None:
        with (
            patch.object(Tool, "resolve_executable", return_value=None),
            pytest.raises(ToolError, match="not found"),
        ):
            run_step(TleapWrapper(tmp_path), tmp_path / "out", lambda: _done())

    def test_a_missing_output_is_a_tool_error(self, tmp_path, molpy_records) -> None:
        wrapper = TleapWrapper(tmp_path)
        with (
            patch.object(Tool, "resolve_executable", return_value="/bin/tleap"),
            pytest.raises(ToolError, match="out.prmtop unwritten") as raised,
        ):
            run_step(wrapper, tmp_path / "out.prmtop", lambda: _done(stdout="warn"))
        assert raised.value.returncode == 0
        assert raised.value.tail == "warn"
        (record,) = (r for r in molpy_records if r.message == "step output missing")
        assert record.logger_name == "molpy.wrapper.tleap"
        assert record.extra["output"] == str(tmp_path / "out.prmtop")


class TestLogging:
    def test_a_run_logs_start_and_finish(self, tmp_path, molpy_records) -> None:
        with patch("subprocess.run", return_value=_done()):
            Parmchk2Wrapper(tmp_path).run(["-i", "a.mol2"])
        started, finished = (
            r for r in molpy_records if r.logger_name.startswith("molpy.wrapper")
        )
        assert started.logger_name == finished.logger_name == "molpy.wrapper.parmchk2"
        assert started.message == "process started"
        assert finished.message == "process finished"
        assert finished.extra["command"] == ["parmchk2", "-i", "a.mol2"]
        assert finished.extra["cwd"] == str(tmp_path)
        assert finished.extra["returncode"] == 0
        assert finished.extra["elapsed_s"] >= 0.0

    @pytest.mark.parametrize(("cls", "tool"), _WRAPPERS)
    def test_each_wrapper_logs_under_its_namespace(
        self, cls, tool, tmp_path, molpy_records
    ) -> None:
        with patch("subprocess.run", return_value=_done(1)):
            cls(tmp_path).run([], check=False)
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert failed.logger_name == f"molpy.wrapper.{tool}"
        assert str(failed.level) == "ERROR"
        assert failed.extra["returncode"] == 1

    def test_a_timeout_is_logged(self, tmp_path, molpy_records) -> None:
        config = load_config({"wrapper": {"timeout": 1}})
        expired = subprocess.TimeoutExpired("tleap", 1)
        with (
            patch("subprocess.run", side_effect=expired),
            pytest.raises(ToolError),
        ):
            TleapWrapper(tmp_path, config=config).run([])
        (record,) = (r for r in molpy_records if r.message == "process timed out")
        assert record.extra["timeout_s"] == 1.0
