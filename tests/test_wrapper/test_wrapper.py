"""The Wrapper base: settings from molpy's configuration, logged runs."""

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


def _done(returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout="", stderr="")


@pytest.mark.parametrize(("cls", "tool"), _WRAPPERS)
def test_package_defaults(cls: type[Wrapper], tool: str) -> None:
    wrapper = cls()
    assert wrapper.exe == tool
    assert wrapper.workdir is None
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
    assert wrapper.exe == f"/opt/amber/bin/{tool}"
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
        "-n",
        "AmberTools25",
        "antechamber",
        "-h",
    ]


def test_incomplete_environment_raises() -> None:
    config = load_config({"wrapper": {"tleap": {"env": "only-env"}}})
    with pytest.raises(ValueError, match="incomplete"):
        TleapWrapper(config=config)


def test_check_names_the_setting_to_fix() -> None:
    config = load_config({"wrapper": {"tleap": {"executable": "no_such_tleap_x"}}})
    with pytest.raises(FileNotFoundError, match="wrapper.tleap.executable"):
        TleapWrapper(config=config).check()


def test_repr() -> None:
    text = repr(TleapWrapper(Path("w")))
    assert "TleapWrapper" in text
    assert "tleap" in text


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
            cls(tmp_path).run([])
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert failed.logger_name == f"molpy.wrapper.{tool}"
        assert str(failed.level) == "ERROR"
        assert failed.extra["returncode"] == 1

    def test_a_timeout_is_logged(self, tmp_path, molpy_records) -> None:
        config = load_config({"wrapper": {"timeout": 1}})
        expired = subprocess.TimeoutExpired("tleap", 1)
        with (
            patch("subprocess.run", side_effect=expired),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            TleapWrapper(tmp_path, config=config).run([])
        (record,) = (r for r in molpy_records if r.message == "process timed out")
        assert record.extra["timeout_s"] == 1.0

    def test_run_step_logs_a_missing_output(self, tmp_path, molpy_records) -> None:
        wrapper = TleapWrapper(tmp_path)
        with (
            patch.object(Wrapper, "is_available", return_value=True),
            pytest.raises(RuntimeError, match="missing"),
        ):
            run_step(wrapper, tmp_path / "out.prmtop", lambda: _done())
        (record,) = (r for r in molpy_records if r.message == "step output missing")
        assert record.logger_name == "molpy.wrapper.tleap"
        assert record.extra["output"] == str(tmp_path / "out.prmtop")
