"""Tests for base Wrapper class."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from molpy.wrapper import Wrapper


class MockWrapper(Wrapper):  # noqa: D101
    """Mock implementation of Wrapper for testing."""

    def __init__(self, name: str = "test", exe: str = "test_exe", **kwargs):
        super().__init__(name=name, exe=exe, **kwargs)


def test_wrapper_initialization():
    """Test basic wrapper initialization."""

    wrapper = MockWrapper(name="test_tool", exe="test_exe")
    assert wrapper.name == "test_tool"
    assert wrapper.exe == "test_exe"
    assert wrapper.workdir is None
    assert wrapper.env_vars == {}
    assert getattr(wrapper, "env", None) is None
    assert getattr(wrapper, "env_manager", None) is None


def test_wrapper_with_workdir(tmp_path: Path):
    """Test wrapper with working directory."""

    workdir = tmp_path / "test_workdir"
    wrapper = MockWrapper(name="test", exe="test_exe", workdir=workdir)
    assert wrapper.workdir == workdir


def test_wrapper_with_env():
    """Test wrapper with environment variables."""

    env_vars = {"TEST_VAR": "test_value"}
    wrapper = MockWrapper(name="test", exe="test_exe", env_vars=env_vars)
    assert wrapper.env_vars == env_vars


def test_wrapper_run_basic():
    """Test basic run() method."""

    wrapper = MockWrapper(name="test", exe="echo")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "hello"
        mock_run.return_value.stderr = ""

        wrapper.run(args=["hello"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["echo", "hello"]


def test_wrapper_run_with_workdir(tmp_path: Path):
    """Test run() with working directory."""

    workdir = tmp_path / "test_workdir"
    wrapper = MockWrapper(name="test", exe="echo", workdir=workdir)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        wrapper.run(args=["test"])

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == str(workdir)


def test_wrapper_run_with_cwd_override(tmp_path: Path):
    """Test run() with cwd parameter overriding workdir."""

    workdir = tmp_path / "default_workdir"
    override_cwd = tmp_path / "override_workdir"
    wrapper = MockWrapper(name="test", exe="echo", workdir=workdir)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        wrapper.run(args=["test"], cwd=override_cwd)

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == str(override_cwd)


def test_wrapper_run_with_conda_env_name_prefixes_command():
    """If env/env_manager are set to a conda env name, wrapper.run() should use conda run -n."""

    wrapper = MockWrapper(
        name="test", exe="echo", env="AmberTools25", env_manager="conda"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        wrapper.run(args=["hello"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        argv = call_args[0][0]
        assert argv[1:4] == ["run", "-n", "AmberTools25"]
        assert argv[4:] == ["echo", "hello"]


def test_wrapper_run_with_conda_env_prefix_prefixes_command():
    """If env/env_manager are set to a conda prefix path, wrapper.run() should use conda run -p."""

    wrapper = MockWrapper(
        name="test",
        exe="echo",
        env="/opt/conda/envs/AmberTools25",
        env_manager="conda",
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        wrapper.run(args=["hello"])

        mock_run.assert_called_once()
        argv = mock_run.call_args[0][0]
        assert argv[1:4] == ["run", "-p", "/opt/conda/envs/AmberTools25"]
        assert argv[4:] == ["echo", "hello"]


def test_wrapper_run_with_pip_env_injects_virtualenv_path(tmp_path: Path):
    """If env_manager is pip/venv, Wrapper should inject PATH/VIRTUAL_ENV and not use conda run."""

    venv_prefix = tmp_path / "venv"
    wrapper = MockWrapper(name="test", exe="echo", env=venv_prefix, env_manager="pip")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        wrapper.run(args=["hello"])

        argv = mock_run.call_args[0][0]
        assert argv == ["echo", "hello"]

        call_kwargs = mock_run.call_args[1]
        env = call_kwargs["env"]
        assert env["VIRTUAL_ENV"] == str(venv_prefix)
        expected_bin = venv_prefix / ("Scripts" if os.name == "nt" else "bin")
        assert env["PATH"].split(os.pathsep)[0] == str(expected_bin)


def test_wrapper_repr():
    """Test wrapper string representation."""

    wrapper = MockWrapper(name="test_tool", exe="test_exe")
    repr_str = repr(wrapper)
    assert "MockWrapper" in repr_str
    assert "test_tool" in repr_str
    assert "test_exe" in repr_str
