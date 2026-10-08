"""Tests for base Wrapper class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from molpy.wrapper import Wrapper


class MockWrapper(Wrapper):
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
    assert wrapper.process_environment().is_system


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


def test_wrapper_incomplete_env_raises_on_run():
    wrapper = MockWrapper(name="test", exe="echo", env="only-env")
    with pytest.raises(ValueError, match="incomplete"):
        wrapper.run(args=["hello"])


def test_wrapper_repr():
    """Test wrapper string representation."""

    wrapper = MockWrapper(name="test_tool", exe="test_exe")
    repr_str = repr(wrapper)
    assert "MockWrapper" in repr_str
    assert "test_tool" in repr_str
    assert "test_exe" in repr_str
