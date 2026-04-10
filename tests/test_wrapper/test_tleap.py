"""Tests for TLeapWrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from molpy.wrapper import TLeapWrapper


def test_tleap_wrapper_initialization():
    """Test TLeapWrapper initialization."""

    wrapper = TLeapWrapper(name="tleap", workdir=Path("tmp_tleap"))
    assert wrapper.name == "tleap"
    assert wrapper.exe == "tleap"
    assert wrapper.workdir == Path("tmp_tleap")


def test_tleap_wrapper_default_exe():
    """Test that exe defaults to 'tleap'."""

    wrapper = TLeapWrapper(name="tleap")
    assert wrapper.exe == "tleap"


def test_tleap_wrapper_run_from_script(tmp_path: Path):
    """Test run_from_script() method."""

    workdir = tmp_path / "test_workdir"
    wrapper = TLeapWrapper(name="tleap", workdir=workdir)

    script_text = "source leaprc.gaff\nquit\n"

    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.write_text") as mock_write,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        wrapper.run_from_script(script_text=script_text, script_name="test.in")

        mock_write.assert_called_once_with(script_text)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["tleap", "-f", "test.in"]


def test_tleap_wrapper_run_from_script_default_name(tmp_path: Path):
    """Test run_from_script() with default script name."""

    workdir = tmp_path / "test_workdir"
    wrapper = TLeapWrapper(name="tleap", workdir=workdir)

    script_text = "quit\n"

    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.write_text") as mock_write,
    ):
        mock_run.return_value.returncode = 0
        wrapper.run_from_script(script_text=script_text)

        call_args = mock_run.call_args[0][0]
        assert call_args == ["tleap", "-f", "tleap.in"]


def test_tleap_wrapper_run_from_script_no_workdir():
    """Test run_from_script() raises error when no workdir is available."""

    wrapper = TLeapWrapper(name="tleap", workdir=None)

    with pytest.raises(ValueError, match="requires a working directory"):
        wrapper.run_from_script(script_text="quit\n")
