"""Tests for Tleap."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from molpy.tool.ambertools.programs import Tleap


def test_tleap_initialization():
    """Test Tleap initialization."""

    program = Tleap(Path("tmp_tleap"))
    assert program.executable == "tleap"
    assert program.workdir == Path("tmp_tleap")


def test_tleap_default_exe():
    """Test that the executable defaults to 'tleap'."""

    program = Tleap()
    assert program.executable == "tleap"


def test_tleap_run_from_script(tmp_path: Path):
    """Test run_from_script() method."""

    workdir = tmp_path / "test_workdir"
    program = Tleap(workdir)

    script_text = "source leaprc.gaff\nquit\n"

    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.write_text") as mock_write,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        program.run_from_script(script_text=script_text, script_name="test.in")

        mock_write.assert_called_once_with(script_text, encoding="utf-8")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["tleap", "-f", "test.in"]


def test_tleap_run_from_script_default_name(tmp_path: Path):
    """Test run_from_script() with default script name."""

    workdir = tmp_path / "test_workdir"
    program = Tleap(workdir)

    script_text = "quit\n"

    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.write_text"),
    ):
        mock_run.return_value.returncode = 0
        program.run_from_script(script_text=script_text)

        call_args = mock_run.call_args[0][0]
        assert call_args == ["tleap", "-f", "tleap.in"]


def test_tleap_run_from_script_no_workdir():
    """Test run_from_script() raises error when no workdir is available."""

    program = Tleap()

    with pytest.raises(ValueError, match="requires a working directory"):
        program.run_from_script(script_text="quit\n")
