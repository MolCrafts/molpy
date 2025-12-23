"""Tests for AntechamberWrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from molpy.wrapper import AntechamberWrapper


def test_antechamber_wrapper_initialization():
    """Test AntechamberWrapper initialization."""

    wrapper = AntechamberWrapper(name="antechamber", workdir=Path("tmp_ante"))
    assert wrapper.name == "antechamber"
    assert wrapper.exe == "antechamber"
    assert wrapper.workdir == Path("tmp_ante")


def test_antechamber_wrapper_default_exe():
    """Test that exe defaults to 'antechamber'."""

    wrapper = AntechamberWrapper(name="ante")
    assert wrapper.exe == "antechamber"


def test_antechamber_wrapper_run_raw(tmp_path: Path):
    """Test run_raw() method."""

    wrapper = AntechamberWrapper(name="ante", workdir=tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        args = ["-i", "lig.mol2", "-fi", "mol2", "-o", "out.mol2", "-fo", "mol2"]
        wrapper.run_raw(args=args)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == [
            "antechamber",
            "-i",
            "lig.mol2",
            "-fi",
            "mol2",
            "-o",
            "out.mol2",
            "-fo",
            "mol2",
        ]


def test_antechamber_wrapper_run_raw_with_cwd(tmp_path: Path):
    """Test run_raw() with cwd override."""

    wrapper = AntechamberWrapper(name="ante", workdir=tmp_path / "default")
    override_cwd = tmp_path / "override"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        wrapper.run_raw(["-i", "test.mol2"], cwd=override_cwd)

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == str(override_cwd)
