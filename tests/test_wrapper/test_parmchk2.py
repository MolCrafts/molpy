"""Tests for Parmchk2Wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from molpy.wrapper import Parmchk2Wrapper


def test_parmchk2_wrapper_initialization():
    """Test Parmchk2Wrapper initialization."""

    wrapper = Parmchk2Wrapper(name="parmchk", workdir=Path("tmp_parmchk"))
    assert wrapper.name == "parmchk"
    assert wrapper.exe == "parmchk2"
    assert wrapper.workdir == Path("tmp_parmchk")


def test_parmchk2_wrapper_default_exe():
    """Test that exe defaults to 'parmchk2'."""

    wrapper = Parmchk2Wrapper(name="parmchk")
    assert wrapper.exe == "parmchk2"


def test_parmchk2_wrapper_run_raw(tmp_path: Path):
    """Test run_raw() method."""

    wrapper = Parmchk2Wrapper(name="parmchk", workdir=tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        args = ["-i", "lig.mol2", "-f", "mol2", "-o", "lig.frcmod"]
        wrapper.run_raw(args=args)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == [
            "parmchk2",
            "-i",
            "lig.mol2",
            "-f",
            "mol2",
            "-o",
            "lig.frcmod",
        ]


def test_parmchk2_wrapper_generate_parameters(tmp_path: Path):
    """Test generate_parameters() method."""

    wrapper = Parmchk2Wrapper(name="parmchk", workdir=tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        wrapper.generate_parameters(
            input_file="tfsi_gaff2.mol2",
            output_file="tfsi.frcmod",
            input_format="mol2",
            parameter_level=2,
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == [
            "parmchk2",
            "-i",
            "tfsi_gaff2.mol2",
            "-f",
            "mol2",
            "-o",
            "tfsi.frcmod",
            "-p",
            "2",
        ]
