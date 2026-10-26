"""Tests for Parmchk2."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from molpy.tool.ambertools.programs import Parmchk2


def test_parmchk2_initialization():
    """Test Parmchk2 initialization."""

    program = Parmchk2(Path("tmp_parmchk"))
    assert program.executable == "parmchk2"
    assert program.workdir == Path("tmp_parmchk")


def test_parmchk2_default_exe():
    """Test that the executable defaults to 'parmchk2'."""

    program = Parmchk2()
    assert program.executable == "parmchk2"


def test_parmchk2_run(tmp_path: Path):
    """Test run() with raw arguments."""

    program = Parmchk2(tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        args = ["-i", "lig.mol2", "-f", "mol2", "-o", "lig.frcmod"]
        program.run(args)

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


def test_parmchk2_generate_parameters(tmp_path: Path):
    """Test generate_parameters() method."""

    program = Parmchk2(tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        program.generate_parameters(
            input_file="tfsi_gaff2.mol2",
            output_file="tfsi.frcmod",
            input_format="mol2",
            parameter_set="gaff2",
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
            "-s",
            "gaff2",
        ]
