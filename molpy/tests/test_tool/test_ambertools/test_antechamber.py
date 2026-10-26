"""Tests for Antechamber."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import molpy as mp
from molpy.tool.ambertools.programs import Antechamber


def test_antechamber_initialization():
    """Test Antechamber initialization."""

    program = Antechamber(Path("tmp_ante"))
    assert program.executable == "antechamber"
    assert program.workdir == Path("tmp_ante")


def test_antechamber_default_exe():
    """Test that the executable defaults to 'antechamber'."""

    program = Antechamber()
    assert program.executable == "antechamber"


def test_antechamber_run(tmp_path: Path):
    """Test run() with raw arguments."""

    program = Antechamber(tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        args = ["-i", "lig.mol2", "-fi", "mol2", "-o", "out.mol2", "-fo", "mol2"]
        program.run(args)

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


def test_antechamber_atomtype_assign(tmp_path: Path):
    """Test atomtype_assign() method."""

    program = Antechamber(tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        program.atomtype_assign(
            input_file="tfsi.pdb",
            output_file="tfsi_gaff2.mol2",
            input_format="pdb",
            output_format="mol2",
            charge_method="bcc",
            parameter_set="gaff2",
            net_charge=-1,
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == [
            "antechamber",
            "-i",
            "tfsi.pdb",
            "-fi",
            "pdb",
            "-o",
            "tfsi_gaff2.mol2",
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-at",
            "gaff2",
            "-nc",
            "-1",
        ]


def test_a_failing_atomtype_assign_raises(tmp_path: Path):
    """check=True is the default: a non-zero exit is a ToolError, not a value."""
    program = Antechamber(tmp_path)
    failed = subprocess.CompletedProcess([], 2, "", "Error: cannot open tfsi.pdb\n")
    with (
        patch("subprocess.run", return_value=failed),
        pytest.raises(mp.tool.ToolError, match="cannot open tfsi.pdb") as raised,
    ):
        program.atomtype_assign("tfsi.pdb", "tfsi.mol2", parameter_set="gaff2")
    assert raised.value.tool == "ambertools.antechamber"
    assert raised.value.returncode == 2
    assert raised.value.tail == "Error: cannot open tfsi.pdb"


def test_check_false_returns_the_failed_process(tmp_path: Path):
    program = Antechamber(tmp_path)
    failed = subprocess.CompletedProcess([], 2, "", "")
    with patch("subprocess.run", return_value=failed):
        assert (
            program.atomtype_assign(
                "a.pdb", "a.mol2", parameter_set="gaff2", check=False
            ).returncode
            == 2
        )
