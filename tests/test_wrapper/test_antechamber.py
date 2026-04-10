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


def test_antechamber_wrapper_atomtype_assign(tmp_path: Path):
    """Test atomtype_assign() method."""

    wrapper = AntechamberWrapper(name="ante", workdir=tmp_path / "work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        wrapper.atomtype_assign(
            input_file="tfsi.pdb",
            output_file="tfsi_gaff2.mol2",
            input_format="pdb",
            output_format="mol2",
            charge_method="bcc",
            atom_type="gaff2",
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
