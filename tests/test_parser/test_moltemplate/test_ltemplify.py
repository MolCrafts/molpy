"""Tests for ``ltemplify`` / ``write_moltemplate`` — the inverse of the reader."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from molpy.io.forcefield.moltemplate import read_moltemplate_system
from molpy.parser.moltemplate import ltemplify, write_moltemplate

FIXTURES = Path(__file__).parent / "fixtures"


def _counts(a):
    return (
        len(list(a.atoms)),
        len(list(a.bonds)),
        len(list(a.angles)),
        len(list(a.dihedrals)),
        len(list(a.impropers)),
    )


def test_roundtrip_tip3p(tmp_path):
    src = FIXTURES / "tip3p.lt"
    system, ff = read_moltemplate_system(src)
    dst = tmp_path / "tip3p_regen.lt"
    write_moltemplate(system, ff, dst)
    assert dst.exists()
    # Re-read the generated file and check counts match.
    new_system, _ = read_moltemplate_system(dst)
    assert _counts(new_system) == _counts(system)


def test_ltemplify_returns_str(tmp_path):
    src = FIXTURES / "tip3p.lt"
    system, ff = read_moltemplate_system(src)
    text = ltemplify(system, ff, class_name="Water")
    assert "Water {" in text
    assert '"Data Atoms"' in text
    assert '"Data Bonds"' in text
    assert "sys = new Water" in text


def test_cli_ltemplify(tmp_path):
    src = FIXTURES / "tip3p.lt"
    dst = tmp_path / "tip3p_regen.lt"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "molpy",
            "moltemplate",
            "ltemplify",
            str(src),
            str(dst),
            "--class-name",
            "RegenTIP3P",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert dst.exists()
    text = dst.read_text()
    assert "RegenTIP3P {" in text
    assert "sys = new RegenTIP3P" in text
