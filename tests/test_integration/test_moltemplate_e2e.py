"""End-to-end moltemplate execution + multi-engine emission."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from molpy.cli import main

FIXTURES = (
    Path(__file__).parent.parent / "test_parser" / "test_moltemplate" / "fixtures"
)


@pytest.fixture
def water_script() -> Path:
    return FIXTURES / "tip3p.lt"


def test_run_all_engines_produces_usable_outputs(tmp_path, water_script):
    rc = main(
        [
            "moltemplate",
            "run",
            str(water_script),
            "--emit",
            "all",
            "--out-dir",
            str(tmp_path),
            "--prefix",
            "w",
        ]
    )
    assert rc == 0

    # LAMMPS input set present
    for fname in ("w.data", "w.in.settings", "w.in.init", "w.in"):
        assert (tmp_path / fname).exists()

    # OpenMM script parses as Python
    py = (tmp_path / "w.py").read_text()
    ast.parse(py)

    # GROMACS mdp templates present and non-empty
    for fname in ("em.mdp", "nvt.mdp"):
        assert (tmp_path / fname).stat().st_size > 0

    # XML FF contains a ForceField root
    xml_text = (tmp_path / "w.xml").read_text()
    assert "<ForceField" in xml_text


def test_convert_and_reparse_round_trip(tmp_path, water_script):
    out_xml = tmp_path / "tip3p.xml"
    rc = main(["moltemplate", "convert", str(water_script), str(out_xml)])
    assert rc == 0
    assert out_xml.exists()

    # Re-parse to confirm the XML is schema-compatible.
    from molpy.core.forcefield import AtomStyle, AtomType
    from molpy.io.forcefield.xml import read_xml_forcefield

    ff = read_xml_forcefield(out_xml)
    astyle = ff.get_style_by_name("full", AtomStyle)
    assert astyle is not None
    names = {t.name for t in astyle.types.bucket(AtomType)}
    assert {"O", "H"} <= names


def test_editing_primitives_on_built_system(water_script):
    """Phase-1 editing API usable on a moltemplate-built system."""
    from molpy.core.atomistic import Atom
    from molpy.io.forcefield.moltemplate import read_moltemplate_system

    atomistic, ff = read_moltemplate_system(water_script)
    n = atomistic.rename_type("O", "OW", kind=Atom)
    assert n == 2  # two water molecules
    # Property setter
    n2 = atomistic.set_property(lambda a: a.get("type") == "H", "mass", 1.008)
    assert n2 == 4
