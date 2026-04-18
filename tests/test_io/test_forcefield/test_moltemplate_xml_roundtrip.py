"""End-to-end: .lt -> ForceField -> XML -> re-parse."""

from __future__ import annotations

from pathlib import Path

import pytest

from molpy.core.forcefield import AtomStyle, AtomType
from molpy.io.forcefield.moltemplate import read_moltemplate
from molpy.io.forcefield.xml import XMLForceFieldWriter, read_xml_forcefield

FIXTURES = (
    Path(__file__).parent.parent.parent
    / "test_parser"
    / "test_moltemplate"
    / "fixtures"
)


@pytest.fixture
def tip3p_xml(tmp_path):
    ff = read_moltemplate(FIXTURES / "tip3p.lt")
    out = tmp_path / "tip3p.xml"
    XMLForceFieldWriter(out).write(ff)
    return out


def test_xml_file_is_written(tip3p_xml):
    assert tip3p_xml.exists()
    text = tip3p_xml.read_text()
    assert "<ForceField" in text
    assert "<AtomTypes>" in text
    assert "<HarmonicBondForce>" in text
    assert "<HarmonicAngleForce>" in text


def test_xml_atomtypes_named(tip3p_xml):
    text = tip3p_xml.read_text()
    # After builder fix, O and H atom types from Data Masses should emit name="O" / name="H"
    assert 'name="O"' in text
    assert 'name="H"' in text


def test_xml_reparse_recovers_styles(tip3p_xml):
    ff2 = read_xml_forcefield(tip3p_xml)
    astyle = ff2.get_style_by_name("full", AtomStyle)
    assert astyle is not None
    by_name = {t.name: t for t in astyle.types.bucket(AtomType)}
    assert "O" in by_name
    assert "H" in by_name
