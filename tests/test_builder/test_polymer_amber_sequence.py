"""Tests for CGSmiles sequence extraction in polymer Amber builder."""

from molpy.builder.polymer.ambertools.polymer_amber import AmberPolymerBuilder


def _builder_without_init() -> AmberPolymerBuilder:
    return AmberPolymerBuilder.__new__(AmberPolymerBuilder)


def test_parse_cgsmiles_sequence_repeat() -> None:
    builder = _builder_without_init()
    seq = builder._parse_cgsmiles_sequence("{[#EO]|4}")
    assert seq == ["EO", "EO", "EO", "EO"]


def test_parse_cgsmiles_sequence_with_terminal_caps() -> None:
    builder = _builder_without_init()
    seq = builder._parse_cgsmiles_sequence("{[#MeL][#EO]|3[#MeR]}")
    assert seq == ["MeL", "EO", "EO", "EO", "MeR"]
