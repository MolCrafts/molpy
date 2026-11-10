"""``read_psmiles`` across the seam: a PSMILES repeat unit becomes the
fragment IR whose two ``*`` are ``$`` descriptors, and its refusals cross as
``SmilesError`` stamped ``"psmiles"``.

The parsing rules themselves are pinned by the Rust unit tests in
``crates/molrs/src/line_notation/psmiles.rs``; these tests prove only that
Python reaches them. Fixtures are inline strings.
"""

from __future__ import annotations

import molpy as mp
import pytest


def test_stars_cross_as_dollar_descriptors() -> None:
    mol, sites = mp.io.smiles.read_psmiles("[*]CC([*])c1ccccc1").to_atomistic()
    assert mol.n_atoms == 8
    assert [descriptor.kind for _, descriptor in sites] == ["$", "$"]


def test_unit_builds_a_two_port_template() -> None:
    template = mp.io.smiles.read_psmiles("*CC*").to_template()
    assert template.n_ports == 2


def test_star_count_refusal_is_a_psmiles_error() -> None:
    with pytest.raises(mp.io.smiles.SmilesError) as excinfo:
        mp.io.smiles.read_psmiles("*CC")
    assert excinfo.value.kind == "PsmilesStarCount"
    assert excinfo.value.notation == "psmiles"


def test_bridging_star_refusal_is_a_psmiles_error() -> None:
    with pytest.raises(mp.io.smiles.SmilesError) as excinfo:
        mp.io.smiles.read_psmiles("*C*C")
    assert excinfo.value.kind == "PsmilesStarNotTerminal"
