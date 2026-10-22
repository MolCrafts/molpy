"""``write_openmm_system``: a force field plus the residue templates of a
typed frame, which OpenMM's ``createSystem`` needs for the PDB
``write_pdb`` writes of the same frame.

The tests check the written structure: one template per molecule, named
as the PDB names it, and a refusal for an atom type the force field lacks.
"""

from __future__ import annotations

from conftest import capture

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import molpy as mp
import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[3] / "crates/molrs/src/ff/testdata/openmm"
OFFSET = 50.0  # Å between copies


def _molecules(name: str, copies: int) -> tuple[mp.core.Frame, dict]:
    """``copies`` of the ``name`` fixture's molecule, typed and charged, with
    no names and no residue columns, each copy ``OFFSET`` Å further along x."""
    case = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    n = len(case["types"])
    positions = np.concatenate(
        [np.asarray(case["positions"]) + [OFFSET * c, 0.0, 0.0] for c in range(copies)]
    )
    atoms = mp.core.Block()
    for k, key in enumerate("xyz"):
        atoms.insert(key, positions[:, k].copy())
    atoms.insert("type", case["types"] * copies)
    atoms.insert("element", case["elements"] * copies)
    atoms.insert("charge", np.tile(np.asarray(case["charges"]), copies))
    bonds = np.asarray(case["bonds"], dtype=np.uint64)
    bonds = np.concatenate([bonds + c * n for c in range(copies)])
    block = mp.core.Block()
    block.insert("atomi", bonds[:, 0].copy())
    block.insert("atomj", bonds[:, 1].copy())
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = block
    return frame, case


def _forcefield() -> mp.ff.forcefield.ForceField:
    return mp.io.openmm_xml.OpenmmXmlReader().read(FIXTURES / "opls.xml")


def test_one_template_per_molecule_named_as_the_pdb_names_it(tmp_path: Path) -> None:
    frame, case = _molecules("opls", 2)
    text = capture(
        mp.io.openmm_xml.OpenmmXmlWriter().write_system, _forcefield(), frame
    )
    path = tmp_path / "system.xml"
    mp.io.openmm_xml.OpenmmXmlWriter().write_system(path, _forcefield(), frame)
    assert path.read_text(encoding="utf-8") == text

    (residue,) = ET.fromstring(text).iter("Residue")
    assert residue.get("name") == "MOL"
    atoms = residue.findall("Atom")
    assert [a.get("type") for a in atoms] == case["types"]
    assert [float(a.get("charge")) for a in atoms] == case["charges"]
    assert len(residue.findall("Bond")) == len(case["bonds"])
    assert residue.findall("ExternalBond") == []

    pdb = mp.io.read_pdb(mp.io.Text(capture(mp.io.write_pdb, frame)))
    names = [a.get("name") for a in atoms]
    assert list(pdb["atoms"]["name"]) == names * 2
    assert list(pdb["atoms"]["res_id"]) == [1] * 12 + [2] * 12


def test_an_atom_type_the_force_field_lacks_is_a_value_error() -> None:
    frame, case = _molecules("opls", 1)
    frame["atoms"]["type"] = ["opls_999"] + case["types"][1:]
    with pytest.raises(ValueError, match="opls_999"):
        capture(mp.io.openmm_xml.OpenmmXmlWriter().write_system, _forcefield(), frame)
