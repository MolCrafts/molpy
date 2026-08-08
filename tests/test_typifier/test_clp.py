"""CL&P typifier — one construction, two molecules, no combinatorial fan-out.

Cost is paid once at the module-scoped ``clp`` fixture (see conftest). Every
test below only calls ``.typify`` or inspects static XML data.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from molpy.core.atomistic import Angle, Dihedral
from molpy.core.forcefield import AtomType
from molpy.data.forcefield import get_forcefield_path, list_forcefields
from molpy.io.forcefield.xml import read_xml_forcefield
from molpy.typifier import ClpTypifier, OPLSAATypifier

from .conftest import bf4_graph, c4c1im_graph

# Hard-coded from paduagroup/clandp il.ff (CL&P; JPCB 108 (2004) 2038,
# DOI 10.1021/jp0362133). Units: charge e; sigma Å; epsilon kcal/mol
# (il.ff kJ/mol / 4.184) — values a typified atom carries after OPLS unit conversion.
REF: dict[str, dict[str, float]] = {
    "NA": {"charge": 0.15, "sigma_A": 3.25, "epsilon_kcal": 0.17},
    "CR": {"charge": -0.11, "sigma_A": 3.55, "epsilon_kcal": 0.07},
    "CW": {"charge": -0.13, "sigma_A": 3.55, "epsilon_kcal": 0.07},
    "HCR": {"charge": 0.21, "sigma_A": 2.42, "epsilon_kcal": 0.03},
    "HCW": {"charge": 0.21, "sigma_A": 2.42, "epsilon_kcal": 0.03},
    "C1": {"charge": -0.17, "sigma_A": 3.5, "epsilon_kcal": 0.065999},
    "H1": {"charge": 0.13, "sigma_A": 2.5, "epsilon_kcal": 0.03},
    "C2": {"charge": 0.01, "sigma_A": 3.5, "epsilon_kcal": 0.065999},
    "CS": {"charge": -0.12, "sigma_A": 3.5, "epsilon_kcal": 0.065999},
    "CT": {"charge": -0.18, "sigma_A": 3.5, "epsilon_kcal": 0.065999},
    "HC": {"charge": 0.06, "sigma_A": 2.5, "epsilon_kcal": 0.03},
    "CBT": {"charge": 0.35, "sigma_A": 3.5, "epsilon_kcal": 0.065999},
    "SBT": {"charge": 1.02, "sigma_A": 3.55, "epsilon_kcal": 0.25},
    "NBT": {"charge": -0.66, "sigma_A": 3.25, "epsilon_kcal": 0.17},
    "OBT": {"charge": -0.53, "sigma_A": 3.15, "epsilon_kcal": 0.200134},
    "F1": {"charge": -0.16, "sigma_A": 3.118, "epsilon_kcal": 0.061042},
    "N3A": {"charge": -0.76, "sigma_A": 3.25, "epsilon_kcal": 0.17},
    "CZA": {"charge": 0.64, "sigma_A": 3.3, "epsilon_kcal": 0.065999},
    "NZA": {"charge": -0.76, "sigma_A": 3.2, "epsilon_kcal": 0.17},
}


def _assert_fully_typed(struct) -> None:
    assert all(a.get("type") is not None for a in struct.atoms)
    assert all(b.get("type") is not None for b in struct.bonds)
    for angle in struct.links.bucket(Angle):
        assert angle.get("type") is not None
    for dihedral in struct.links.bucket(Dihedral):
        assert dihedral.get("type") is not None


# --- static data (no typifier) ------------------------------------------------


def test_clp_is_not_an_oplsaa_subclass():
    assert not issubclass(ClpTypifier, OPLSAATypifier)


def test_clp_xml_is_packaged_and_separate_from_oplsaa():
    path = Path(get_forcefield_path("clp.xml"))
    assert path.exists()
    assert "clp.xml" in list_forcefields()
    opls = Path(get_forcefield_path("oplsaa.xml")).read_text()
    assert "jp0362133" not in opls
    assert 'name="NBT"' not in opls


def test_clp_xml_combining_and_14_scaling():
    root = ET.parse(get_forcefield_path("clp.xml")).getroot()
    nb = root.find("NonbondedForce")
    assert nb is not None
    assert float(nb.get("coulomb14scale")) == pytest.approx(0.5)
    assert float(nb.get("lj14scale")) == pytest.approx(0.5)


def test_clp_xml_reads_as_a_forcefield():
    ff = read_xml_forcefield(get_forcefield_path("clp.xml"))
    assert len(list(ff.get_types(AtomType))) > 0


# --- typify (shared clp fixture) ---------------------------------------------


def test_clp_typifier_owns_the_builtin_overlay(clp: ClpTypifier):
    assert clp.ff is not None
    assert ClpTypifier.load_forcefield() is clp.ff


def test_cation_and_anion_typify_once(clp: ClpTypifier):
    """One cation + one anion: ring types, full typing, net charge, LJ/charge ref.

    Covers the former multi-param matrix (four anions × charge × LJ × pipeline)
    without rebuilding ForceFieldParams for every ion.
    """
    cation = clp.typify(c4c1im_graph())
    types = [a.get("type") for a in cation.atoms]
    assert types[0] == "NA" and types[2] == "NA"
    assert types[1] == "CR"
    assert types[3] == "CW" and types[4] == "CW"
    assert types[6] == "HCR"
    assert types[8] == "HCW" and types[9] == "HCW"
    _assert_fully_typed(cation)
    assert sum(a.get("charge") for a in cation.atoms) == pytest.approx(1.0, abs=1e-6)

    for atom in cation.atoms:
        ref = REF.get(atom.get("type"))
        if ref is None:
            continue
        assert atom.get("charge") == pytest.approx(ref["charge"], abs=1e-4)
        assert atom.get("sigma") == pytest.approx(ref["sigma_A"], rel=1e-4)
        assert atom.get("epsilon") == pytest.approx(ref["epsilon_kcal"], rel=1e-4)

    anion = clp.typify(bf4_graph())
    assert {a.get("type") for a in anion.atoms} == {"B", "FBF"}
    _assert_fully_typed(anion)
    assert sum(a.get("charge") for a in anion.atoms) == pytest.approx(-1.0, abs=1e-6)
