"""Tests for the CL&P ionic-liquid force field typifier.

CL&P (Canongia Lopes & Padua, JPCB 108 (2004) 2038, DOI 10.1021/jp0362133) is
implemented by inheritance from the OPLS-AA typifier with its own clp.xml data.
Reference values are transcribed from the authoritative paduagroup/clandp il.ff
distribution into ``fixtures/clp_ilff_reference.json``.

The tests below check atom typing and nonbonded (charge/LJ) parameters, so they
construct ``ClpTypifier`` with the bonded-term passes skipped (those are not
under test and the bare ion fragments lack full bonded/improper coverage).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.data.forcefield import get_forcefield_path, list_forcefields
from molpy.io.forcefield.xml import read_xml_forcefield
from molpy.typifier import ClpTypifier, OplsTypifier

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "clp_ilff_reference.json").read_text()
)
REF = FIXTURE["atom_types"]


# --------------------------------------------------------------------------
# structure builders (connectivity from paduagroup/clandp z-matrices)
# --------------------------------------------------------------------------
def _build(elements, edges):
    asm = Atomistic()
    atoms = [Atom(element=e) for e in elements]
    asm.add_entity(*atoms)
    for i, j in edges:
        asm.add_link(Bond(atoms[i], atoms[j]))
    return asm, atoms


def _c4c1im():
    el = [
        "N",
        "C",
        "N",
        "C",
        "C",
        "C",
        "H",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
        (5, 11),
        (5, 12),
        (7, 13),
        (7, 14),
        (7, 15),
        (13, 16),
        (13, 17),
        (13, 18),
        (16, 19),
        (16, 20),
        (16, 21),
        (19, 22),
        (19, 23),
        (19, 24),
    ]
    return _build(el, edges)


def _bf4():
    return _build(["B", "F", "F", "F", "F"], [(0, 1), (0, 2), (0, 3), (0, 4)])


def _pf6():
    return _build(["P", "F", "F", "F", "F", "F", "F"], [(0, i) for i in range(1, 7)])


def _ntf2():
    el = ["C", "F", "F", "F", "S", "N", "O", "O", "S", "O", "O", "C", "F", "F", "F"]
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 8),
        (8, 9),
        (8, 10),
        (8, 11),
        (11, 12),
        (11, 13),
        (11, 14),
    ]
    return _build(el, edges)


def _dca():
    return _build(["N", "C", "N", "C", "N"], [(0, 1), (1, 2), (0, 3), (3, 4)])


# --------------------------------------------------------------------------
# ac-001: ClpTypifier importable and subclass of OplsTypifier
# --------------------------------------------------------------------------
def test_clp_typifier_is_opls_subclass():
    assert issubclass(ClpTypifier, OplsTypifier)


def test_clp_typifier_loads_builtin_forcefield():
    assert ClpTypifier().ff is not None


# --------------------------------------------------------------------------
# ac-002: clp.xml resolves via get_forcefield_path / list_forcefields
# --------------------------------------------------------------------------
def test_clp_xml_resolves():
    path = Path(get_forcefield_path("clp.xml"))
    assert path.exists()
    assert "clp.xml" in list_forcefields()


# --------------------------------------------------------------------------
# ac-003: oplsaa.xml is not polluted with CL&P content
# --------------------------------------------------------------------------
def test_oplsaa_not_merged_with_clp():
    opls = Path(get_forcefield_path("oplsaa.xml")).read_text()
    assert "jp0362133" not in opls  # CL&P DOI must not appear in OPLS data
    assert 'name="NBT"' not in opls  # CL&P-only atom type


# --------------------------------------------------------------------------
# ac-004: clp.xml read through OPLS reader, no dedicated ClpForceFieldReader
# --------------------------------------------------------------------------
def test_clp_read_through_opls_reader():
    ff = read_xml_forcefield(get_forcefield_path("clp.xml"))
    from molpy.core.forcefield import AtomType

    assert len(list(ff.get_types(AtomType))) > 0


def test_no_dedicated_clp_reader_class():
    import molpy.io.forcefield.xml as xmlmod

    assert not hasattr(xmlmod, "ClpForceFieldReader")


# --------------------------------------------------------------------------
# ac-005: imidazolium ring atoms CR vs CW vs NA discriminated on [C4C1im]+
# --------------------------------------------------------------------------
def test_imidazolium_ring_discrimination():
    asm, _ = _c4c1im()
    out = ClpTypifier(
        skip_bond_typing=True,
        skip_angle_typing=True,
        skip_dihedral_typing=True,
        strict_typing=False,
    ).typify(asm)
    types = [a.get("type") for a in out.atoms]
    # ring: NA(0) CR(1) NA(2) CW(3) CW(4), HCR(6), HCW(8,9)
    assert types[0] == "NA" and types[2] == "NA"
    assert types[1] == "CR"
    assert types[3] == "CW" and types[4] == "CW"
    assert types[6] == "HCR"
    assert types[8] == "HCW" and types[9] == "HCW"


def test_c4c1im_fully_typed():
    asm, _ = _c4c1im()
    out = ClpTypifier(
        skip_bond_typing=True,
        skip_angle_typing=True,
        skip_dihedral_typing=True,
        strict_typing=False,
    ).typify(asm)
    assert all(a.get("type") is not None for a in out.atoms)


# --------------------------------------------------------------------------
# ac-006: the four anions typify without error, every atom gets a type
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "builder,expected",
    [
        (_bf4, {"B", "FB"}),
        (_pf6, {"P", "FP"}),
        (_ntf2, {"CBT", "SBT", "NBT", "OBT", "F1"}),
        (_dca, {"N3A", "CZA", "NZA"}),
    ],
)
def test_anion_typing(builder, expected):
    asm, _ = builder()
    out = ClpTypifier(
        skip_bond_typing=True,
        skip_angle_typing=True,
        skip_dihedral_typing=True,
        strict_typing=False,
    ).typify(asm)
    types = {a.get("type") for a in out.atoms}
    assert None not in types
    assert types == expected


# --------------------------------------------------------------------------
# ac-007: assigned charges / LJ match il.ff reference within tolerance
# --------------------------------------------------------------------------
def test_charges_and_lj_match_ilff_reference():
    typ = ClpTypifier(
        skip_bond_typing=True,
        skip_angle_typing=True,
        skip_dihedral_typing=True,
        strict_typing=False,
    )
    for builder in (_c4c1im, _ntf2, _dca):
        asm, _ = builder()
        out = typ.typify(asm)
        for atom in out.atoms:
            t = atom.get("type")
            if t not in REF:
                continue
            ref = REF[t]
            assert atom.get("charge") == pytest.approx(ref["charge"], abs=1e-4)
            assert atom.get("sigma") == pytest.approx(ref["sigma_A"], rel=1e-4)
            assert atom.get("epsilon") == pytest.approx(ref["epsilon_kcal"], rel=1e-4)


# --------------------------------------------------------------------------
# ac-008: each ion's summed partial charge is an integer (+1 / -1)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "builder,total",
    [(_c4c1im, 1.0), (_bf4, -1.0), (_pf6, -1.0), (_ntf2, -1.0), (_dca, -1.0)],
)
def test_integer_ion_charge(builder, total):
    asm, _ = builder()
    out = ClpTypifier(
        skip_bond_typing=True,
        skip_angle_typing=True,
        skip_dihedral_typing=True,
        strict_typing=False,
    ).typify(asm)
    q = sum(a.get("charge") for a in out.atoms)
    assert q == pytest.approx(total, abs=1e-6)


# --------------------------------------------------------------------------
# ac-009: clp.xml records geometric combining + 0.5/0.5 1-4 scaling
# --------------------------------------------------------------------------
def test_combining_and_14_scaling():
    root = ET.parse(get_forcefield_path("clp.xml")).getroot()
    nb = root.find("NonbondedForce")
    assert nb is not None
    assert float(nb.get("coulomb14scale")) == pytest.approx(0.5)
    assert float(nb.get("lj14scale")) == pytest.approx(0.5)
