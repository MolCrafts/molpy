"""The ethyl PEO 25-mer matches GroPoB tutorial/PEO_CH3 — with real AmberTools.

GroPoB cuts one oligomer (``PEO.ac``, the monomers already bonded) with
three control files and sequences ``HPT PEO PEO PEO TPT``. ``{[#PEO]|5}``
is that sequence. The cuts below are those control files. Their residue
names differ from the three-character truncation; the charges and the GAFF
coefficients do not.

Fixtures are the files committed in Teoroo-CMC/GroPoB ``tutorial/PEO_CH3``.

These tests run antechamber, parmchk2, prepgen and tleap. They use the
programs on ``PATH``, else those under ``$AMBERHOME/bin``, and skip when
neither has all four (CI has no AmberTools). ``test_amber_builder.py``
covers the builder without them.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import pytest

import molpy as mp
from molpy.ff.forcefield import AngleType, BondType, DihedralType
from molpy.builder import AmberCut, AmberPieces, AmberPolymerBuilder

_TOOLS = ("antechamber", "parmchk2", "prepgen", "tleap")


def _amber_env() -> dict[str, object] | None:
    """``env`` / ``env_manager`` for the installed AmberTools, or None."""
    if all(shutil.which(tool) for tool in _TOOLS):
        return {}
    home = os.environ.get("AMBERHOME")
    if home and all((Path(home) / "bin" / tool).is_file() for tool in _TOOLS):
        return {"env": Path(home), "env_manager": "venv"}
    return None


_ENV = _amber_env()
pytestmark = pytest.mark.skipif(
    _ENV is None,
    reason="AmberTools (antechamber, parmchk2, prepgen, tleap) is not installed",
)

# GroPoB tutorial/PEO_CH3 control files. The oligomer is their PEO.ac.
# Head omits the tail methyl, tail omits the head methyl, chain omits both.
_HEAD_OMIT = ("C3", "H6", "H7", "H8")
_TAIL_OMIT = ("C7", "H15", "H16", "H17")
_CUTS = {
    "PEO": {
        "head": AmberCut(tail="C6", post_tail_type="c3", omit=_TAIL_OMIT),
        "chain": AmberCut(
            head="C1",
            tail="C6",
            pre_head_type="c3",
            post_tail_type="c3",
            omit=_HEAD_OMIT + _TAIL_OMIT,
        ),
        "tail": AmberCut(head="C1", pre_head_type="c3", omit=_HEAD_OMIT),
    }
}


@pytest.fixture
def gropob(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "gropob"


def _peo_template(data: Path) -> mp.Atomistic:
    frame = mp.io.read_amber_ac(data / "PEO.ac")
    del frame["atoms"]["xyz"]  # x, y, z are there too; a graph column is 1-D
    return mp.Atomistic.from_frame(frame)


def _seed_monomer(data: Path, work: Path) -> None:
    monomer = work / "monomers" / "PEO"
    monomer.mkdir(parents=True)
    shutil.copy(data / "PEO.ac", monomer / "PEO.ac")
    shutil.copy(data / "PEO_initial.mol2", monomer / "PEO.mol2")
    shutil.copy(data / "PEO_initial.frcmod", monomer / "PEO.frcmod")


def _build(library, cuts, work: Path, sites: str = "{[#PEO]|5}"):
    return AmberPolymerBuilder(
        library,
        cuts,
        force_field="gaff",
        charge_method="bcc",
        work_dir=work,
        **(_ENV or {}),
    ).assemble(mp.io.cgsmiles.CgSmilesIr(sites).to_coarsegrain())


def _type_charges(frame: mp.Frame) -> dict[str, list[float]]:
    atoms = frame["atoms"]
    grouped: dict[str, list[float]] = {}
    for kind, charge in zip(atoms["type"], atoms["charge"], strict=True):
        grouped.setdefault(str(kind), []).append(float(charge))
    return {kind: sorted(values) for kind, values in grouped.items()}


def _coeffs(forcefield) -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    for kind in (BondType, AngleType, DihedralType):
        for entry in forcefield.get_types(kind):
            found[entry.name] = {
                key: float(entry.get(key)) for key in entry.keys() if key != "id"
            }
    return found


def test_prepgen_from_their_ac_matches_the_25mer(gropob, tmp_path):
    _seed_monomer(gropob, tmp_path)
    result = _build({"PEO": _peo_template(gropob)}, _CUTS, tmp_path)
    (script,) = (tmp_path / "chains").glob("*/polymer.in")
    assert "mol = sequence { HPE PEO PEO PEO TPE }" in script.read_text()

    reference_ff, reference = mp.io.read_amber_prmtop_system(
        gropob / "PEO_25mer.prmtop"
    )
    frame = result.chain.to_frame()
    atoms = frame["atoms"]
    assert atoms.nrows == 183
    assert Counter(atoms["type"]) == Counter({"os": 25, "c3": 52, "h1": 100, "hc": 6})
    assert abs(float(sum(atoms["charge"]))) <= 0.01
    dihedrals = frame["dihedrals"]
    quartets = list(
        zip(
            dihedrals["atomi"],
            dihedrals["atomj"],
            dihedrals["atomk"],
            dihedrals["atoml"],
            strict=True,
        )
    )
    assert len(quartets) == len(set(quartets))
    ours = _type_charges(frame)
    theirs = _type_charges(reference)
    assert ours.keys() == theirs.keys()
    for kind in ours:
        assert ours[kind] == pytest.approx(theirs[kind], abs=0.01)
    assert min(ours["os"]) >= -0.45
    assert max(ours["os"]) <= -0.40
    left = _coeffs(result.forcefield)
    right = _coeffs(reference_ff)
    assert left.keys() == right.keys()
    for name in left:
        for key in left[name] | right[name]:
            assert left[name].get(key, 0.0) == pytest.approx(
                right[name].get(key, 0.0), abs=1e-4
            )


def test_rerunning_antechamber_keeps_ether_oxygens(gropob, tmp_path):
    result = _build({"PEO": _peo_template(gropob)}, _CUTS, tmp_path)
    reference = mp.io.read_amber_prmtop(gropob / "PEO_25mer.prmtop")
    ours = _type_charges(result.chain.to_frame())
    theirs = _type_charges(reference)
    diffs = [
        abs(left - right)
        for kind in ours
        for left, right in zip(ours[kind], theirs[kind], strict=True)
    ]
    assert sum(diffs) / len(diffs) <= 0.02
    assert max(ours["os"]) <= -0.40
    assert result.chain.n_atoms == 183


def test_pieces_build_a_methoxy_capped_10mer(tmp_path):
    """Three SMILES in, CH3-(OCH2CH2)10-OCH3 out, with ether oxygens."""
    oligomer, cuts = AmberPieces("COCC", "OCC", "OCCOC").oligomer(seed=42)
    result = _build({"PEO": oligomer}, {"PEO": cuts}, tmp_path, "{[#PEO]|10}")
    frame = result.chain.to_frame()
    atoms = frame["atoms"]
    elements = Counter(str(element) for element in atoms["element"])
    assert elements == Counter({"C": 22, "O": 11, "H": 46})
    assert Counter(atoms["type"])["os"] == 11
    assert abs(float(sum(atoms["charge"]))) <= 0.01
    assert max(_type_charges(frame)["os"]) <= -0.30
