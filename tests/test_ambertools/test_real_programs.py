"""The AmberTools routes against the real programs.

These tests run antechamber, sqm, parmchk2, prepgen and tleap. They use the
programs on ``PATH``, else those under ``$AMBERHOME/bin``, and skip when
neither has all of them (CI has no AmberTools); the faked tests next to the
code cover the same classes without them.

They pin what the default routes rest on:

* AM1-BCC: molrs ``BccModel`` corrects sqm's AM1 charges exactly as
  antechamber ``-c bcc`` does, so :class:`molpy.ff.charge.Am1BccModel` (sqm
  for the QM step, the correction native) gives antechamber's charges.
* GAFF: the native ``AtdTypifier`` + ``GaffTypifier`` assign the types and
  parameter values tleap's prmtop carries for the same ``parameter_set``.
* The AmberTools pipeline: tleap reads the double-quoted paths molpy writes,
  spaces and all, and the builder's one ``-fo ac`` antechamber run feeds
  parmchk2 (``-f ac``) and prepgen.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy._ambertools import AmberSteps, input_mol2, net_formal_charge
from molpy.config import load_config
from molpy.core import Angle, Dihedral, Improper
from molpy.ff.charge import Am1BccModel, BccModel
from molpy.ff.typifier import AntechamberTypifier, AtdTypifier, GaffTypifier

_TOOLS = ("antechamber", "sqm", "parmchk2", "prepgen", "tleap")


def _amber_env() -> dict[str, object] | None:
    """The ``[wrapper]`` run overrides for the installed AmberTools, or None."""
    if all(shutil.which(tool) for tool in _TOOLS):
        return {}
    home = os.environ.get("AMBERHOME")
    if home and all((Path(home) / "bin" / tool).is_file() for tool in _TOOLS):
        return {"wrapper": {"env": home, "env_manager": "venv"}}
    return None


_ENV = _amber_env()
pytestmark = pytest.mark.skipif(
    _ENV is None,
    reason="AmberTools (antechamber, sqm, parmchk2, prepgen, tleap) is not installed",
)

MOLECULES = {
    "ethanol": "CCO",
    "acetate": "CC(=O)[O-]",
    "benzene": "c1ccccc1",
    "tfsi": "O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F",
}


@pytest.fixture
def config():
    return load_config(_ENV)


def _embedded(smiles: str) -> mp.Atomistic:
    mol = mp.io.smiles.SmilesIr(smiles).to_atomistic()
    return mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(mol)[0]


def _sqm_mulliken(sqm_out: Path) -> np.ndarray:
    """The Mulliken charges sqm printed last (three decimals)."""
    text = sqm_out.read_text(encoding="utf-8")
    block = text.rsplit("Element       Mulliken Charge", 1)[-1]
    charges = []
    for line in block.splitlines()[1:]:
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit():
            break
        charges.append(float(fields[2]))
    return np.array(charges)


def _antechamber_bcc(mol: mp.Atomistic, directory: Path, config) -> np.ndarray:
    """antechamber ``-c bcc`` charges, its sqm.out left in ``directory``."""
    directory.mkdir(parents=True)
    source = directory / "input.mol2"
    source.write_text(input_mol2(mol, rename=True), encoding="utf-8")
    ac = directory / "bcc.ac"
    AmberSteps(directory, config).antechamber(
        source,
        ac,
        parameter_set="gaff2",
        charge_method="bcc",
        net_charge=net_formal_charge(mol),
    )
    return np.asarray(mp.io.read_ac(ac)["atoms"]["charge"], dtype=np.float64)


class TestAm1Bcc:
    @pytest.mark.parametrize("name", MOLECULES)
    def test_the_native_correction_of_one_sqm_run_is_antechamber_bcc(
        self, name, tmp_path, config
    ):
        mol = _embedded(MOLECULES[name])
        bcc = _antechamber_bcc(mol, tmp_path / "bcc", config)
        am1 = _sqm_mulliken(tmp_path / "bcc" / "sqm.out")
        native = BccModel(parameter_set="bcc").assign(mol, qm=am1)
        # The ac prints six decimals.
        assert native.tolist() == pytest.approx(bcc.tolist(), abs=1e-6)

    @pytest.mark.parametrize("name", MOLECULES)
    def test_the_model_gives_antechamber_s_charges(self, name, tmp_path, config):
        mol = _embedded(MOLECULES[name])
        bcc = _antechamber_bcc(mol, tmp_path / "bcc", config)
        model = Am1BccModel(parameter_set="bcc", work_dir=tmp_path / "m", config=config)
        charges = model.assign(mol)
        # Two sqm runs can round one Mulliken charge differently (0.001 e).
        assert charges.tolist() == pytest.approx(bcc.tolist(), abs=1e-3)
        assert float(charges.sum()) == pytest.approx(net_formal_charge(mol), abs=0.01)

    def test_the_default_route_types_natively_with_am1bcc_charges(
        self, tmp_path, config
    ):
        mol = _embedded(MOLECULES["acetate"])
        labelled = AtdTypifier(parameter_set="gaff2").typify(mol)
        model = Am1BccModel(parameter_set="bcc", work_dir=tmp_path, config=config)
        charges = model.assign(labelled)
        for atom, charge in zip(labelled.atoms, charges, strict=True):
            atom["charge"] = float(charge)
        gaff = GaffTypifier(parameter_set="gaff2")
        typed = gaff.typify(labelled)
        assert [atom["charge"] for atom in typed.atoms] == pytest.approx(
            charges.tolist()
        )
        assert sorted({atom["type"] for atom in typed.atoms}) == ["c", "c3", "hc", "o"]
        assert sorted({bond["type"] for bond in typed.bonds}) == [
            "c-c3",
            "c-o",
            "c3-hc",
        ]


def _rows(graph: mp.Atomistic) -> dict[int, int]:
    return {atom.handle: row for row, atom in enumerate(graph.atoms)}


def _terms(graph: mp.Atomistic, links, keys: tuple[str, ...]) -> dict:
    """Endpoint rows (either orientation) → the term's parameters."""
    row = _rows(graph)
    found = {}
    for link in links:
        ends = tuple(row[a.handle] for a in link.endpoints)
        found[min(ends, ends[::-1])] = tuple(float(link[key]) for key in keys)
    return found


def _torsions(graph: mp.Atomistic) -> dict:
    """Atom quartet → its non-zero (periodicity, k, phase) cosine terms."""
    row = _rows(graph)
    found = {}
    for link in graph.links.exact_bucket(Dihedral):
        ends = tuple(row[a.handle] for a in link.endpoints)
        terms = []
        index = 1
        while link.get(f"k{index}") is not None:
            k = float(link[f"k{index}"])
            if k != 0.0:
                terms.append(
                    (
                        float(link[f"periodicity{index}"]),
                        round(k, 6),
                        float(link[f"phase{index}"]),
                    )
                )
            index += 1
        found[min(ends, ends[::-1])] = sorted(terms)
    return found


def _assert_close(left: dict, right: dict, tolerance: tuple[float, ...]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        for ours, theirs, tol in zip(left[key], right[key], tolerance, strict=True):
            assert ours == pytest.approx(theirs, abs=tol), key


class TestNativeGaffIsTleaps:
    """For one ``parameter_set``, native GAFF is tleap's, value for value."""

    @pytest.mark.parametrize("parameter_set", ["gaff", "gaff2"])
    @pytest.mark.parametrize("name", MOLECULES)
    def test_types_and_parameter_values_agree(
        self, name, parameter_set, tmp_path, config
    ):
        mol = _embedded(MOLECULES[name])
        leap = AntechamberTypifier(
            parameter_set=parameter_set, work_dir=tmp_path, config=config
        ).typify(mol)
        native = GaffTypifier(parameter_set=parameter_set).typify(
            AtdTypifier(parameter_set=parameter_set).typify(mol)
        )
        assert [a["type"] for a in native.atoms] == [a["type"] for a in leap.atoms]
        # parmchk2 prints an estimated length to 3 decimals and an angle to
        # 2; a prmtop stores θ0 in radians.
        _assert_close(
            _terms(native, native.bonds, ("k", "r0")),
            _terms(leap, leap.bonds, ("k", "r0")),
            (1e-6, 5e-4),
        )
        _assert_close(
            _terms(native, native.links.exact_bucket(Angle), ("k", "theta0")),
            _terms(leap, leap.links.exact_bucket(Angle), ("k", "theta0")),
            (1e-6, 5e-3),
        )
        assert _torsions(native) == _torsions(leap)
        # A prmtop improper may run backwards (the same angle).
        improper = ("k", "periodicity", "phase")
        _assert_close(
            _terms(native, native.links.exact_bucket(Improper), improper),
            _terms(leap, leap.links.exact_bucket(Improper), improper),
            (1e-6, 0.0, 1e-6),
        )

    @pytest.mark.parametrize(
        ("parameter_set", "k", "r0"),
        [("gaff", 356.9, 1.3497), ("gaff2", 334.28, 1.3512)],
    )
    def test_c3_f_is_the_generation_s_row(self, parameter_set, k, r0, tmp_path, config):
        """gaff.dat 1.81 and gaff2.dat 2.2.30 disagree; each route follows its set."""
        mol = _embedded(MOLECULES["tfsi"])
        for typed in (
            AntechamberTypifier(
                parameter_set=parameter_set, work_dir=tmp_path, config=config
            ).typify(mol),
            GaffTypifier(parameter_set=parameter_set).typify(
                AtdTypifier(parameter_set=parameter_set).typify(mol)
            ),
        ):
            bonds = [b for b in typed.bonds if b["type"].split("@")[0] == "c3-f"]
            assert bonds
            assert {(b["k"], b["r0"]) for b in bonds} == {(k, r0)}


class TestPipeline:
    def test_tleap_reads_quoted_paths_with_spaces(self, tmp_path, config):
        work = tmp_path / "amber work" / "with spaces"
        typed = AntechamberTypifier(
            parameter_set="gaff2", work_dir=work, config=config
        ).typify(_embedded("CCO"))
        assert typed.n_atoms == 9
        (prmtop,) = work.glob("*/MOL.prmtop")
        assert " " in str(prmtop)

    def test_the_builder_s_ac_feeds_parmchk2_and_prepgen(
        self, tmp_path, config, molpy_records
    ):
        work = tmp_path / "peo work"
        oligomer, cuts = mp.builder.AmberPieces("COCC", "OCC", "OCCOC").oligomer(
            seed=42
        )
        result = mp.builder.AmberPolymerBuilder(
            {"PEO": oligomer},
            {"PEO": cuts},
            parameter_set="gaff2",
            charge_method="bcc",
            work_dir=work,
            config=config,
        ).assemble(mp.io.cgsmiles.CgSmilesIr("{[#PEO]|4}").to_coarsegrain())

        commands = [
            r.extra["command"] for r in molpy_records if r.message == "process finished"
        ]
        programs = Counter(Path(command[0]).name for command in commands)
        assert programs == Counter(
            {"antechamber": 1, "parmchk2": 1, "prepgen": 3, "tleap": 1}
        )
        (ante,) = (c for c in commands if Path(c[0]).name == "antechamber")
        (parmchk2,) = (c for c in commands if Path(c[0]).name == "parmchk2")
        assert ante[ante.index("-fo") + 1] == "ac"
        assert parmchk2[parmchk2.index("-f") + 1] == "ac"
        assert parmchk2[parmchk2.index("-i") + 1] == ante[ante.index("-o") + 1]

        frame = result.chain.to_frame()
        elements = Counter(str(e) for e in frame["atoms"]["element"])
        assert elements == Counter({"C": 10, "O": 5, "H": 22})
        assert Counter(frame["atoms"]["type"])["os"] == 5
        assert abs(float(sum(frame["atoms"]["charge"]))) <= 0.01
