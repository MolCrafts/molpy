"""AntechamberTypifier / TLeapTypifier with the AmberTools executables faked.

``subprocess.run`` is patched (as ``tests/test_wrapper`` does): each tool call
copies a committed output fixture into place — ``mol2/litfsi_gaff2.mol2`` for
antechamber, ``frcmod/litfsi_gaff2.frcmod`` for parmchk2 and
``prmtop/LiTFSI.prmtop`` for tleap. The tests assert what the typifiers own:
the net charge they hand antechamber, how prmtop rows land on graph rows, the
frcmod TLeapTypifier writes, and the errors.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import molpy as mp
from molpy.core import Angle, Dihedral
from molpy.wrapper import Wrapper

# LiTFSI in LiTFSI.prmtop row order: [F-C(F2)-S(O2)-N-S(O2)-C(F3)]- and Li+.
ELEMENTS = ["F", "C", "F", "F", "S", "O", "O", "N", "S", "O", "O", "C", "F", "F", "F"]
ELEMENTS += ["Li"]
BONDS = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6), (4, 7)]
BONDS += [(7, 8), (8, 9), (8, 10), (8, 11), (11, 12), (11, 13), (11, 14)]
TYPES = ["f", "c3", "f", "f", "s6", "o", "o", "ne", "sy", "o", "o", "c3", "f", "f"]
TYPES += ["f", "Li+"]
CHARGES = [-0.271633, 0.5547, -0.271633, -0.271633, 1.4837, -0.6132, -0.6568]
CHARGES += [-1.0591, 1.6824, -0.6568, -0.6568, 0.5547, -0.271633, -0.271633]
CHARGES += [-0.271633, 1.0]
N_ROW, LI_ROW = 7, 15


def _litfsi() -> mp.Atomistic:
    """Untyped LiTFSI with formal charges on N (-1) and Li (+1)."""
    graph = mp.Atomistic()
    atoms = [
        graph.def_atom(element=e, x=1.5 * i, y=0.0, z=0.0)
        for i, e in enumerate(ELEMENTS)
    ]
    for i, j in BONDS:
        graph.def_bond(atoms[i], atoms[j])
    atoms[N_ROW]["formal_charge"] = -1.0
    atoms[LI_ROW]["formal_charge"] = 1.0
    return graph


def _typed_litfsi() -> mp.Atomistic:
    """LiTFSI already carrying its GAFF2 types and charges."""
    graph = _litfsi()
    for atom, name, charge in zip(graph.atoms, TYPES, CHARGES, strict=True):
        atom["type"] = name
        atom["charge"] = charge
    return graph


def _link(graph: mp.Atomistic, links, rows: tuple[int, ...]):
    """The relation of ``links`` whose endpoints are graph atoms ``rows``."""
    row = {atom.handle: index for index, atom in enumerate(graph.atoms)}
    for link in links:
        ends = tuple(row[a.handle] for a in link.endpoints)
        if ends in (rows, rows[::-1]):
            return link
    raise LookupError(rows)


class FakeAmberTools:
    """Stands in for ``subprocess.run``: a tool copies its fixture output into place."""

    def __init__(self, data: Path, *, fail: str | None = None) -> None:
        self.outputs = {
            "antechamber": data / "mol2" / "litfsi_gaff2.mol2",
            "parmchk2": data / "frcmod" / "litfsi_gaff2.frcmod",
            "tleap": data / "prmtop" / "LiTFSI.prmtop",
        }
        self.fail = fail
        self.calls: dict[str, list[str]] = {}
        self.scripts: list[str] = []

    def __call__(self, argv: list[str], *, cwd: str, **_: object):
        tool, args = argv[0], argv[1:]
        self.calls[tool] = args
        if tool == self.fail:
            return subprocess.CompletedProcess(argv, 1, "", f"{tool}: fatal error")
        if tool == "tleap":
            script = (Path(cwd) / args[1]).read_text()
            self.scripts.append(script)
            save = next(s for s in script.splitlines() if s.startswith("saveamberparm"))
            target = save.split()[2]
        else:
            target = args[args.index("-o") + 1]
        shutil.copy(self.outputs[tool], target)
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def tools(TEST_DATA_DIR: Path):
    fake = FakeAmberTools(TEST_DATA_DIR)
    with (
        patch.object(Wrapper, "is_available", return_value=True),
        patch("molpy.wrapper._base.subprocess.run", side_effect=fake),
    ):
        yield fake


class TestAntechamberTypifier:
    def test_net_charge_is_the_sum_of_formal_charges(self, tools, tmp_path):
        mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path).typify(_litfsi())
        args = tools.calls["antechamber"]
        assert args[args.index("-nc") + 1] == "0"
        assert args[args.index("-at") + 1] == "gaff2"
        assert args[args.index("-c") + 1] == "bcc"

        anion = _litfsi()
        del list(anion.atoms)[LI_ROW]["formal_charge"]
        mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path).typify(anion)
        args = tools.calls["antechamber"]
        assert args[args.index("-nc") + 1] == "-1"

    def test_runs_antechamber_parmchk2_then_tleap(self, tools, tmp_path):
        mp.ff.typifier.AntechamberTypifier(atom_type="gaff", work_dir=tmp_path).typify(
            _litfsi()
        )
        parmchk2 = tools.calls["parmchk2"]
        assert parmchk2[parmchk2.index("-s") + 1] == "gaff"
        lines = tools.scripts[0].splitlines()
        assert lines[0] == "source leaprc.gaff"
        assert lines[1].startswith("MOL = loadmol2 ")
        assert lines[2].startswith("loadamberparams ")

    def test_prmtop_rows_map_onto_graph_rows(self, tools, tmp_path):
        graph = _litfsi()
        typed = mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path).typify(graph)

        atoms = list(typed.atoms)
        assert [a["type"] for a in atoms] == TYPES
        assert [a["charge"] for a in atoms] == pytest.approx(CHARGES)
        assert atoms[N_ROW]["mass"] == pytest.approx(14.01)
        assert atoms[LI_ROW]["mass"] == pytest.approx(6.94)

        bond = _link(typed, typed.bonds, (0, 1))
        assert bond["type"] == "c3-f"
        assert bond["k"] == pytest.approx(356.9)  # RK, E = k (r - r0)^2
        assert bond["r0"] == pytest.approx(1.3497)
        assert _link(typed, typed.bonds, (7, 8))["type"] == "ne-sy"

        angle = _link(typed, typed.links.exact_bucket(Angle), (4, 7, 8))
        assert angle["type"] == "s6-ne-sy"
        assert angle["k"] == pytest.approx(65.9)  # TK
        assert len(list(typed.links.exact_bucket(Angle))) == 25

        dihedral = _link(typed, typed.links.exact_bucket(Dihedral), (4, 7, 8, 11))
        # A proper is named in TypeName::orient's spelling: the smaller of
        # the forward and the reversed type tuple (c3 < s6).
        assert dihedral["type"] == "c3-sy-ne-s6"
        assert dihedral["k1"] == pytest.approx(6.8)
        assert dihedral["k2"] == pytest.approx(0.5)
        assert len(list(typed.links.exact_bucket(Dihedral))) == 24

        assert all(a.get("type") is None for a in graph.atoms)

    def test_forcefield_holds_the_assigned_types(self, tools, tmp_path):
        ante = mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path)
        ante.typify(_litfsi())
        ff = ante.forcefield()
        (atom_style,) = ff.get_styles("atom")
        assert {t.name for t in atom_style.types} == set(TYPES)
        bond = ff.get_style("bond", "harmonic").get_type_by_name("c3-f")
        assert bond["k"] == pytest.approx(356.9)
        pair = ff.get_style("pair", "lj/cut").get_type_by_name("f")
        assert pair["sigma"] == pytest.approx(3.118145514)

    def test_graph_bond_missing_from_prmtop_raises(self, tools, tmp_path):
        graph = _litfsi()
        atoms = list(graph.atoms)
        graph.def_bond(atoms[N_ROW], atoms[LI_ROW])
        with pytest.raises(ValueError, match=r"bond \(7, 15\) is not in the prmtop"):
            mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path).typify(graph)

    def test_failing_tool_raises_with_its_stderr(self, TEST_DATA_DIR, tmp_path):
        fake = FakeAmberTools(TEST_DATA_DIR, fail="antechamber")
        with (
            patch.object(Wrapper, "is_available", return_value=True),
            patch("molpy.wrapper._base.subprocess.run", side_effect=fake),
            pytest.raises(RuntimeError, match="antechamber: fatal error"),
        ):
            mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path).typify(_litfsi())

    def test_missing_tool_raises(self, tmp_path):
        with (
            patch.object(Wrapper, "is_available", return_value=False),
            pytest.raises(RuntimeError, match="antechamber is not available"),
        ):
            mp.ff.typifier.AntechamberTypifier(work_dir=tmp_path).typify(_litfsi())


class TestTLeapTypifier:
    def test_writes_the_given_forcefield_as_frcmod(self, tools, tmp_path):
        ff = mp.ff.forcefield.ForceField("tfsi", units="real")
        ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 5.0 / 6.0])
        atoms = ff.def_style("atom", "full")
        c3 = atoms.def_type("c3", mass=12.01)
        f = atoms.def_type("f", mass=19.0)
        ff.def_style("bond", "harmonic").def_type("c3-f", c3, f, k=356.9, r0=1.3497)

        leap = mp.ff.typifier.TLeapTypifier(forcefield=ff, work_dir=tmp_path)
        leap.typify(_typed_litfsi())

        lines = tools.scripts[0].splitlines()
        assert lines[0] == "source leaprc.gaff2"
        frcmod = Path(lines[1].removeprefix("loadamberparams "))
        rows = [line.split() for line in frcmod.read_text().splitlines()]
        assert ["c3", "12.010000"] in rows  # MASS
        assert ["c3-f", "356.900000", "1.349700"] in rows  # BOND, RK = k
        assert lines[2].startswith("MOL = loadmol2 ")
        assert set(tools.calls) == {"tleap"}

    def test_keeps_types_and_charges_and_assigns_terms(self, tools, tmp_path):
        typed = mp.ff.typifier.TLeapTypifier(work_dir=tmp_path).typify(_typed_litfsi())
        atoms = list(typed.atoms)
        assert [a["type"] for a in atoms] == TYPES
        assert [a["charge"] for a in atoms] == CHARGES
        assert _link(typed, typed.bonds, (8, 11))["type"] == "c3-sy"
        script = tools.scripts[0].splitlines()
        assert not any(line.startswith("loadamberparams") for line in script)

    def test_type_change_raises(self, tools, tmp_path):
        graph = _typed_litfsi()
        list(graph.atoms)[0]["type"] = "c3"
        with pytest.raises(ValueError, match="type c3 -> f"):
            mp.ff.typifier.TLeapTypifier(work_dir=tmp_path).typify(graph)

    def test_charge_change_raises(self, tools, tmp_path):
        graph = _typed_litfsi()
        list(graph.atoms)[0]["charge"] = -0.2
        with pytest.raises(ValueError, match="charge -0.2 -> "):
            mp.ff.typifier.TLeapTypifier(work_dir=tmp_path).typify(graph)

    def test_untyped_atom_raises(self, tmp_path):
        with pytest.raises(ValueError, match="needs AMBER types and charges"):
            mp.ff.typifier.TLeapTypifier(work_dir=tmp_path).typify(_litfsi())

    def test_failing_tleap_raises_with_its_stderr(self, TEST_DATA_DIR, tmp_path):
        fake = FakeAmberTools(TEST_DATA_DIR, fail="tleap")
        with (
            patch.object(Wrapper, "is_available", return_value=True),
            patch("molpy.wrapper._base.subprocess.run", side_effect=fake),
            pytest.raises(RuntimeError, match="tleap: fatal error"),
        ):
            mp.ff.typifier.TLeapTypifier(work_dir=tmp_path).typify(_typed_litfsi())
