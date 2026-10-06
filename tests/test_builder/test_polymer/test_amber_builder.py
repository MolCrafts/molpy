"""AmberPolymerBuilder drives antechamber, parmchk2, prepgen and tleap.

The executables are faked, so none of this needs AmberTools. The cuts
written into the prepgen control files are the AmberCut values the caller
passed. The chain is not built with Assembler, and a prepi file is not
edited after prepgen writes it. ``test_amber_gropob.py`` runs the real
programs when they are installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import molpy as mp
from molpy.builder.polymer import (
    AmberCut,
    AmberPieces,
    AmberPolymerBuilder,
)
from molpy.wrapper import Wrapper


def _inpcrd(path: Path, n_atoms: int = 16) -> None:
    values: list[float] = []
    for index in range(n_atoms):
        values.extend((0.1 * index, 0.0, 0.0))
    lines = ["mol", f"{n_atoms:5d}"]
    row: list[str] = []
    for value in values:
        row.append(f"{value:12.7f}")
        if len(row) == 6:
            lines.append("".join(row))
            row = []
    if row:
        lines.append("".join(row))
    path.write_text("\n".join(lines) + "\n")


def _ac_from_mol2(mol2: Path, ac: Path) -> None:
    lines = mol2.read_text().splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    names = []
    for line in lines[start:]:
        if line.startswith("@"):
            break
        if line.strip():
            names.append(line.split()[1])
    lines = ["CHARGE      0.00 ( 0 )", f"Formula: C{len(names)}"]
    for index, name in enumerate(names, start=1):
        kind = "hc" if name.startswith("H") else "c3"
        lines.append(
            f"ATOM {index:6d}  {name:<4} MOL {index:5d}"
            f"    0.000   0.000   0.000  0.000000        {kind}"
        )
    if len(names) >= 2:
        lines.append("BOND    1    1    2    1     C1   C2")
    ac.write_text("\n".join(lines) + "\n")


def _target(args: list[str], cwd: str) -> Path:
    target = Path(args[args.index("-o") + 1])
    return target if target.is_absolute() else Path(cwd) / target


class _FakeTools:
    """Stands in for ``subprocess.run``: each tool writes a plausible output."""

    def __init__(self, prmtop: Path, *, fail: str | None = None) -> None:
        self.prmtop = prmtop
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def __call__(self, argv: list[str], *, cwd: str, **_: object):
        tool = Path(argv[0]).name
        args = argv[1:]
        self.calls.append((tool, args))
        if tool == self.fail:
            return subprocess.CompletedProcess(argv, 1, f"{tool}: cannot do it", "")
        if tool == "tleap":
            script = (Path(cwd) / args[1]).read_text()
            save = next(
                line for line in script.splitlines() if line.startswith("saveamberparm")
            )
            prmtop, inpcrd = save.split()[2:]
            shutil.copy(self.prmtop, prmtop)
            _inpcrd(Path(inpcrd))
        elif tool == "antechamber":
            source = Path(args[args.index("-i") + 1])
            if not source.is_absolute():
                source = Path(cwd) / source
            target = _target(args, cwd)
            if args[args.index("-fo") + 1] == "ac":
                _ac_from_mol2(source, target)
            else:
                target.write_text("fake mol2\n")
        elif tool == "parmchk2":
            _target(args, cwd).write_text(
                "Remark\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n\nIMPROPER\n\nNONBON\n"
            )
        else:
            _target(args, cwd).write_text(f"fake prepi {len(self.calls)}\n")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _ether() -> mp.Atomistic:
    """CH3-O-CH2-CH2-O-CH3 with Amber atom names and no ports."""
    graph = mp.Conformer(seed=1).generate(mp.io.read_smiles("COCCOC"))[0]
    for index, atom in enumerate(graph.atoms, start=1):
        atom["name"] = f"{atom['element']}{index}"
    return graph


def _cuts() -> dict[str, dict[str, AmberCut]]:
    """The omit lists and connection atoms are the caller's, written through."""
    return {
        "EO": {
            "head": AmberCut(tail="C4", post_tail="O5", omit=("C1",)),
            "chain": AmberCut(
                head="O2",
                tail="C4",
                pre_head_type="c3",
                post_tail_type="os",
                omit=("C1", "O5", "C6"),
            ),
            "tail": AmberCut(head="O2", pre_head="C1", omit=("C6",)),
        }
    }


def _omit_names(text: str) -> list[str]:
    return [
        line.split()[1] for line in text.splitlines() if line.startswith("OMIT_NAME")
    ]


def _script(work: Path) -> str:
    (script,) = (work / "chains").glob("*/polymer.in")
    return script.read_text()


def _controls(work: Path, label: str = "EO") -> dict[str, str]:
    monomer = work / "monomers" / label
    return {
        name: (monomer / f"{label}.{name}").read_text()
        for name in ("head", "chain", "tail")
    }


@pytest.fixture
def tools(TEST_DATA_DIR: Path):
    fake = _FakeTools(TEST_DATA_DIR / "prmtop" / "LiTFSI.prmtop")
    with (
        patch.object(Wrapper, "is_available", return_value=True),
        patch("molpy.wrapper.base.subprocess.run", side_effect=fake),
        patch(
            "molrs.builder.Assembler.assemble", side_effect=AssertionError("no link")
        ),
    ):
        yield fake


def _build(work: Path, sites: str = "{[#EO]|3}", **kwargs) -> mp.Frame:
    options = {"force_field": "gaff", "work_dir": work} | kwargs
    library = options.pop("library", {"EO": _ether()})
    cuts = options.pop("cuts", _cuts())
    builder = AmberPolymerBuilder(library, cuts, **options)
    return builder.assemble(mp.CGSmilesIR(sites).to_coarsegrain())


class TestAmberPieces:
    def test_pieces_name_the_peo_junction_without_atom_indices(self):
        oligomer, cuts = AmberPieces("COCC", "OCC", "OCCOC").oligomer(seed=1)
        assert oligomer.n_atoms == 30
        assert cuts["head"].tail == "C3"
        assert cuts["chain"].head == "O2"
        assert cuts["chain"].tail == "C5"
        assert cuts["tail"].head == "O3"
        assert cuts["head"].head is None
        assert cuts["head"].post_tail == "O2"
        assert cuts["chain"].pre_head == "C3"
        assert cuts["chain"].post_tail == "O3"
        assert cuts["tail"].pre_head == "C5"

    def test_each_residue_omits_exactly_the_other_two_monomers(self):
        oligomer, cuts = AmberPieces("COCC", "OCC", "OCCOC").oligomer(seed=1)
        names = {str(atom["name"]) for atom in oligomer.atoms}
        kept = {variant: names - set(cut.omit) for variant, cut in cuts.items()}
        assert kept["head"] | kept["chain"] | kept["tail"] == names
        assert not kept["head"] & kept["chain"]
        assert not kept["chain"] & kept["tail"]
        # CH3-O-CH2-CH2 | O-CH2-CH2 | O-CH2-CH2-O-CH3, hydrogens included
        assert [len(kept[v]) for v in ("head", "chain", "tail")] == [11, 7, 12]

    def test_each_piece_is_a_smiles_of_its_own(self):
        with pytest.raises(mp.SmilesError, match="unmatched ring closure"):
            AmberPieces("C1OC", "C1O", "C").oligomer(seed=1)

    def test_pieces_feed_the_builder(self, tools, tmp_path):
        oligomer, cuts = AmberPieces("COCC", "OCC", "OCCOC").oligomer(seed=1)
        result = _build(
            tmp_path, "{[#PEO]|4}", library={"PEO": oligomer}, cuts={"PEO": cuts}
        )
        assert result.monomer_count == 4
        assert "mol = sequence { HPE PEO PEO TPE }" in _script(tmp_path)
        controls = _controls(tmp_path, "PEO")
        assert "TAIL_NAME C3" in controls["head"]
        assert "HEAD_NAME O2" in controls["chain"]
        assert "TAIL_NAME C5" in controls["chain"]
        assert "HEAD_NAME O3" in controls["tail"]


class TestAssemble:
    def test_sequence_is_prepgen_then_tleap_not_assembler(self, tools, tmp_path):
        result = _build(tmp_path, charge_method="bcc")

        assert tools.names() == [
            "antechamber",
            "antechamber",
            "parmchk2",
            "prepgen",
            "prepgen",
            "prepgen",
            "tleap",
        ]
        formats = [
            args[args.index("-fo") + 1]
            for name, args in tools.calls
            if name == "antechamber"
        ]
        assert formats == ["mol2", "ac"]
        first = tools.calls[0][1]
        assert first[first.index("-c") + 1] == "bcc"
        assert first[first.index("-at") + 1] == "gaff"
        assert first[first.index("-nc") + 1] == "0"
        assert first[first.index("-fi") + 1] == "mol2"  # bonds given, not perceived
        source = (tmp_path / "monomers" / "EO" / "EO.input.mol2").read_text()
        assert "@<TRIPOS>BOND" in source
        script = _script(tmp_path)
        assert "source leaprc.gaff" in script
        assert "loadamberparams" in script
        assert "loadamberprep" in script
        assert "loadmol2" not in script
        assert "mol = sequence { HEO EO TEO }" in script
        assert result.monomer_count == 3
        assert result.chain.n_atoms == 16
        assert all(isinstance(atom["type"], str) for atom in result.chain.atoms)
        assert result.prmtop_path.parent == result.inpcrd_path.parent
        assert result.prmtop_path.is_file()
        # the typifiers' likeness: AMBER units, types without prmtop row ids
        assert result.forcefield.units == "real"
        assert list(result.forcefield.get_styles("bond"))
        (atom_style,) = result.forcefield.get_styles("atom")
        assert all("id" not in t.params for t in atom_style.types)

        cuts = _controls(tmp_path)
        assert "HEAD_NAME" not in cuts["head"]
        assert "TAIL_NAME C4" in cuts["head"]
        assert "POST_TAIL_TYPE c3" in cuts["head"]  # O5's type, read from the ac
        assert _omit_names(cuts["head"]) == ["C1"]
        assert "HEAD_NAME O2" in cuts["chain"]
        assert "TAIL_NAME C4" in cuts["chain"]
        assert "PRE_HEAD_TYPE c3" in cuts["chain"]
        assert "POST_TAIL_TYPE os" in cuts["chain"]
        assert _omit_names(cuts["chain"]) == ["C1", "O5", "C6"]
        assert "TAIL_NAME" not in cuts["tail"]
        assert "HEAD_NAME O2" in cuts["tail"]
        assert "PRE_HEAD_TYPE c3" in cuts["tail"]
        assert _omit_names(cuts["tail"]) == ["C6"]
        assert all("CHARGE 0\n" in text for text in cuts.values())
        prepi = tmp_path / "monomers" / "EO" / "HEO.prepi"
        assert prepi.read_text().startswith("fake prepi")

    def test_one_chain_residue_serves_every_middle_site(self, tools, tmp_path):
        result = _build(tmp_path, "{[#EO]|4}")
        assert result.monomer_count == 4
        assert "mol = sequence { HEO EO EO TEO }" in _script(tmp_path)
        assert (tmp_path / "monomers" / "EO" / "EO.prepi").is_file()
        assert (tmp_path / "monomers" / "EO" / "HEO.prepi").is_file()
        assert tools.names().count("prepgen") == 3

    def test_two_sites_need_only_head_and_tail(self, tools, tmp_path):
        cuts = _cuts()
        del cuts["EO"]["chain"]
        _build(tmp_path, "{[#EO]|2}", cuts=cuts)
        assert tools.names().count("prepgen") == 2
        assert "mol = sequence { HEO TEO }" in _script(tmp_path)
        assert not (tmp_path / "monomers" / "EO" / "EO.prepi").exists()

    def test_net_charge_is_the_sum_of_formal_charges(self, tools, tmp_path):
        ether = _ether()
        list(ether.atoms)[1]["formal_charge"] = -1
        _build(tmp_path, library={"EO": ether})
        first = tools.calls[0][1]
        assert first[first.index("-nc") + 1] == "-1"

    def test_declared_net_charges_win(self, tools, tmp_path):
        _build(tmp_path, net_charges={"EO": 2})
        first = tools.calls[0][1]
        assert first[first.index("-nc") + 1] == "2"

    def test_net_charges_must_list_every_label(self, tools, tmp_path):
        with pytest.raises(KeyError, match="no entry for 'EO'"):
            _build(tmp_path, net_charges={"PEO": 0})


class TestCache:
    def test_a_second_build_runs_nothing(self, tools, tmp_path):
        _build(tmp_path)
        done = len(tools.calls)
        _build(tmp_path)
        assert len(tools.calls) == done

    def test_a_changed_cut_reruns_its_prepgen_and_tleap(self, tools, tmp_path):
        _build(tmp_path)
        before = len(tools.calls)
        cuts = _cuts()
        cuts["EO"]["tail"] = AmberCut(head="O2", pre_head="C1", omit=("C6", "H16"))
        _build(tmp_path, cuts=cuts)
        assert tools.names()[before:] == ["prepgen", "tleap"]
        assert _omit_names(_controls(tmp_path)["tail"]) == ["C6", "H16"]
        assert len(list((tmp_path / "chains").iterdir())) == 2

    def test_a_changed_charge_method_reruns_antechamber(self, tools, tmp_path):
        _build(tmp_path)
        before = len(tools.calls)
        _build(tmp_path, charge_method="gas")
        assert tools.names()[before:] == [
            "antechamber",
            "antechamber",
            "parmchk2",
            "prepgen",
            "prepgen",
            "prepgen",
            "tleap",
        ]


class TestGroPoBOligomer:
    """GroPoB's PEO.ac put in place by hand: only prepgen and tleap run."""

    @staticmethod
    def _seed(data: Path, work: Path) -> mp.Atomistic:
        monomer = work / "monomers" / "PEO"
        monomer.mkdir(parents=True)
        shutil.copy(data / "PEO.ac", monomer / "PEO.ac")
        shutil.copy(data / "PEO_initial.mol2", monomer / "PEO.mol2")
        shutil.copy(data / "PEO_initial.frcmod", monomer / "PEO.frcmod")
        frame = mp.io.read_amber_ac(data / "PEO.ac")
        del frame["atoms"]["xyz"]
        return mp.Atomistic.from_frame(frame)

    def test_connection_types_come_from_their_ac(self, tools, tmp_path, TEST_DATA_DIR):
        oligomer = self._seed(TEST_DATA_DIR / "gropob", tmp_path)
        head_methyl = ("C3", "H6", "H7", "H8")
        tail_methyl = ("C7", "H15", "H16", "H17")
        cuts = {
            "PEO": {
                "head": AmberCut(tail="C6", post_tail="C7", omit=tail_methyl),
                "chain": AmberCut(
                    head="C1",
                    tail="C6",
                    pre_head="C3",
                    post_tail="C7",
                    omit=head_methyl + tail_methyl,
                ),
                "tail": AmberCut(head="C1", pre_head="C3", omit=head_methyl),
            }
        }
        _build(tmp_path, "{[#PEO]|5}", library={"PEO": oligomer}, cuts=cuts)
        assert "antechamber" not in tools.names()
        assert "parmchk2" not in tools.names()
        assert "mol = sequence { HPE PEO PEO PEO TPE }" in _script(tmp_path)
        controls = _controls(tmp_path, "PEO")
        assert controls["head"] == (
            "TAIL_NAME C6\nPOST_TAIL_TYPE c3\n"
            "OMIT_NAME C7\nOMIT_NAME H15\nOMIT_NAME H16\nOMIT_NAME H17\nCHARGE 0\n"
        )
        assert controls["chain"].startswith(
            "HEAD_NAME C1\nTAIL_NAME C6\nPRE_HEAD_TYPE c3\nPOST_TAIL_TYPE c3\n"
        )
        assert _omit_names(controls["chain"]) == list(head_methyl + tail_methyl)
        assert controls["tail"].startswith("HEAD_NAME C1\nPRE_HEAD_TYPE c3\n")


class TestRefused:
    def test_a_branched_site_graph(self, tools, tmp_path):
        sites = mp.CoarseGrain()
        beads = [sites.def_bead(bead_type="EO") for _ in range(4)]
        for bead in beads[1:]:
            sites.def_cgbond(beads[0], bead)
        with pytest.raises(ValueError, match="linear"):
            AmberPolymerBuilder({"EO": _ether()}, _cuts(), work_dir=tmp_path).assemble(
                sites
            )

    def test_a_ring(self, tools, tmp_path):
        sites = mp.CoarseGrain()
        beads = [sites.def_bead(bead_type="EO") for _ in range(3)]
        for left, right in zip(beads, beads[1:] + beads[:1], strict=True):
            sites.def_cgbond(left, right)
        with pytest.raises(ValueError, match="linear"):
            AmberPolymerBuilder({"EO": _ether()}, _cuts(), work_dir=tmp_path).assemble(
                sites
            )

    def test_one_site(self, tools, tmp_path):
        with pytest.raises(ValueError, match="head site and a tail site"):
            _build(tmp_path, "{[#EO]}")

    def test_not_a_site_graph(self, tmp_path):
        builder = AmberPolymerBuilder({"EO": _ether()}, _cuts(), work_dir=tmp_path)
        with pytest.raises(TypeError, match="coarse-grain site graph"):
            builder.assemble(_ether())  # type: ignore[arg-type]

    def test_a_label_without_an_oligomer(self, tools, tmp_path):
        with pytest.raises(ValueError, match=r"\['PO'\] are not in the library"):
            _build(tmp_path, "{[#EO][#PO]}")

    def test_a_residue_without_a_cut(self, tools, tmp_path):
        cuts = _cuts()
        del cuts["EO"]["chain"]
        with pytest.raises(ValueError, match=r"no prepgen cut for \['EO chain'\]"):
            _build(tmp_path, cuts=cuts)
        assert tools.calls == []

    @pytest.mark.parametrize(
        ("variant", "cut", "error", "match"),
        [
            (
                "middle",
                AmberCut(head="O2", tail="C4"),
                ValueError,
                "head, chain or tail",
            ),
            (
                "head",
                AmberCut(tail="C4", pre_head_type="c3"),
                ValueError,
                "no PRE_HEAD",
            ),
            ("tail", AmberCut(head="O2", post_tail="C1"), ValueError, "no POST_TAIL"),
            ("chain", AmberCut(head="O2"), ValueError, "needs a tail atom"),
            ("tail", AmberCut(), ValueError, "needs a head atom"),
            ("head", AmberCut(tail="C4", omit=("X9",)), ValueError, r"\['X9'\]"),
            ("head", {"tail": "C4"}, TypeError, "must be AmberCut"),
        ],
    )
    def test_a_bad_cut_before_any_tool_runs(
        self, tools, tmp_path, variant, cut, error, match
    ):
        with pytest.raises(error, match=match):
            AmberPolymerBuilder({"EO": _ether()}, {"EO": {variant: cut}})
        assert tools.calls == []

    def test_bead_types_that_share_a_residue_name(self):
        with pytest.raises(ValueError, match="tleap residue 'HPE'"):
            AmberPolymerBuilder({"PEO": _ether(), "PEOX": _ether()}, {})

    def test_a_repeated_atom_name(self):
        ether = _ether()
        for atom in ether.atoms:
            atom["name"] = "C"
        with pytest.raises(ValueError, match="repeats Amber atom name 'C'"):
            AmberPolymerBuilder({"EO": ether}, {})

    def test_a_template_that_is_not_atomistic(self):
        with pytest.raises(TypeError, match="must be Atomistic"):
            AmberPolymerBuilder({"EO": mp.CoarseGrain()}, {})  # type: ignore[dict-item]

    def test_unnamed_atoms_get_element_and_row(self):
        graph = mp.Conformer(seed=1).generate(mp.io.read_smiles("CO"))[0]
        builder = AmberPolymerBuilder({"MO": graph}, {})
        names = [str(atom["name"]) for atom in builder.library["MO"].atoms]
        assert names[:2] == ["C1", "O2"]
        assert all(atom.get("name") is None for atom in graph.atoms)


class TestToolFailures:
    def test_missing_tool(self, tmp_path):
        with (
            patch.object(Wrapper, "is_available", return_value=False),
            pytest.raises(RuntimeError, match="antechamber is not available"),
        ):
            _build(tmp_path)

    def test_failing_prepgen_carries_its_stdout(self, TEST_DATA_DIR, tmp_path):
        fake = _FakeTools(TEST_DATA_DIR / "prmtop" / "LiTFSI.prmtop", fail="prepgen")
        with (
            patch.object(Wrapper, "is_available", return_value=True),
            patch("molpy.wrapper.base.subprocess.run", side_effect=fake),
            pytest.raises(RuntimeError, match="prepgen: cannot do it"),
        ):
            _build(tmp_path)

    def test_failing_tleap_leaves_no_chain_to_reuse(self, TEST_DATA_DIR, tmp_path):
        prmtop = TEST_DATA_DIR / "prmtop" / "LiTFSI.prmtop"
        with (
            patch.object(Wrapper, "is_available", return_value=True),
            patch(
                "molpy.wrapper.base.subprocess.run",
                side_effect=_FakeTools(prmtop, fail="tleap"),
            ),
            pytest.raises(RuntimeError, match="tleap failed"),
        ):
            _build(tmp_path)
        fake = _FakeTools(prmtop)
        with (
            patch.object(Wrapper, "is_available", return_value=True),
            patch("molpy.wrapper.base.subprocess.run", side_effect=fake),
        ):
            _build(tmp_path)
        assert fake.names() == ["tleap"]


def test_tleap_typifier_refuses_a_graph_that_still_has_ports(tmp_path):
    graph = mp.Conformer(seed=1).generate(mp.io.read_smiles("COC"))[0]
    atoms = list(graph.atoms)
    for atom in atoms:
        atom["type"] = "c3"
        atom["charge"] = 0.0
    graph.def_port(atoms[0], atoms[1], ">")
    with pytest.raises(ValueError, match="ports"):
        mp.typifier.TLeapTypifier(work_dir=tmp_path).typify(graph)
