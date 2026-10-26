"""Prmtop assignment keeps the prmtop's own terms.

A multi-term improper is one type per cosine, named ``<quartet>@<periodicity>``.
A cmap grid stays on its type. ``exclude_14`` stays on the torsion it flags.
``special_bonds`` are the file's majority 1-4 weights. The Amber seed
declares none, the first assignment declares the file's, and a later prmtop
whose weights differ is refused. The prmtop's
special pairs (``pairs``: each pair's ``n_bonds``, and the scale of a 1-4
whose own scale differs from that majority) replace the graph's. A charge
list of the wrong length, a prmtop
with no ``special_bonds``, or a bond the file does not share is refused
before the graph is rewritten. A proper dihedral is
stored in the prmtop's order, which is the order its signed angle is priced in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.tool.ambertools.prmtop import AmberTypifier, Prmtop, PrmtopTypifier

FIXTURES = (
    Path(__file__).resolve().parents[3] / "crates/molrs/src/io/amber/testdata/prmtop"
)

_QUARTET = (1, 16, 3, 4)


def _graph_of(frame: mp.Frame) -> mp.Atomistic:
    """Atoms and bonds in prmtop row order, so prmtop atom *i* is graph atom *i*."""
    graph = mp.Atomistic()
    atoms = [
        graph.add_atom(atomic_number=int(number))
        for number in frame["atoms"]["atomic_number"]
    ]
    bonds = frame["bonds"]
    for left, right in zip(bonds["atomi"], bonds["atomj"], strict=True):
        graph.add_bond(atoms[int(left)], atoms[int(right)])
    return graph


def _index(graph: mp.Atomistic) -> dict[int, int]:
    return {atom.handle: i for i, atom in enumerate(graph.atoms)}


def _ends(index: dict[int, int], relation: mp.core.RelationRef) -> tuple[int, ...]:
    return tuple(index[atom.handle] for atom in relation.endpoints)


def _canonical(atoms: tuple[int, ...]) -> tuple[int, ...]:
    return min(atoms, atoms[::-1])


def _type_names(ff: mp.ff.forcefield.ForceField) -> list[tuple[str, str, str]]:
    return [
        (style.category, style.name, term.name)
        for style in ff.get_styles(mp.ff.forcefield.Style)
        for term in style.get_types()
    ]


def _weights(ff: mp.ff.forcefield.ForceField) -> tuple[list[float], list[float]]:
    weights = ff.special_bonds
    assert weights is not None
    return weights


def _endpoint_pairs(block: mp.Block) -> list[tuple[int, int]]:
    return [
        (int(i), int(j)) for i, j in zip(block["atomi"], block["atomj"], strict=True)
    ]


def _stated_column(block: mp.Block, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``key``'s values and which cells are real; absent when the block lacks it."""
    if key not in block:
        return None
    values = np.asarray(block[key], dtype=np.float64)
    mask = block.validity(key)
    valid = (
        np.ones(values.shape[0], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    return values, valid


def _proper_rows(
    frame: mp.Frame,
) -> list[tuple[tuple[int, ...], str, bool]]:
    """Proper dihedrals as ``(endpoints, type, exclude_14)``, in file order."""
    block = frame["dihedrals"]
    columns = [block[key] for key in ("atomi", "atomj", "atomk", "atoml")]
    return [
        (tuple(int(atom) for atom in atoms), str(name), bool(flag))
        for atoms, name, flag in zip(
            zip(*columns, strict=True),
            block["type"],
            block["exclude_14"],
            strict=True,
        )
    ]


def _relation_pairs(frame: mp.Frame, block: str) -> list[tuple[int, int]]:
    if block not in frame or frame[block].n_rows == 0:
        return []
    return _endpoint_pairs(frame[block])


def _term(ff: mp.ff.forcefield.ForceField, category: str, style: str, name: str):
    found = ff.get_style(category, style)
    assert found is not None
    term = found.get_type_by_name(name)
    assert term is not None
    return term


class _TwoPrmtops(mp.ff.typifier.Typifier[mp.Atomistic]):
    """Assign two prmtops, in order, into one output."""

    def __init__(self, first: Path, second: Path) -> None:
        super().__init__()
        self._files = [Prmtop(first), Prmtop(second)]
        self._n = 0

    def source_forcefield(self) -> mp.ff.forcefield.ForceField:
        return AmberTypifier().source_forcefield()

    def assign(self, graph: mp.Atomistic) -> mp.ff.typifier.TypeAssignment:
        prmtop = self._files[self._n]
        self._n += 1
        return prmtop.assignment(graph, prmtop.frame["atoms"]["charge"])


def test_multi_term_impropers_keep_each_cosine_and_its_14_flag() -> None:
    path = FIXTURES / "gaff2_multi.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    typifier = PrmtopTypifier(path)
    typed = typifier.typify(_graph_of(frame))
    index = _index(typed)

    hits = [imp for imp in typed.impropers if _ends(index, imp) == _QUARTET]
    by_name = {imp["type"]: imp for imp in hits}
    assert set(by_name) == {"c3-ca-na-cc@2", "c3-ca-na-cc@3"}
    assert by_name["c3-ca-na-cc@2"]["k"] == pytest.approx(1.1)
    assert by_name["c3-ca-na-cc@2"]["periodicity"] == pytest.approx(2)
    assert by_name["c3-ca-na-cc@2"]["phase"] == pytest.approx(180.0)
    assert by_name["c3-ca-na-cc@2"].get("exclude_14") is True
    assert by_name["c3-ca-na-cc@3"]["k"] == pytest.approx(0.35)
    assert by_name["c3-ca-na-cc@3"]["periodicity"] == pytest.approx(3)
    assert by_name["c3-ca-na-cc@3"]["phase"] == pytest.approx(0.0)
    assert by_name["c3-ca-na-cc@3"].get("exclude_14") is True

    ff = typifier.forcefield()
    assert _term(ff, "improper", "periodic", "c3-ca-na-cc@2").params[
        "k"
    ] == pytest.approx(1.1)
    assert _term(ff, "improper", "periodic", "c3-ca-na-cc@3").params[
        "k"
    ] == pytest.approx(0.35)

    expected: dict[tuple[int, ...], tuple[str, bool]] = {}
    block = frame["dihedrals"]
    columns = [block[key] for key in ("atomi", "atomj", "atomk", "atoml")]
    for atoms, name, flag in zip(
        zip(*columns, strict=True), block["type"], block["exclude_14"], strict=True
    ):
        key = _canonical(tuple(int(atom) for atom in atoms))
        expected[key] = (str(name), bool(flag))
    for dihedral in typed.dihedrals:
        name, flag = expected[_canonical(_ends(index, dihedral))]
        assert dihedral["type"] == name
        assert dihedral.get("exclude_14") is flag


def test_glycam_keeps_the_file_14_weights() -> None:
    path = FIXTURES / "glycam.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    typifier = PrmtopTypifier(path)
    assert typifier.forcefield().special_bonds is None

    typed = typifier.typify(_graph_of(frame))
    lj, coul = _weights(typifier.forcefield())
    assert lj == pytest.approx([0.0, 0.0, 1.0])
    assert coul == pytest.approx([0.0, 0.0, 1.0])

    index = _index(typed)
    expected: dict[tuple[int, ...], bool] = {}
    block = frame["dihedrals"]
    columns = [block[key] for key in ("atomi", "atomj", "atomk", "atoml")]
    for atoms, flag in zip(
        zip(*columns, strict=True), block["exclude_14"], strict=True
    ):
        expected[_canonical(tuple(int(atom) for atom in atoms))] = bool(flag)
    assert any(expected.values())
    for dihedral in typed.dihedrals:
        assert (
            dihedral.get("exclude_14") is expected[_canonical(_ends(index, dihedral))]
        )


def test_ff19sb_cmap_grid_reaches_the_assigned_type() -> None:
    path = FIXTURES / "ff19sb.parm7"
    reader, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    source = _term(reader, "cmap", "charmm", "C-N-XC-C-N")
    typifier = PrmtopTypifier(path)
    typed = typifier.typify(_graph_of(frame))
    assigned = _term(typifier.forcefield(), "cmap", "charmm", "C-N-XC-C-N")
    grid = np.asarray(assigned.params["grid"])
    assert grid.shape == (24, 24)
    np.testing.assert_allclose(grid, np.asarray(source.params["grid"]))
    terms = list(typed.terms("cmaps"))
    assert [term["type"] for term in terms] == [
        str(name) for name in frame["cmaps"]["type"]
    ]


def test_a_later_prmtop_with_other_14_weights_is_refused() -> None:
    glycam = FIXTURES / "glycam.parm7"
    amber = FIXTURES / "ff14sb.parm7"
    _, glycam_frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(glycam)
    _, amber_frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(amber)
    typifier = _TwoPrmtops(glycam, amber)
    typifier.typify(_graph_of(glycam_frame))
    before = _type_names(typifier.forcefield())
    with pytest.raises(ValueError, match="special_bonds"):
        typifier.typify(_graph_of(amber_frame))
    lj, coul = _weights(typifier.forcefield())
    assert lj == pytest.approx([0.0, 0.0, 1.0])
    assert coul == pytest.approx([0.0, 0.0, 1.0])
    assert _type_names(typifier.forcefield()) == before


_SCALES = ("coul_scale", "lj_scale", "epsilon", "sigma", "charge_product")


def _assert_same_pairs(out: mp.Frame, source: mp.Frame) -> None:
    """``out``'s special pairs are ``source``'s: rows, classes and cells."""
    assert _relation_pairs(out, "pairs") == _relation_pairs(source, "pairs")
    np.testing.assert_array_equal(
        np.asarray(out["pairs"]["n_bonds"]), np.asarray(source["pairs"]["n_bonds"])
    )
    for key in _SCALES:
        found = _stated_column(out["pairs"], key)
        wanted = _stated_column(source["pairs"], key)
        if wanted is None:
            assert found is None or not found[1].any()
            continue
        assert found is not None
        np.testing.assert_array_equal(found[1], wanted[1])
        np.testing.assert_allclose(found[0][found[1]], wanted[0][wanted[1]])


def test_glycam_per_pair_14_scales_reach_the_typed_graph() -> None:
    path = FIXTURES / "glycam.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    typed = PrmtopTypifier(path).typify(_graph_of(frame))
    out = typed.to_frame()

    pairs = out["pairs"]
    one_four = np.asarray(pairs["n_bonds"]) == 3
    coul, coul_stated = _stated_column(pairs, "coul_scale")
    lj, lj_stated = _stated_column(pairs, "lj_scale")
    own = one_four & coul_stated
    assert own.any()
    np.testing.assert_allclose(coul[own], 1.0 / 1.2)
    np.testing.assert_allclose(lj[one_four & lj_stated], 1.0 / 2.0)
    _assert_same_pairs(out, frame)


def test_a_uniform_prmtop_states_no_pair_scale() -> None:
    path = FIXTURES / "ff14sb.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    typed = PrmtopTypifier(path).typify(_graph_of(frame))
    out = typed.to_frame()

    _assert_same_pairs(out, frame)
    assert out["pairs"].n_rows > 0
    for key in _SCALES:
        stated = _stated_column(out["pairs"], key)
        assert stated is None or not stated[1].any(), key


def test_pairs_already_on_the_graph_are_replaced_by_the_prmtop() -> None:
    path = FIXTURES / "glycam.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    graph = _graph_of(frame)
    atoms = list(graph.atoms)
    graph.add_term("pairs", [atoms[0], atoms[5]], n_bonds=0, coul_scale=0.3)

    out = PrmtopTypifier(path).typify(graph).to_frame()
    _assert_same_pairs(out, frame)


def test_a_missing_bond_is_refused_before_the_graph_is_rewritten() -> None:
    path = FIXTURES / "glycam.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    graph = _graph_of(frame)
    graph.del_bonds([next(iter(graph.bonds))])
    n_bonds = graph.n_bonds
    with pytest.raises(ValueError, match="bond"):
        Prmtop(path).assignment(graph, frame["atoms"]["charge"])
    assert graph.n_bonds == n_bonds
    assert graph.n_angles == 0
    assert graph.n_dihedrals == 0
    assert graph.n_impropers == 0


def test_a_short_charge_list_is_refused_before_the_graph_is_rewritten() -> None:
    path = FIXTURES / "glycam.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    graph = _graph_of(frame)
    charges = [float(charge) for charge in frame["atoms"]["charge"]]
    with pytest.raises(ValueError, match="charge"):
        Prmtop(path).assignment(graph, charges[:-1])
    assert graph.n_angles == 0
    assert graph.n_dihedrals == 0
    assert graph.n_impropers == 0


def test_a_prmtop_without_special_bonds_is_refused_before_rewrite() -> None:
    path = FIXTURES / "glycam.parm7"
    _, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)
    graph = _graph_of(frame)
    prmtop = Prmtop(path)
    prmtop.forcefield = AmberTypifier().source_forcefield()
    with pytest.raises(ValueError, match="special_bonds"):
        prmtop.assignment(graph, frame["atoms"]["charge"])
    assert graph.n_angles == 0
    assert graph.n_dihedrals == 0
    assert graph.n_impropers == 0
