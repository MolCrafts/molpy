"""Python-binding coverage for ``GaffTypifier`` and ``ForceField.materialize_params``.

The GAFF / GAFF2 matching itself (table rows, parmchk2's estimates, tleap's
impropers) is owned by the Rust tests in ``molrs/src/ff/typifier/gaff`` and
checked there against an AmberTools-built prmtop; these tests prove the
binding seam: construction, the ATD → GAFF composition, the output force
field, and the per-row parameter columns. The two pinned numbers below are
parmchk2's (AmberTools 26.1) estimates for ethylene, quoted, not recomputed.
"""

from __future__ import annotations

import math
from pathlib import Path

import molrs
import numpy as np
import pytest


def _acetanilide() -> molrs.core.Atomistic:
    """Acetanilide with hydrogens and 3D coordinates."""
    heavy = molrs.io.smiles.SmilesIr("CC(=O)Nc1ccccc1").to_atomistic()
    mol, _ = molrs.conformer.Conformer(seed=7).generate(heavy)
    return mol


def _typed(parameter_set: str, mol: molrs.core.Atomistic | None = None):
    """``(typifier, typed Atomistic)``: ATD types, then the GAFF table."""
    mol = _acetanilide() if mol is None else mol
    labelled = molrs.ff.typifier.AtdTypifier(parameter_set=parameter_set).typify(mol)
    gaff = molrs.ff.typifier.GaffTypifier(parameter_set=parameter_set)
    return gaff, gaff.typify(labelled)


def _ethylene() -> molrs.core.Atomistic:
    mol = molrs.core.Atomistic()
    c1 = mol.def_atom(element="C", type="c2")
    c2 = mol.def_atom(element="C", type="c2")
    mol.def_bond(c1, c2)
    for c in (c1, c1, c2, c2):
        mol.def_bond(c, mol.def_atom(element="H", type="ha"))
    return mol


# ---------------------------------------------------------------------------
# GaffTypifier
# ---------------------------------------------------------------------------


def test_gaff_typifier_is_exposed() -> None:
    assert "GaffTypifier" in molrs.ff.typifier.__all__
    assert molrs.ff.typifier.GaffTypifier is molrs.ff.typifier.GaffTypifier
    gaff = molrs.ff.typifier.GaffTypifier(parameter_set="gaff2")
    assert isinstance(gaff, molrs.ff.typifier.Typifier)
    assert gaff.parameter_set == "gaff2"
    assert repr(gaff) == "GaffTypifier(parameter_set='gaff2')"


def test_the_parameter_set_is_required_and_checked() -> None:
    with pytest.raises(TypeError):
        molrs.ff.typifier.GaffTypifier()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="gaff, gaff2"):
        molrs.ff.typifier.GaffTypifier(parameter_set="amber")


@pytest.mark.parametrize("parameter_set", ["gaff", "gaff2"])
def test_atd_then_gaff_types_and_prices_acetanilide(parameter_set: str) -> None:
    gaff, typed = _typed(parameter_set)
    frame = typed.to_frame()
    assert frame["bonds"].n_rows == 19
    assert frame["angles"].n_rows == 30
    assert frame["dihedrals"].n_rows == 38
    assert frame["impropers"].n_rows > 0
    ff = gaff.forcefield()
    assert ff.name == parameter_set
    assert ff.special_bonds == ([0.0, 0.0, 0.5], [0.0, 0.0, pytest.approx(1 / 1.2)])
    styles = {(s.category, s.name) for s in ff.styles}
    assert ("improper", "periodic") in styles
    # Charges are not a GAFF parameter: a charge model supplies them.
    atoms = frame["atoms"]
    atoms.insert("charge", molrs.ff.charge.GasteigerModel().assign(typed))
    frame["atoms"] = atoms
    frame["pairs"] = molrs.ff.potential.intramolecular_pairs(frame, ff)
    energy = molrs.ff.compile.PotentialCompiler(ff).compile(frame).calc_energy(frame)
    assert math.isfinite(energy)


def test_an_untyped_atom_is_refused() -> None:
    gaff = molrs.ff.typifier.GaffTypifier(parameter_set="gaff")
    with pytest.raises(ValueError, match="carries no"):
        gaff.typify(_acetanilide())


@pytest.mark.parametrize(
    ("parameter_set", "k", "analog"),
    [("gaff", 1.1, "X-X-ca-ha"), ("gaff2", 10.5, "X-X-cc-X")],
)
def test_ethylene_impropers_are_parmchk2s(
    parameter_set: str, k: float, analog: str
) -> None:
    """parmchk2: ``c2-ha-c2-ha 1.1 Same as X -X -ca-ha`` (GAFF),
    ``10.5 Same as X -X -cc-X`` (GAFF2)."""
    gaff = molrs.ff.typifier.GaffTypifier(parameter_set=parameter_set)
    typed = gaff.typify(_ethylene())
    assert typed.to_frame()["impropers"].n_rows == 2
    (improper,) = gaff.forcefield().get_types("improper")
    assert improper.params["k"] == k
    assert improper.params["estimate_analog"] == analog


def test_a_native_gaff_subclass_cannot_override_match() -> None:
    with pytest.raises(TypeError, match="native typifier"):

        class _Bad(molrs.ff.typifier.GaffTypifier):  # pragma: no cover - refused
            def assign(self, graph):
                return None


# ---------------------------------------------------------------------------
# ForceField.materialize_params
# ---------------------------------------------------------------------------


def _column(frame: molrs.core.Frame, block: str, key: str) -> np.ndarray:
    return np.asarray(frame[block][key])


def test_gaff2_reference_columns_are_the_typed_parameters() -> None:
    gaff, typed = _typed("gaff2")
    ff = gaff.forcefield()
    frame = typed.to_frame()
    written = ff.materialize_params(frame, prefix="gaff2_")

    assert {"gaff2_k", "gaff2_r0"} <= set(written["bonds"])
    assert {"gaff2_k", "gaff2_theta0"} <= set(written["angles"])
    assert {"gaff2_epsilon", "gaff2_sigma", "gaff2_mass"} <= set(written["atoms"])
    assert "gaff2_k1" in written["dihedrals"]
    assert "gaff2_k" in written["impropers"]

    by_name = {t.name: t.params for t in ff.get_types("bond")}
    for label, k, r0 in zip(
        frame["bonds"]["type"],
        _column(frame, "bonds", "gaff2_k"),
        _column(frame, "bonds", "gaff2_r0"),
    ):
        assert by_name[str(label)]["k"] == k
        assert by_name[str(label)]["r0"] == r0
    # theta0 in degrees, as the force-field IR stores it.
    theta0 = _column(frame, "angles", "gaff2_theta0")
    assert 100.0 < theta0.min() and theta0.max() < 130.0


def test_a_parameter_a_rows_type_lacks_is_a_null_cell() -> None:
    """Two bond styles in one block: each row carries its own style's
    parameters, and the other style's are holes, not zeros."""
    ff = molrs.ff.forcefield.ForceField("toy")
    atoms = ff.def_style("atom", "full")
    a = atoms.def_type("A", mass=12.0)
    b = atoms.def_type("B", mass=1.0)
    ff.def_style("bond", "harmonic").def_type("A-B", a, b, k=300.0, r0=1.1)
    ff.def_style("bond", "morse").def_type("A-A", a, a, d0=80.0, alpha=2.0, r0=1.5)

    frame = molrs.core.Frame()
    block = molrs.core.Block()
    block.insert("type", ["A", "A", "B"])
    frame["atoms"] = block
    bonds = molrs.core.Block()
    bonds.insert("atomi", np.array([0, 1], dtype=np.uint64))
    bonds.insert("atomj", np.array([1, 2], dtype=np.uint64))
    bonds.insert("type", ["A-A", "A-B"])
    frame["bonds"] = bonds

    written = ff.materialize_params(frame, prefix="ref_")
    assert written["bonds"] == ["ref_alpha", "ref_d0", "ref_k", "ref_r0"]
    np.testing.assert_array_equal(_column(frame, "bonds", "ref_r0"), [1.5, 1.1])
    np.testing.assert_array_equal(
        np.asarray(frame["bonds"].validity("ref_k")), [False, True]
    )
    np.testing.assert_array_equal(
        np.asarray(frame["bonds"].validity("ref_d0")), [True, False]
    )
    assert frame["bonds"].validity("ref_r0") is None


def test_per_atom_lennard_jones_is_the_self_row() -> None:
    ff = molrs.ff.forcefield.ForceField("toy")
    atoms = ff.def_style("atom", "full")
    a = atoms.def_type("A", mass=12.0)
    b = atoms.def_type("B", mass=1.0)
    lj = ff.def_style("pair", "lj/cut")
    lj.def_type("A", a, epsilon=0.1, sigma=3.4)
    lj.def_type("B", b, epsilon=0.02, sigma=2.5)
    ff.def_style("pair", "coul/cut", {"coulomb": 332.0})
    ff.def_style("bond", "harmonic").def_type("A-B", a, b, k=300.0, r0=1.1)

    frame = molrs.core.Frame()
    block = molrs.core.Block()
    block.insert("type", ["A", "B"])
    frame["atoms"] = block
    bonds = molrs.core.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint64))
    bonds.insert("atomj", np.array([1], dtype=np.uint64))
    bonds.insert("type", ["A-B"])
    frame["bonds"] = bonds

    written = ff.materialize_params(frame, prefix="")
    assert written == {"bonds": ["k", "r0"], "atoms": ["epsilon", "mass", "sigma"]}
    np.testing.assert_array_equal(_column(frame, "atoms", "sigma"), [3.4, 2.5])
    np.testing.assert_array_equal(_column(frame, "atoms", "epsilon"), [0.1, 0.02])
    np.testing.assert_array_equal(_column(frame, "bonds", "k"), [300.0])


def test_a_frame_of_another_field_is_refused() -> None:
    gaff, typed = _typed("gaff2")
    frame = typed.to_frame()
    other = molrs.ff.forcefield.ForceField("other")
    other.def_style("bond", "harmonic")
    with pytest.raises(ValueError, match="no bond style defines"):
        other.materialize_params(frame, prefix="x_")


# ---------------------------------------------------------------------------
# AtdTypifier bond orders — antechamber's Kekulé structure
# ---------------------------------------------------------------------------


def _mol2(
    elements: list[str], bonds: list[tuple[int, int]], orders=None
) -> molrs.core.Atomistic:
    """A molecule as a mol2 file lists it: atoms by element, bonds in file
    order, single unless ``orders`` says otherwise."""
    mol = molrs.core.Atomistic()
    atoms = [mol.def_atom(element=e) for e in elements]
    for k, (i, j) in enumerate(bonds):
        order = 1 if orders is None else orders[k]
        mol.def_bond(atoms[i], atoms[j], bond_type=order, bond_number=order)
    return mol


_COT_BONDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)] + [
    (c, 8 + c) for c in range(8)
]
_AZULENE_BONDS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (3, 7),
    (7, 8),
    (8, 9),
    (0, 9),
] + [(c, 10 + h) for h, c in enumerate([0, 1, 2, 4, 5, 6, 8, 9])]


def _types(mol: molrs.core.Atomistic, parameter_set: str, **kw) -> list[str]:
    typed = molrs.ff.typifier.AtdTypifier(parameter_set=parameter_set, **kw).typify(mol)
    return [str(t) for t in typed.to_frame()["atoms"]["type"]]


@pytest.mark.parametrize("parameter_set", ["gaff", "gaff2"])
def test_azulene_and_cyclooctatetraene_type_as_antechamber(parameter_set: str) -> None:
    # antechamber -at gaff / gaff2 (AmberTools 26.1) on the same mol2 files.
    azulene = _mol2(["C"] * 10 + ["H"] * 8, _AZULENE_BONDS)
    assert (
        " ".join(_types(azulene, parameter_set)[:10]) == "cc cc cd cd cc cc cd cd cc cc"
    )
    cot = _mol2(["C"] * 8 + ["H"] * 8, _COT_BONDS)
    assert " ".join(_types(cot, parameter_set)[:8]) == "cc cc cd cd cc cc cd cd"


def test_the_drawn_kekule_structure_is_kept_only_when_asked() -> None:
    drawn = [2, 1, 2, 1, 2, 1, 2, 1] + [1] * 8
    cot = _mol2(["C"] * 8 + ["H"] * 8, _COT_BONDS, drawn)
    # antechamber ignores the input's orders; so does the default.
    assert " ".join(_types(cot, "gaff")[:8]) == "cc cc cd cd cc cc cd cd"
    assert (
        " ".join(_types(cot, "gaff", bond_orders="input")[:8])
        == "cc cd cd cc cc cd cd cc"
    )


def test_bond_orders_is_checked_and_reported() -> None:
    atd = molrs.ff.typifier.AtdTypifier(parameter_set="gaff")
    assert atd.bond_orders == "perceive"
    assert repr(atd) == "AtdTypifier(parameter_set='gaff', bond_orders='perceive')"
    assert (
        molrs.ff.typifier.AtdTypifier(
            parameter_set="bcc", bond_orders="input"
        ).bond_orders
        == "input"
    )
    with pytest.raises(ValueError, match="perceive"):
        molrs.ff.typifier.AtdTypifier(parameter_set="gaff", bond_orders="kekule")


def test_find_bond_orders_judges_from_connectivity() -> None:
    cot = _mol2(["C"] * 8 + ["H"] * 8, _COT_BONDS)
    out = molrs.perceive.assign_bond_orders(cot)
    numbers = [int(n) for n in out.to_frame()["bonds"]["bond_number"]]
    assert numbers[:8] == [1, 2, 1, 2, 1, 2, 1, 2]


# ---------------------------------------------------------------------------
# One force field from the native route and from tleap's prmtop
# ---------------------------------------------------------------------------

#: AmberTools 26.0 builds of the molecules below (``scripts/gaff_merge_fixtures.sh``).
GAFF_FIXTURES = Path(__file__).resolve().parents[2] / "molrs/src/io/amber/testdata/gaff"


def _ethanol() -> molrs.core.Atomistic:
    """Ethanol in the atom and bond order of ``ethanol_gaff.mol2``."""
    mol = molrs.core.Atomistic()
    c1, c2, o1 = (mol.def_atom(element=e) for e in ("C", "C", "O"))
    h = [mol.def_atom(element="H") for _ in range(6)]
    bonds = [(c1, c2), (c1, h[0]), (c1, h[1]), (c1, h[2])]
    bonds += [(c2, o1), (c2, h[3]), (c2, h[4]), (o1, h[5])]
    for i, j in bonds:
        mol.def_bond(i, j)
    return mol


def _type_names(ff: molrs.ff.forcefield.ForceField) -> set[tuple[str, str, str]]:
    return {(s.category, s.name, t.name) for s in ff.styles for t in s.get_types()}


def test_native_gaff_and_the_prmtop_of_ethanol_merge_into_one() -> None:
    """The native route names terms as the prmtop reader does (``c3-c3-oh-ho``,
    terms by periodicity) and prices them alike, so ``merge`` takes the two
    either way round and adds no type."""
    gaff, _ = _typed("gaff", _ethanol())
    native = gaff.forcefield()
    torsion = {t.name: t.params for t in native.get_types("dihedral")}["c3-c3-oh-ho"]
    assert (torsion["periodicity1"], torsion["k1"]) == (1.0, 0.25)
    prmtop = molrs.io.read_prmtop_forcefield(GAFF_FIXTURES / "ethanol_gaff.parm7")
    assert _type_names(prmtop) <= _type_names(native)
    names = _type_names(native)
    assert _type_names(prmtop.merge(native)) == names
    again, _ = _typed("gaff", _ethanol())
    prmtop = molrs.io.read_prmtop_forcefield(GAFF_FIXTURES / "ethanol_gaff.parm7")
    assert _type_names(again.forcefield().merge(prmtop)) == names
