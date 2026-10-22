"""VirtualSiteBuilder / DrudeBuilder / Tip4pBuilder.

Drude tests read a hand-typed CL&P [C4C1im]+ (``mol2/c4c1im_clp_typed.mol2``);
Tip4p uses plain water.
"""

from __future__ import annotations


import math
from pathlib import Path

import pytest

import molpy as mp

#: 4πε₀ in e² / (kJ/mol·Å): the paduagroup/clandpol polarizer's value.
FOUR_PI_EPS0 = 0.0007197587


@pytest.fixture
def cation(TEST_DATA_DIR: Path) -> mp.Atomistic:
    """[C4C1im]+ with CL&P types and charges; the fixture names atoms by element."""
    frame = mp.io.read_mol2(TEST_DATA_DIR / "mol2" / "c4c1im_clp_typed.mol2")
    frame["atoms"]["element"] = frame["atoms"]["name"]
    mp.Element.fill_atomic_numbers(frame)
    return mp.Atomistic.from_frame(frame)


def _water(charge_o: float = -0.8, charge_h: float = 0.4):
    asm = mp.Atomistic()
    o = asm.add_atom(atomic_number=8, charge=charge_o, x=0.0, y=0.0, z=0.0)
    h1 = asm.add_atom(atomic_number=1, charge=charge_h, x=0.757, y=0.586, z=0.0)
    h2 = asm.add_atom(atomic_number=1, charge=charge_h, x=-0.757, y=0.586, z=0.0)
    asm.add_bond(o, h1)
    asm.add_bond(o, h2)
    return asm, o


def _drudes(struct):
    return [a for a in struct.atoms if a.get("vsite") == "drude"]


def _rows(struct, block):
    """The ``block`` relation rows of ``struct``'s frame, one dict per row."""
    frame = struct.to_frame()
    if block not in frame.keys():
        return []
    blk = frame[block]
    cols = {k: blk[k].tolist() for k in blk.keys()}
    n = len(next(iter(cols.values())))
    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


class TestVirtualSiteBuilder:
    def test_builder_is_an_abstract_transform(self):
        with pytest.raises(TypeError):
            mp.builder.VirtualSiteBuilder()


class TestDrudeBuilder:
    def test_is_a_virtual_site_transform(self):
        assert issubclass(mp.builder.DrudeBuilder, mp.builder.VirtualSiteBuilder)


class TestTip4pBuilder:
    def test_is_a_virtual_site_transform(self):
        assert issubclass(mp.builder.Tip4pBuilder, mp.builder.VirtualSiteBuilder)


def test_builders_are_subclasses():
    assert issubclass(mp.builder.DrudeBuilder, mp.builder.VirtualSiteBuilder)
    assert issubclass(mp.builder.Tip4pBuilder, mp.builder.VirtualSiteBuilder)


def test_the_clpol_table_is_molrs_s():
    table = mp.ff.params.clpol_polarizability()
    assert table["CR"]["k_D"] == 4184.0 and table["CR"]["alpha"] > 0
    assert table["HC"]["k_D"] == 0.0


def _atoms_by_row(struct):
    return list(struct.atoms)


def test_drude_shell_is_typed_from_core(cation):
    out = mp.builder.DrudeBuilder().apply(cation)
    atoms = _atoms_by_row(out)
    rows = _rows(out, "drudes")
    assert rows
    for row in rows:
        core, shell = atoms[row["atomi"]], atoms[row["atomj"]]
        assert core.get("vsite") is None and shell.get("vsite") == "drude"
        assert shell.get("type") == "D" + core.get("type")
        assert row["type"] == f"{core.get('type')}-{shell.get('type')}"


def test_drude_shell_prefix_is_configurable(cation):
    out = mp.builder.DrudeBuilder(drude_prefix="DP_").apply(cation)
    assert all(s.get("type").startswith("DP_") for s in _drudes(out))


def test_drude_apply_does_not_mutate_input(cation):
    struct = cation
    n_before = len(list(struct.atoms))
    q_before = sum(a.get("charge") for a in struct.atoms)
    out = mp.builder.DrudeBuilder().apply(struct)
    assert out is not struct
    assert len(list(struct.atoms)) == n_before
    assert sum(a.get("charge") for a in struct.atoms) == q_before
    assert "drudes" not in struct.term_blocks()


def test_drude_count_matches_heavy_atoms_no_hydrogen(cation):
    struct = cation
    out = mp.builder.DrudeBuilder().apply(struct)
    heavy = [a for a in struct.atoms if a.get("atomic_number") != 1]
    assert len(_drudes(out)) == len(heavy)
    for a in out.atoms:
        if a.get("atomic_number") == 1:
            assert a.get("vsite") is None


def test_drude_springs_are_drudes_rows_not_bonds(cation):
    """The core–shell spring is a molrs ``drudes`` row, never a ``bonds`` row."""
    out = mp.builder.DrudeBuilder().apply(cation)
    rows = _rows(out, "drudes")
    assert len(rows) == len(_drudes(out))
    assert {r["style"] for r in rows} == {"harmonic"}
    shells = {a.handle for a in _drudes(out)}
    for bond in out.bonds:
        assert bond.itom.handle not in shells and bond.jtom.handle not in shells
    assert len(list(out.bonds)) == len(list(cation.bonds))


def test_drude_spring_force_constant(cation):
    """alpha.ff's k_D is kJ/mol/Å² in the ½k r² form; molrs's
    ``drude/harmonic`` is ``k*r^2`` in kcal/mol, so k = k_D / 2 in kcal."""
    out = mp.builder.DrudeBuilder().apply(cation)
    kcal_per_kj = mp.core.UnitRegistry().factor("kJ", "kcal")
    rows = _rows(out, "drudes")
    assert all(r["k"] == pytest.approx(4184.0 / 2 * kcal_per_kj) for r in rows)
    assert rows[0]["k"] == pytest.approx(500.0)


def test_drude_rows_carry_alpha_and_thole(cation):
    """CL&Pol's ``a_thole`` survives as the ``drudes`` row's ``thole``."""
    out = mp.builder.DrudeBuilder().apply(cation)
    table = mp.ff.params.clpol_polarizability()
    atoms = _atoms_by_row(out)
    rows = _rows(out, "drudes")
    assert rows
    for row in rows:
        params = table[atoms[row["atomi"]].get("type")]
        assert row["alpha"] == params["alpha"]
        assert row["thole"] == params["a_thole"] == 2.6


def test_drude_shell_has_no_ad_hoc_columns_or_element(cation):
    """Spring data lives on the ``drudes`` row; a shell is marked by ``vsite``,
    not by a non-IUPAC ``element``."""
    out = mp.builder.DrudeBuilder().apply(cation)
    assert _drudes(out)
    for shell in _drudes(out):
        assert "k_D" not in shell and "alpha" not in shell
        assert "element" not in shell


def test_alpha_recovered_from_drude_params(cation):
    out = mp.builder.DrudeBuilder().apply(cation)
    kcal_per_kj = mp.core.UnitRegistry().factor("kJ", "kcal")
    atoms = _atoms_by_row(out)
    rows = _rows(out, "drudes")
    assert rows
    for row in rows:
        q_d = atoms[row["atomj"]].get("charge")
        k_d = 2.0 * row["k"] / kcal_per_kj
        assert q_d**2 / (FOUR_PI_EPS0 * k_d) == pytest.approx(row["alpha"], rel=1e-7)
        assert row["alpha"] > 0
    assert mp.ff.params.clpol_polarizability()["CR"]["alpha"] == 1.122


def test_cation_charge_conserved(cation):
    out = mp.builder.DrudeBuilder().apply(cation)
    total = sum(a.get("charge") for a in out.atoms)
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_drude_accepts_read_clpol_alpha_rows(cation, tmp_path):
    """``mp.io.read_clpol_alpha`` returns a list of rows; the builder takes it."""
    table = mp.ff.params.clpol_polarizability()
    lines = [
        f"{t} {p['m_D']} {p['q_D_sign']} {p['k_D']} {p['alpha']} {p['a_thole']}"
        for t, p in table.items()
    ]
    path = tmp_path / "alpha.ff"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = mp.io.read_clpol_alpha(path)
    assert isinstance(rows, list)
    from_rows = mp.builder.DrudeBuilder(rows).apply(cation)
    default = mp.builder.DrudeBuilder().apply(cation)
    assert _rows(from_rows, "drudes") == _rows(default, "drudes")


def test_drude_later_row_wins(cation):
    """A type given twice keeps the file's last word (``read_clpol_alpha``)."""
    table = mp.ff.params.clpol_polarizability()
    rows = [{"type_name": t, **p} for t, p in table.items()]
    rows.append({**rows[[r["type_name"] for r in rows].index("CR")], "a_thole": 1.5})
    out = mp.builder.DrudeBuilder(rows).apply(cation)
    atoms = _atoms_by_row(out)
    tholes = {atoms[r["atomi"]].get("type"): r["thole"] for r in _rows(out, "drudes")}
    assert tholes["CR"] == 1.5


def test_drude_rejects_untyped_graph():
    """An untyped graph raises instead of silently gaining zero shells."""
    asm = mp.Atomistic()
    c = asm.add_atom(atomic_number=6, charge=0.0, x=0.0, y=0.0, z=0.0)
    h = asm.add_atom(atomic_number=1, charge=0.0, x=1.0, y=0.0, z=0.0)
    asm.add_bond(c, h)
    with pytest.raises(ValueError, match="type"):
        mp.builder.DrudeBuilder().apply(asm)


def test_drude_rejects_incomplete_table_row():
    with pytest.raises(ValueError, match="a_thole"):
        mp.builder.DrudeBuilder(
            {"CR": {"m_D": 0.4, "q_D_sign": -1.0, "k_D": 4184.0, "alpha": 1.0}}
        )


def _msite(out):
    msites = [a for a in out.atoms if a.get("vsite") == "massless"]
    assert len(msites) == 1
    return msites[0]


def test_tip4p_msite_placement_and_charge_transfer():
    water, o = _water()
    q_o = o.get("charge")
    n_bonds_before = len(list(water.bonds))
    out = mp.builder.Tip4pBuilder().apply(water)
    m = _msite(out)
    assert math.isclose(m.get("x"), 0.0, abs_tol=1e-9)
    assert m.get("y") > 0.0
    assert math.isclose(m.get("charge"), q_o, abs_tol=1e-12)
    out_o = next(a for a in out.atoms if a.get("atomic_number") == 8)
    assert math.isclose(out_o.get("charge"), 0.0, abs_tol=1e-12)
    assert len(list(out.bonds)) == n_bonds_before
    assert "element" not in m


def test_tip4p_records_average3_construction():
    """The M-site is tied to O, H, H by a ``virtual_sites`` ``average3`` row
    whose weights reproduce its position."""
    water, _ = _water()
    out = mp.builder.Tip4pBuilder().apply(water)
    atoms = _atoms_by_row(out)
    (row,) = _rows(out, "virtual_sites")
    assert row["style"] == "average3"
    site, o, h1, h2 = (atoms[row[k]] for k in ("atomi", "atomj", "atomk", "atoml"))
    assert site.get("vsite") == "massless"
    assert o.get("atomic_number") == 8
    assert {h1.get("atomic_number"), h2.get("atomic_number")} == {1}
    w = (row["w1"], row["w2"], row["w3"])
    assert sum(w) == pytest.approx(1.0)
    assert w[1] == pytest.approx(w[2])
    for axis in ("x", "y", "z"):
        rebuilt = sum(wi * a.get(axis) for wi, a in zip(w, (o, h1, h2)))
        assert rebuilt == pytest.approx(site.get(axis), abs=1e-12)
    om = math.dist([site[k] for k in "xyz"], [o[k] for k in "xyz"])
    assert om == pytest.approx(0.1546)


def test_tip4p_uses_minimum_image_across_the_box():
    """A water split by the periodic boundary still gets M on its bisector."""
    asm = mp.Atomistic()
    o = asm.add_atom(atomic_number=8, charge=-0.8, x=0.1, y=5.0, z=5.0)
    h1 = asm.add_atom(atomic_number=1, charge=0.4, x=9.343, y=5.586, z=5.0)
    h2 = asm.add_atom(atomic_number=1, charge=0.4, x=0.857, y=5.586, z=5.0)
    asm.add_bond(o, h1)
    asm.add_bond(o, h2)
    out = mp.builder.Tip4pBuilder(box=mp.Box(mp.core.Cell.lengths([10.0] * 3))).apply(
        asm
    )
    m = _msite(out)
    assert m.get("x") == pytest.approx(0.1, abs=1e-9)
    assert m.get("y") == pytest.approx(5.0 + 0.1546)
    (row,) = _rows(out, "virtual_sites")
    assert row["w2"] == pytest.approx(row["w3"])


def test_tip4p_finds_hydrogens_by_adjacency():
    """Only O atoms bonded to exactly two H atoms are hosts."""
    asm = mp.Atomistic()
    o = asm.add_atom(atomic_number=8, charge=-0.4, x=0.0, y=0.0, z=0.0)
    h = asm.add_atom(atomic_number=1, charge=0.4, x=0.96, y=0.0, z=0.0)
    asm.add_bond(o, h)
    out = mp.builder.Tip4pBuilder().apply(asm)
    assert not [a for a in out.atoms if a.get("vsite") == "massless"]
    assert _rows(out, "virtual_sites") == []


def test_drude_rejects_uncharged_host():
    asm = mp.Atomistic()
    asm.add_atom(atomic_number=6, type="CR", x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValueError, match="charge"):
        mp.builder.DrudeBuilder().apply(asm)
