"""VirtualSiteBuilder / DrudeBuilder / Tip4pBuilder.

Drude tests read a hand-typed CL&P [C4C1im]+ (``mol2/c4c1im_clp_typed.mol2``);
Tip4p uses plain water.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import molpy as mp
from molpy.builder import DrudeBuilder, Tip4pBuilder, VirtualSiteBuilder
from molpy.ff.params import clpol_polarizability

#: 4πε₀ in e² / (kJ/mol·Å): the paduagroup/clandpol polarizer's value.
FOUR_PI_EPS0 = 0.0007197587


@pytest.fixture
def cation(TEST_DATA_DIR: Path) -> mp.Atomistic:
    """[C4C1im]+ with CL&P types and charges; the fixture names atoms by element."""
    frame = mp.io.read_mol2(TEST_DATA_DIR / "mol2" / "c4c1im_clp_typed.mol2")
    frame["atoms"]["element"] = frame["atoms"]["name"]
    return mp.Atomistic.from_frame(frame)


def _water(charge_o: float = -0.8, charge_h: float = 0.4):
    asm = mp.Atomistic()
    o = asm.def_atom(element="O", charge=charge_o, x=0.0, y=0.0, z=0.0)
    h1 = asm.def_atom(element="H", charge=charge_h, x=0.757, y=0.586, z=0.0)
    h2 = asm.def_atom(element="H", charge=charge_h, x=-0.757, y=0.586, z=0.0)
    asm.def_bond(o, h1)
    asm.def_bond(o, h2)
    return asm, o


def _drudes(struct):
    return [a for a in struct.atoms if a.get("vsite") == "drude"]


def _drude_bonds(struct):
    return [b for b in struct.bonds if b.get("style") == "drude"]


class TestVirtualSiteBuilder:
    def test_builder_is_an_abstract_transform(self):
        with pytest.raises(TypeError):
            VirtualSiteBuilder()


class TestDrudeBuilder:
    def test_is_a_virtual_site_transform(self):
        assert issubclass(DrudeBuilder, VirtualSiteBuilder)


class TestTip4pBuilder:
    def test_is_a_virtual_site_transform(self):
        assert issubclass(Tip4pBuilder, VirtualSiteBuilder)


def test_builders_are_subclasses():
    assert issubclass(DrudeBuilder, VirtualSiteBuilder)
    assert issubclass(Tip4pBuilder, VirtualSiteBuilder)


def test_the_clpol_table_is_molrs_s():
    table = clpol_polarizability()
    assert table["CR"]["k_D"] == 4184.0 and table["CR"]["alpha"] > 0
    assert table["HC"]["k_D"] == 0.0


def test_drude_shell_is_typed_from_core(cation):
    out = DrudeBuilder().apply(cation)
    shells = _drudes(out)
    assert shells
    assert all(s.get("type") and s.get("type").startswith("D") for s in shells)
    for bond in _drude_bonds(out):
        core, shell = bond.itom, bond.jtom
        if core.get("vsite") == "drude":
            core, shell = shell, core
        assert shell.get("type") == "D" + core.get("type")


def test_drude_shell_prefix_is_configurable(cation):
    out = DrudeBuilder(drude_prefix="DP_").apply(cation)
    assert all(s.get("type").startswith("DP_") for s in _drudes(out))


def test_drude_apply_does_not_mutate_input(cation):
    struct = cation
    n_before = len(list(struct.atoms))
    q_before = sum(a.get("charge") for a in struct.atoms)
    out = DrudeBuilder().apply(struct)
    assert out is not struct
    assert len(list(struct.atoms)) == n_before
    assert sum(a.get("charge") for a in struct.atoms) == q_before


def test_drude_count_matches_heavy_atoms_no_hydrogen(cation):
    struct = cation
    out = DrudeBuilder().apply(struct)
    heavy = [a for a in struct.atoms if a.get("element") != "H"]
    assert len(_drudes(out)) == len(heavy)
    for a in out.atoms:
        if a.get("element") == "H":
            assert a.get("vsite") is None


def test_drude_spring_force_constant(cation):
    """alpha.ff is kJ/mol; molrs stores kcal/mol (÷4.184)."""
    out = DrudeBuilder().apply(cation)
    springs = _drude_bonds(out)
    assert len(springs) == len(_drudes(out))
    assert all(b.get("k") == pytest.approx(4184.0 / 4.184) for b in springs)
    assert all(b.get("r0") == 0.0 for b in springs)


def test_alpha_recovered_from_drude_params(cation):
    out = DrudeBuilder().apply(cation)
    table = clpol_polarizability()
    for shell in _drudes(out):
        q_d, k_d, alpha = shell.get("charge"), shell.get("k_D"), shell.get("alpha")
        assert q_d**2 / (FOUR_PI_EPS0 * k_d) == pytest.approx(alpha, rel=1e-7)
        assert alpha > 0
    assert table["CR"]["alpha"] == 1.122


def test_cation_charge_conserved(cation):
    out = DrudeBuilder().apply(cation)
    total = sum(a.get("charge") for a in out.atoms)
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_tip4p_msite_placement_and_charge_transfer():
    water, o = _water()
    q_o = o.get("charge")
    n_bonds_before = len(list(water.bonds))
    out = Tip4pBuilder().apply(water)
    msites = [a for a in out.atoms if a.get("vsite") == "massless"]
    assert len(msites) == 1
    m = msites[0]
    assert math.isclose(m.get("x"), 0.0, abs_tol=1e-9)
    assert m.get("y") > 0.0
    assert math.isclose(m.get("charge"), q_o, abs_tol=1e-12)
    out_o = next(a for a in out.atoms if a.get("element") == "O")
    assert math.isclose(out_o.get("charge"), 0.0, abs_tol=1e-12)
    assert len(list(out.bonds)) == n_bonds_before
