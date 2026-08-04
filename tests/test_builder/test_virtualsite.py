"""VirtualSiteBuilder / DrudeBuilder / Tip4pBuilder — no per-test CL&P rebuild.

Drude tests need a CL&P-typed cation; that graph is typified **once** at module
import via the shared production-cached ``ClpTypifier``. Tip4p uses plain water.
"""

from __future__ import annotations

import math

import pytest

from molpy import Atomistic
from molpy.builder import DrudeBuilder, Tip4pBuilder, VirtualSiteBuilder
from molpy.builder.virtualsite import FOUR_PI_EPS0, K_DRUDE, load_polarizability
from molpy.data.forcefield import get_forcefield_path
from molpy.typifier import ClpTypifier

# ---------------------------------------------------------------------------
# one typed cation for the whole module
# ---------------------------------------------------------------------------


def _c4c1im_graph() -> Atomistic:
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
    asm = Atomistic()
    atoms = [asm.def_atom(element=e) for e in el]
    for i, j in edges:
        asm.def_bond(atoms[i], atoms[j])
    return asm


# Typed once. ``ClpTypifier`` construction is production-cached; ``.typify`` is cheap.
_TYPED_CATION = ClpTypifier(strict=False).typify(_c4c1im_graph())


def _typed_cation() -> Atomistic:
    """Return a copy so a test can never poison the shared graph."""
    return _TYPED_CATION.copy()


def _water(charge_o: float = -0.8, charge_h: float = 0.4):
    asm = Atomistic()
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


def test_alpha_ff_resolves_and_loads():
    path = get_forcefield_path("alpha.ff")
    table = load_polarizability(path)
    assert table["CR"]["k_D"] == 4184.0 and table["CR"]["alpha"] > 0
    assert table["HC"]["k_D"] == 0.0


def test_drude_shell_is_typed_from_core():
    out = DrudeBuilder().apply(_typed_cation())
    shells = _drudes(out)
    assert shells
    assert all(s.get("type") and s.get("type").startswith("D") for s in shells)
    for bond in _drude_bonds(out):
        core, shell = bond.itom, bond.jtom
        if core.get("vsite") == "drude":
            core, shell = shell, core
        assert shell.get("type") == "D" + core.get("type")


def test_drude_shell_prefix_is_configurable():
    out = DrudeBuilder(drude_prefix="DP_").apply(_typed_cation())
    assert all(s.get("type").startswith("DP_") for s in _drudes(out))


def test_drude_apply_does_not_mutate_input():
    struct = _typed_cation()
    n_before = len(list(struct.atoms))
    q_before = sum(a.get("charge") for a in struct.atoms)
    out = DrudeBuilder().apply(struct)
    assert out is not struct
    assert len(list(struct.atoms)) == n_before
    assert sum(a.get("charge") for a in struct.atoms) == q_before


def test_drude_count_matches_heavy_atoms_no_hydrogen():
    struct = _typed_cation()
    out = DrudeBuilder().apply(struct)
    heavy = [a for a in struct.atoms if a.get("element") != "H"]
    assert len(_drudes(out)) == len(heavy)
    for a in out.atoms:
        if a.get("element") == "H":
            assert a.get("vsite") is None


def test_drude_spring_force_constant():
    """alpha.ff is kJ/mol; molrs stores kcal/mol (÷4.184)."""
    out = DrudeBuilder().apply(_typed_cation())
    springs = _drude_bonds(out)
    assert len(springs) == len(_drudes(out))
    assert K_DRUDE == 4184.0
    assert all(b.get("k") == pytest.approx(K_DRUDE / 4.184) for b in springs)
    assert all(b.get("r0") == 0.0 for b in springs)


def test_alpha_recovered_from_drude_params():
    out = DrudeBuilder().apply(_typed_cation())
    table = load_polarizability()
    for shell in _drudes(out):
        q_d, k_d, alpha = shell.get("charge"), shell.get("k_D"), shell.get("alpha")
        assert q_d**2 / (FOUR_PI_EPS0 * k_d) == alpha
        assert alpha > 0
    assert table["CR"]["alpha"] == 1.122


def test_cation_charge_conserved():
    out = DrudeBuilder().apply(_typed_cation())
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
