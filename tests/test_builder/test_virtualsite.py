"""Tests for VirtualSiteBuilder / DrudeBuilder / Tip4pBuilder.

DrudeBuilder is the CL&Pol polarizer: it turns a CL&P-typed structure into a
Drude system. Tip4pBuilder proves the base class is not Drude-specific.
"""

import math

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.builder import DrudeBuilder, Tip4pBuilder, VirtualSiteBuilder
from molpy.builder.virtualsite import FOUR_PI_EPS0, K_DRUDE, load_polarizability
from molpy.data.forcefield import get_forcefield_path
from molpy.typifier import ClpTypifier


def _c4c1im_typed():
    """A CL&P atom+pair-typed [C4C1im]+ (types, charges, masses assigned)."""
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
    # No skip_* knobs: a term this force field cannot match is left undecided,
    # which is what "skip bonded typing" used to spell.
    return ClpTypifier(strict=False).typify(asm)


def _water(charge_o=-0.8, charge_h=0.4):
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


def test_drude_shell_is_typed_from_core():
    """Each Drude shell gets its own atom type ``D<core-type>`` (no untyped site)."""
    out = DrudeBuilder().apply(_c4c1im_typed())
    shells = _drudes(out)
    assert shells
    assert all(s.get("type") and s.get("type").startswith("D") for s in shells)
    # The shell type is the core type with the prefix.
    for bond in _drude_bonds(out):
        core, shell = bond.itom, bond.jtom
        if core.get("vsite") == "drude":
            core, shell = shell, core
        assert shell.get("type") == "D" + core.get("type")


def test_drude_shell_prefix_is_configurable():
    out = DrudeBuilder(drude_prefix="DP_").apply(_c4c1im_typed())
    assert all(s.get("type").startswith("DP_") for s in _drudes(out))


# --- ac-009: data file --------------------------------------------------------
def test_alpha_ff_resolves_and_loads():
    path = get_forcefield_path("alpha.ff")
    table = load_polarizability(path)
    assert table["CR"]["k_D"] == 4184.0 and table["CR"]["alpha"] > 0
    assert table["HC"]["k_D"] == 0.0  # hydrogen: no Drude


# --- builders are VirtualSiteBuilder subclasses -------------------------------
def test_builders_are_subclasses():
    assert issubclass(DrudeBuilder, VirtualSiteBuilder)
    assert issubclass(Tip4pBuilder, VirtualSiteBuilder)


# --- ac-003: apply does not mutate input -------------------------------------
def test_drude_apply_does_not_mutate_input():
    struct = _c4c1im_typed()
    n_before = len(list(struct.atoms))
    q_before = sum(a.get("charge") for a in struct.atoms)
    out = DrudeBuilder().apply(struct)
    assert out is not struct
    assert len(list(struct.atoms)) == n_before
    assert sum(a.get("charge") for a in struct.atoms) == q_before


# --- ac-004: Drude count == polarizable heavy atoms; H excluded ---------------
def test_drude_count_matches_heavy_atoms_no_hydrogen():
    struct = _c4c1im_typed()
    out = DrudeBuilder().apply(struct)
    heavy = [a for a in struct.atoms if a.get("element") != "H"]
    assert len(_drudes(out)) == len(heavy)
    # No hydrogen received a Drude: every Drude's core (bond partner) is heavy.
    for a in out.atoms:
        if a.get("element") == "H":
            assert a.get("vsite") is None


# --- ac-005: each Drude has one core-shell harmonic bond, K = 4184 ------------
def test_drude_spring_force_constant():
    out = DrudeBuilder().apply(_c4c1im_typed())
    springs = _drude_bonds(out)
    assert len(springs) == len(_drudes(out))
    assert all(b.get("k") == K_DRUDE == 4184.0 for b in springs)
    assert all(b.get("r0") == 0.0 for b in springs)


# --- ac-006: alpha recovered from assigned q_D, k_D ---------------------------
def test_alpha_recovered_from_drude_params():
    out = DrudeBuilder().apply(_c4c1im_typed())
    table = load_polarizability()
    for shell in _drudes(out):
        q_d, k_d, alpha = shell.get("charge"), shell.get("k_D"), shell.get("alpha")
        assert q_d**2 / (FOUR_PI_EPS0 * k_d) == alpha  # exact by construction
        assert alpha > 0
    # alpha values came from alpha.ff (sanity: CR alpha present in table)
    assert table["CR"]["alpha"] == 1.122


# --- ac-007: ion net charge stays integer +1 after augmentation ---------------
def test_cation_charge_conserved():
    out = DrudeBuilder().apply(_c4c1im_typed())
    total = sum(a.get("charge") for a in out.atoms)
    assert math.isclose(total, 1.0, abs_tol=1e-9)


# --- ac-008: Tip4pBuilder emits M-site on bisector, moves O charge, no spring -
def test_tip4p_msite_placement_and_charge_transfer():
    water, o = _water()
    q_o = o.get("charge")
    n_bonds_before = len(list(water.bonds))
    out = Tip4pBuilder().apply(water)
    msites = [a for a in out.atoms if a.get("vsite") == "massless"]
    assert len(msites) == 1
    m = msites[0]
    # M on the +y HOH bisector, below O by d_om
    assert math.isclose(m.get("x"), 0.0, abs_tol=1e-9)
    assert m.get("y") > 0.0
    assert math.isclose(m.get("charge"), q_o, abs_tol=1e-12)
    # O left neutral; no new bond added (rigid geometric, not a spring)
    out_o = next(a for a in out.atoms if a.get("element") == "O")
    assert math.isclose(out_o.get("charge"), 0.0, abs_tol=1e-12)
    assert len(list(out.bonds)) == n_bonds_before
