"""Tests for CL&Pol scaleLJ epsilon scaling."""

import pytest

from molpy import Atom, AtomisticForcefield
from molpy.core.forcefield import PairType
from molpy.core.ops import (
    FragmentScaling,
    compute_k_ij,
    load_fragment_scaling_data,
    scale_lj,
)


def _ff():
    """A tiny FF with one cross (CR-B) and one intra (CR-CR) pair type."""
    ff = AtomisticForcefield()
    astyle = ff.def_atomstyle("full")
    cr = astyle.def_type("CR", type_="CR", class_="CR")
    b = astyle.def_type("B", type_="B", class_="B")
    ps = ff.def_pairstyle("lj/cut")
    ps.def_type(cr, b, epsilon=1.0, sigma=3.5)  # inter-fragment -> scaled
    ps.def_type(cr, cr, epsilon=2.0, sigma=3.6)  # intra -> NOT scaled
    return ff


def _pt(ff, a, b):
    for pt in ff.get_types(PairType):
        if {pt.itom.name, pt.jtom.name} == {a, b}:
            return pt
    raise KeyError((a, b))


def _fragments(r=4.0):
    """Cation (CR) at origin, anion (B) at distance r along x."""
    return {
        "c2c1im": [Atom(type="CR", mass=12.0, x=0.0, y=0.0, z=0.0)],
        "bf4": [Atom(type="B", mass=11.0, x=r, y=0.0, z=0.0)],
    }


DATA = {
    "c2c1im": FragmentScaling("c2c1im", 1.0, 1.1558, 12.383),
    "bf4": FragmentScaling("bf4", -1.0, 0.0, 3.078),
}


# --- ac-001 -------------------------------------------------------------------
def test_importable():
    assert all(
        callable(x) or x
        for x in (scale_lj, compute_k_ij, load_fragment_scaling_data, FragmentScaling)
    )


# --- ac-005: closed form ------------------------------------------------------
def test_compute_k_ij_closed_form():
    # fr_i non-polarizable, fr_j polarizable -> single induction term from fr_j
    fi = FragmentScaling("a", q=-1.0, mu=0.0, alpha=3.0, polarizable=False)
    fj = FragmentScaling("b", q=1.0, mu=1.5, alpha=10.0, polarizable=True)
    r = 4.0
    expected = 1.0 / (
        1.0 + 0.254952 * r**2 * fj.q**2 / fj.alpha + 0.106906 * fj.mu**2 / fj.alpha
    )
    assert compute_k_ij(fi, fj, r) == pytest.approx(expected, rel=1e-6)


def test_k_ij_in_unit_interval():
    assert 0.0 < compute_k_ij(DATA["c2c1im"], DATA["bf4"], 3.088) <= 1.0


def test_alpha_zero_raises():
    bad = FragmentScaling("x", 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        compute_k_ij(bad, DATA["bf4"], 4.0)


# --- ac-006: mu^2 term carries no r^2 prefactor -------------------------------
def test_mu_term_has_no_r2_prefactor():
    fi = FragmentScaling("a", q=-1.0, mu=0.0, alpha=3.0, polarizable=False)
    fj = FragmentScaling("b", q=1.0, mu=2.0, alpha=8.0, polarizable=True)
    r1, r2 = 3.0, 6.0
    inv1 = 1.0 / compute_k_ij(fi, fj, r1)
    inv2 = 1.0 / compute_k_ij(fi, fj, r2)
    # difference in 1/k is purely the q^2 (r^2) term; mu^2 term cancels exactly
    only_q = 0.254952 * (r1**2 - r2**2) * fj.q**2 / fj.alpha
    assert (inv1 - inv2) == pytest.approx(only_q, rel=1e-9)


# --- ac-002/003/004: scale_lj operator ----------------------------------------
def test_scale_lj_scales_cross_epsilon():
    ff = _ff()
    r = 4.0
    out = scale_lj(ff, _fragments(r), DATA)
    k = compute_k_ij(DATA["c2c1im"], DATA["bf4"], r)
    assert _pt(out, "CR", "B").get("epsilon") == pytest.approx(1.0 * k)
    assert 0.0 < k <= 1.0


def test_scale_lj_leaves_intra_and_sigma_and_input():
    ff = _ff()
    out = scale_lj(ff, _fragments(4.0), DATA)
    # intra-fragment CR-CR untouched
    assert _pt(out, "CR", "CR").get("epsilon") == 2.0
    # sigma untouched on the scaled cross pair
    assert _pt(out, "CR", "B").get("sigma") == 3.5
    # input FF not mutated
    assert _pt(ff, "CR", "B").get("epsilon") == 1.0
    assert out is not ff


def test_scale_sigma_flag():
    out = scale_lj(_ff(), _fragments(4.0), DATA, scale_sigma=True)
    assert _pt(out, "CR", "B").get("sigma") == pytest.approx(3.5 * 0.985)


def test_missing_fragment_data_raises():
    with pytest.raises(KeyError):
        scale_lj(_ff(), _fragments(4.0), {"c2c1im": DATA["c2c1im"]})


# --- ac-008: data file --------------------------------------------------------
def test_fragment_data_file_loads():
    table = load_fragment_scaling_data()
    for name in ("c2c1im", "bf4", "pf6", "ntf2", "dca"):
        assert name in table
    bf4 = table["bf4"]
    assert bf4.q == -1.0 and bf4.alpha == pytest.approx(3.078)
    assert bf4.polarizable is False
