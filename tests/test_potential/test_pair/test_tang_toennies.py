"""Tests for Tang-Toennies damped charge-induced-dipole Coulomb pair potential.

Validates the PairCoulTTStyle -> molrs PairTangToennies kernel path,
including closed-form damping, analytical forces vs finite-difference,
and physical limits.
"""

import math

import numpy as np
import pytest

import molrs
from molpy.potential.pair import PairCoulTTStyle


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_pair_frame(
    coords: list[list[float]],
    atom_types: list[str],
    pair_indices: list[tuple[int, int]],
    pair_types: list[str],
) -> molrs.Frame:
    """Build a minimal Frame with ``atoms`` and ``pairs`` blocks."""
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([c[0] for c in coords], dtype=np.float64))
    atoms.insert("y", np.array([c[1] for c in coords], dtype=np.float64))
    atoms.insert("z", np.array([c[2] for c in coords], dtype=np.float64))
    atoms.insert("type", np.array(atom_types, dtype=str))
    frame["atoms"] = atoms

    pairs = molrs.Block()
    pairs.insert("atomi", np.array([p[0] for p in pair_indices], dtype=np.uint32))
    pairs.insert("atomj", np.array([p[1] for p in pair_indices], dtype=np.uint32))
    pairs.insert("type", np.array(pair_types, dtype=str))
    frame["pairs"] = pairs
    return frame


def _numerical_forces(pots: molrs.Potentials, coords: np.ndarray) -> np.ndarray:
    """Central finite-difference gradient of calc_energy."""
    h = 1e-6
    num = np.zeros_like(coords)
    for k in range(len(coords)):
        cp = coords.copy()
        cm = coords.copy()
        cp[k] += h
        cm[k] -= h
        num[k] = -(pots.calc_energy(cp) - pots.calc_energy(cm)) / (2.0 * h)
    return num


def _tt_closed_form(r: float, b: float, n: int, c: float) -> float:
    """Compute f_n(r) = 1 - c·exp(-b·r)·Σ_{k=0}^n (b·r)^k/k!."""
    br = b * r
    series = 0.0
    term = 1.0
    for k in range(n + 1):
        series += term
        if k < n:
            term *= br / (k + 1)
    return 1.0 - c * math.exp(-br) * series


# ---------------------------------------------------------------------------
# ac-001 — import
# ---------------------------------------------------------------------------


def test_pair_coul_tt_style_importable():
    """ac-001: PairCoulTTStyle importable from molpy.potential.pair."""
    from molpy.potential.pair import PairCoulTTStyle as PCTTS

    assert PCTTS is PairCoulTTStyle


# ---------------------------------------------------------------------------
# ac-002 — calc_energy returns finite scalar
# ---------------------------------------------------------------------------


def test_calc_energy_finite():
    """ac-002: calc_energy returns a finite Python float."""
    ff = molrs.ForceField("tt-test")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=1.0)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert isinstance(energy, float)
    assert math.isfinite(energy)


# ---------------------------------------------------------------------------
# ac-003 — calc_forces shape (n_atoms, 3) + empty pairs
# ---------------------------------------------------------------------------


def test_calc_forces_shape():
    """ac-003: calc_forces returns (n_atoms, 3) with finite values."""
    ff = molrs.ForceField("tt-shape")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=1.0)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    forces = np.asarray(pots.calc_forces(molrs.extract_coords(frame)))
    assert forces.shape == (2, 3)
    assert np.all(np.isfinite(forces))


def test_empty_pairs_zero_energy_and_forces():
    """Empty pair list returns zero energy and zero forces."""
    ff = molrs.ForceField("tt-empty")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=1.0)  # must register ≥1 type

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[],
        pair_types=[],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    forces = np.asarray(pots.calc_forces(molrs.extract_coords(frame)))
    assert energy == pytest.approx(0.0)
    assert np.allclose(forces, 0.0)


# ---------------------------------------------------------------------------
# ac-005 — Tang-Toennies f_n matches closed form
# ---------------------------------------------------------------------------


def test_damping_closed_form_clandpol_canonical():
    """ac-005: TT f_n reproduces closed form with CL&Pol canonical params.

    n=4, b=4.5, c=1.0, r=1.0 — compute f_4(1.0) by hand.
    """
    b, n, c = 4.5, 4, 1.0
    r = 1.0
    q = 1.0

    f = _tt_closed_form(r, b, n, c)
    expected_energy = f * q * q / r

    ff = molrs.ForceField("tt-closed")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=q)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert energy == pytest.approx(expected_energy, abs=1e-10)


def test_damping_closed_form_simple_params():
    """TT result matches closed form with simple test params.

    n=2, b=1, c=1, r=2: Σ(1 + 2 + 2) = 5, f = 1 - e^{-2}·5.
    """
    r = 2.0
    q = 1.0
    f = _tt_closed_form(r, b=1.0, n=2, c=1.0)
    expected_energy = f * q * q / r

    # Use def_pairstyle to set non-default style params
    ff = molrs.ForceField("tt-simple")
    pstyle = ff.def_pairstyle("coul/tt", {"b": 1.0, "n": 2, "c": 1.0})
    pstyle.def_type("A", charge=q)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert energy == pytest.approx(expected_energy, abs=1e-12)


# ---------------------------------------------------------------------------
# ac-006 — analytic force == finite-difference
# ---------------------------------------------------------------------------


def test_forces_match_finite_difference():
    """ac-006: Analytic TT force matches central finite-difference gradient."""
    ff = molrs.ForceField("tt-fd")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=-0.7)
    pstyle.def_type("B", charge=0.5)

    frame = _build_pair_frame(
        coords=[[0.1, -0.2, 0.05], [1.3, 0.6, -0.3]],
        atom_types=["A", "B"],
        pair_indices=[(0, 1)],
        pair_types=["AB"],
    )
    pots = ff.to_potentials(frame)
    coords = molrs.extract_coords(frame)
    analytical = np.asarray(pots.calc_forces(coords)).ravel()
    numerical = _numerical_forces(pots, coords)

    assert np.allclose(analytical, numerical, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# ac-007 — damping → 1 at long range
# ---------------------------------------------------------------------------


def test_damping_approaches_one_at_long_range():
    """ac-007: f_n → 1 as r → ∞ (no damping at long range)."""
    r_far = 100.0
    q = 1.0
    expected_undamped = q * q / r_far  # = 0.01

    ff = molrs.ForceField("tt-long")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=q)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r_far, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert abs(energy - expected_undamped) < 1e-6


# ---------------------------------------------------------------------------
# ac-008 — strong damping at short range
# ---------------------------------------------------------------------------


def test_damping_strong_at_short_range():
    """ac-008: f_n << 1 as r → 0, energy stays finite."""
    r_short = 0.1
    q = 1.0
    expected_undamped = q * q / r_short

    ff = molrs.ForceField("tt-short")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=q)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r_short, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))

    assert energy < expected_undamped
    assert math.isfinite(energy)
    # At r=0.1, f_4(0.1) ≈ 1 - e^{-0.45}*(1 + 0.45 + 0.10125 + 0.0151875 + ...)
    # ≈ 1 - 0.6376*1.567 ≈ 1 - 0.999 ≈ 0.001, so highly damped
    assert energy < 0.5 * expected_undamped


# ---------------------------------------------------------------------------
# Newton's third law
# ---------------------------------------------------------------------------


def test_newtons_third_law():
    """Forces on i and j sum to zero (Newton's third law)."""
    ff = molrs.ForceField("tt-newton")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=-0.7)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [1.2, 0.7, -0.4]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    forces = np.asarray(pots.calc_forces(molrs.extract_coords(frame)))
    net = np.sum(forces, axis=0)
    assert np.allclose(net, [0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Default style params
# ---------------------------------------------------------------------------


def test_default_style_params():
    """CL&Pol canonical defaults: b=4.5, n=4, c=1.0."""
    ff = molrs.ForceField("tt-defaults")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=1.0)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))

    # Verify against explicit closed form with canonical params
    f = _tt_closed_form(r=1.0, b=4.5, n=4, c=1.0)
    expected = f * 1.0 * 1.0 / 1.0
    assert energy == pytest.approx(expected, abs=1e-10)
