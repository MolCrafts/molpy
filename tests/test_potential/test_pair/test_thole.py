"""Tests for Thole damped dipole-dipole / core-shell Coulomb pair potential.

Validates the PairTholeStyle -> molrs PairThole kernel path, including
closed-form damping, analytical forces vs finite-difference, and physical limits.
"""

import math

import numpy as np
import pytest

import molrs
from molpy.potential.pair import PairTholeStyle


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


# ---------------------------------------------------------------------------
# ac-001 — import
# ---------------------------------------------------------------------------


def test_pair_thole_style_importable():
    """ac-001: PairTholeStyle importable from molpy.potential.pair."""
    from molpy.potential.pair import PairTholeStyle as PTS

    assert PTS is PairTholeStyle


# ---------------------------------------------------------------------------
# ac-002 — calc_energy returns finite scalar
# ---------------------------------------------------------------------------


def test_calc_energy_finite():
    """ac-002: calc_energy returns a finite Python float."""
    ff = molrs.ForceField("thole-test")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=1.0, alpha=1.0, a_thole=2.6)

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
    ff = molrs.ForceField("thole-test")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=1.0, alpha=1.0, a_thole=2.6)

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
    ff = molrs.ForceField("thole-empty")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=1.0, alpha=1.0, a_thole=2.6)  # must register ≥1 type

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[],  # no pairs
        pair_types=[],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    forces = np.asarray(pots.calc_forces(molrs.extract_coords(frame)))
    assert energy == pytest.approx(0.0)
    assert np.allclose(forces, 0.0)


# ---------------------------------------------------------------------------
# ac-004 — Thole T_ij matches closed form
# ---------------------------------------------------------------------------


def test_damping_closed_form():
    """ac-004: Thole T_ij reproduces closed form at known r.

    a_thole = 2.6, alpha_i = alpha_j = 1.0 → a_ij = 2.6, s = 2.6.
    r = 2.0 → x = 5.2, T = 1 - (1 + 2.6)·exp(-5.2) = 1 - 3.6·exp(-5.2).
    """
    s_val = 2.6  # a_thole = 2.6, alpha_i = alpha_j = 1.0 → s = 2.6 / 1^(1/6) = 2.6
    r = 2.0
    q = 1.0

    # Closed form
    x = s_val * r
    t = 1.0 - (1.0 + x / 2.0) * math.exp(-x)
    expected_energy = t * q * q / r

    ff = molrs.ForceField("thole-closed")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=q, alpha=1.0, a_thole=2.6)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert energy == pytest.approx(expected_energy, abs=1e-12)


def test_damping_with_asymmetric_atom_types():
    """Thole with different atom-type params for each endpoint.

    a_i = 2.6, a_j = 2.2 → a_ij = 2.4.
    alpha_i = 1.0, alpha_j = 8.0 → (α_iα_j)^(1/6) = 8^(1/6) ≈ 1.4142.
    s = 2.4 / 1.4142 ≈ 1.697.
    """
    r = 1.5
    qi, qj = 0.5, -0.3

    a_ij = (2.6 + 2.2) / 2.0
    s_val = a_ij / (1.0 * 8.0) ** (1.0 / 6.0)
    x = s_val * r
    t = 1.0 - (1.0 + x / 2.0) * math.exp(-x)
    expected_energy = t * qi * qj / r

    ff = molrs.ForceField("thole-asym")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=qi, alpha=1.0, a_thole=2.6)
    pstyle.def_type("B", charge=qj, alpha=8.0, a_thole=2.2)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
        atom_types=["A", "B"],
        pair_indices=[(0, 1)],
        pair_types=["AB"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert energy == pytest.approx(expected_energy, abs=1e-12)


# ---------------------------------------------------------------------------
# ac-006 — analytic force == finite-difference
# ---------------------------------------------------------------------------


def test_forces_match_finite_difference():
    """ac-006: Analytic force matches central finite-difference gradient."""
    ff = molrs.ForceField("thole-fd")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=-0.7, alpha=1.3, a_thole=2.6)
    pstyle.def_type("B", charge=0.5, alpha=0.8, a_thole=2.4)

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
    """ac-007: T_ij → 1 as r → ∞ (no damping at long range)."""
    r_far = 100.0  # Å
    q = 1.0
    # At r=100, T ≈ 1, so energy ≈ q²/r = 0.01
    expected_undamped = q * q / r_far

    ff = molrs.ForceField("thole-long")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=q, alpha=1.0, a_thole=2.6)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r_far, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    # Damped energy should be very close to undamped Coulomb (|T - 1| < 1e-6)
    assert abs(energy - expected_undamped) < 1e-6


# ---------------------------------------------------------------------------
# ac-008 — strong damping at short range
# ---------------------------------------------------------------------------


def test_damping_strong_at_short_range():
    """ac-008: T_ij << 1 as r → 0, energy stays finite."""
    r_short = 0.1  # Å — well inside the damping region
    q = 1.0
    expected_undamped = q * q / r_short  # = 10.0

    ff = molrs.ForceField("thole-short")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=q, alpha=1.0, a_thole=2.6)

    frame = _build_pair_frame(
        coords=[[0.0, 0.0, 0.0], [r_short, 0.0, 0.0]],
        atom_types=["A", "A"],
        pair_indices=[(0, 1)],
        pair_types=["A"],
    )
    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))

    # Damped energy must be strictly less than undamped (damping << 1)
    assert energy < expected_undamped
    assert math.isfinite(energy)
    # At r=0.1, T should be well below 0.5 for s=2.6
    # x = 0.26, T = 1 - (1+0.13)·exp(-0.26) ≈ 0.0309
    assert energy < 0.5 * expected_undamped


# ---------------------------------------------------------------------------
# Newton's third law
# ---------------------------------------------------------------------------


def test_newtons_third_law():
    """Forces on i and j sum to zero (Newton's third law)."""
    ff = molrs.ForceField("thole-newton")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=-0.7, alpha=1.3, a_thole=2.6)

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
