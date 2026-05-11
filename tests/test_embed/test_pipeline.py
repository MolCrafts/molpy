"""Integration tests for :mod:`molpy.embed`.

The molpy embed module is a Python wrapper around the molrs Rust ``embed``
pipeline (built and shipped as the ``molrs`` binary extension). These tests
exercise the wrapper end-to-end on simple skeletons.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

molrs = pytest.importorskip("molrs")

from molpy import Atomistic
from molpy.embed import EmbedReport, generate_3d


def _bond(mol: Atomistic, a, b, order: float):
    return mol.def_bond(a, b, order=order)


def _coords_of(mol: Atomistic) -> np.ndarray:
    return np.array(
        [
            [
                float(a.get("x", math.nan)),
                float(a.get("y", math.nan)),
                float(a.get("z", math.nan)),
            ]
            for a in mol.atoms
        ]
    )


def _all_have_coords(mol: Atomistic) -> bool:
    return bool(np.all(np.isfinite(_coords_of(mol))))


def _min_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return math.inf
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


def _ethanol_skeleton() -> Atomistic:
    mol = Atomistic()
    c1 = mol.def_atom(element="C")
    c2 = mol.def_atom(element="C")
    o = mol.def_atom(element="O")
    _bond(mol, c1, c2, 1.0)
    _bond(mol, c2, o, 1.0)
    return mol


def _butane_skeleton() -> Atomistic:
    mol = Atomistic()
    a = mol.def_atom(element="C")
    b = mol.def_atom(element="C")
    c = mol.def_atom(element="C")
    d = mol.def_atom(element="C")
    _bond(mol, a, b, 1.0)
    _bond(mol, b, c, 1.0)
    _bond(mol, c, d, 1.0)
    return mol


def test_generate_3d_ethanol_assigns_coordinates():
    mol = _ethanol_skeleton()
    out, report = generate_3d(mol, rng_seed=42, add_hydrogens=True)

    assert len(list(out.atoms)) > len(list(mol.atoms))
    assert _all_have_coords(out)
    assert isinstance(report, EmbedReport)
    assert report.final_energy is not None and math.isfinite(report.final_energy)
    stage_names = {s.stage for s in report.stages}
    assert "build_initial" in stage_names
    assert "final_optimize" in stage_names
    assert _min_distance(_coords_of(out)) > 0.35


def test_generate_3d_seed_reproducible():
    mol = _butane_skeleton()
    g1, _ = generate_3d(mol, add_hydrogens=False, rng_seed=7)
    g2, _ = generate_3d(mol, add_hydrogens=False, rng_seed=7)

    c1 = _coords_of(g1)
    c2 = _coords_of(g2)
    assert c1.shape == c2.shape
    np.testing.assert_allclose(c1, c2, atol=1e-12)


def test_generate_3d_speed_presets_validate():
    mol = _ethanol_skeleton()
    for speed in ("fast", "medium", "better"):
        out, report = generate_3d(mol, speed=speed, rng_seed=3, add_hydrogens=False)
        assert _all_have_coords(out)
        assert report.stages, f"no stages reported for speed={speed}"


def test_generate_3d_empty_molecule_returns_error():
    mol = Atomistic()
    with pytest.raises(ValueError, match="empty"):
        generate_3d(mol)


def test_generate_3d_preserves_template_attributes():
    mol = Atomistic()
    a = mol.def_atom(element="C", custom_label="alpha")
    b = mol.def_atom(element="O", custom_label="beta")
    _bond(mol, a, b, 1.0)
    out, _ = generate_3d(mol, add_hydrogens=False, rng_seed=5)
    out_atoms = list(out.atoms)
    assert out_atoms[0].get("custom_label") == "alpha"
    assert out_atoms[1].get("custom_label") == "beta"
