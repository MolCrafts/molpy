"""``mp.optimize``: the L-BFGS minimizer and its one report type.

The minimizer's numerics are proven by the Rust unit tests in
``crates/molrs/src/optimize/``; these check the Python face: the Rust names, the
defaults read from ``LbfgsSettings::DEFAULT``, the report fields, the three
doors of the core ``Lbfgs`` object (each relaxing its input in place), and
that ``minimize(frame)`` honours the frame's ``atoms.free`` mask.
"""

from __future__ import annotations

import math

import molpy as mp
import numpy as np
import pytest
from molpy.ff import ExplicitTerms


def _lj_dimer():
    # One LJ pair at 1.5 sigma: the minimum is at 2^(1/6) sigma.
    return ExplicitTerms(
        "pair", "lj/cut", [[0, 1]], epsilon=1.0, sigma=1.0, cutoff=math.inf
    ).compile()


def _dimer_frame(free: list[bool] | None = None) -> mp.core.Frame:
    atoms = mp.core.Block()
    atoms.insert("x", np.array([0.0, 1.5]))
    atoms.insert("y", np.zeros(2))
    atoms.insert("z", np.zeros(2))
    if free is not None:
        atoms.insert("free", np.array(free))
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    return frame


def _xyz(frame: mp.core.Frame) -> np.ndarray:
    atoms = frame["atoms"]
    return np.stack([np.asarray(atoms[k]) for k in "xyz"], axis=1)


def test_minimize_coords_relaxes_a_dimer_to_the_lj_minimum():
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    report = mp.optimize.Lbfgs(_lj_dimer(), fmax=1e-6).minimize_coords(coords)
    assert isinstance(report, mp.optimize.OptimizationReport)
    assert report.converged
    assert report.final_fmax < 1e-6
    assert report.final_grad_rms <= report.final_fmax
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(
        2 ** (1 / 6), abs=1e-5
    )


def test_minimize_coords_refuses_an_array_it_cannot_write_in_place():
    opt = mp.optimize.Lbfgs(_lj_dimer())
    strided = np.zeros((2, 6))[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        opt.minimize_coords(strided)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        opt.minimize_coords(np.zeros((2, 2)))


def test_minimize_batch_relaxes_each_structure_in_place():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.3, 0.0]]]
    )
    reports = mp.optimize.Lbfgs(_lj_dimer(), fmax=1e-6).minimize_batch(coords)
    assert len(reports) == 2 and all(r.converged for r in reports)
    for pair in coords:
        assert np.linalg.norm(pair[1] - pair[0]) == pytest.approx(
            2 ** (1 / 6), abs=1e-5
        )


def test_minimize_relaxes_a_frame_in_place():
    frame = _dimer_frame()
    report = mp.optimize.Lbfgs(_lj_dimer(), fmax=1e-6).minimize(frame)
    assert report.converged
    xyz = _xyz(frame)
    assert np.linalg.norm(xyz[1] - xyz[0]) == pytest.approx(2 ** (1 / 6), abs=1e-5)


def test_minimize_honours_the_frame_free_mask():
    """A fixed atom (``atoms.free`` False) stays put; only the free one
    moves — the core ``Lbfgs::minimize`` contract. Python ``minimize`` once
    ignored the mask and moved both."""
    frame = _dimer_frame(free=[False, True])
    report = mp.optimize.Lbfgs(_lj_dimer(), fmax=1e-6).minimize(frame)
    assert report.converged
    xyz = _xyz(frame)
    np.testing.assert_array_equal(xyz[0], [0.0, 0.0, 0.0])
    assert xyz[1, 0] == pytest.approx(2 ** (1 / 6), abs=1e-5)
    np.testing.assert_allclose(xyz[1, 1:], 0.0, atol=1e-12)
    # The old, mask-blind path relaxed the same dimer with both atoms free —
    # the fixed atom moved.
    blind = _xyz(_dimer_frame())
    mp.optimize.Lbfgs(_lj_dimer(), fmax=1e-6).minimize_coords(blind)
    assert blind[0, 0] != 0.0


def test_minimize_with_every_atom_fixed_moves_nothing():
    frame = _dimer_frame(free=[False, False])
    mp.optimize.Lbfgs(_lj_dimer()).minimize(frame)
    np.testing.assert_array_equal(_xyz(frame), [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])


def test_defaults_are_the_rust_defaults():
    assert repr(mp.optimize.Lbfgs(_lj_dimer())) == (
        "Lbfgs(fmax=0.05, max_steps=500, max_step=0.2, memory=8)"
    )


class _Tether:
    """``E = ½ k |x − x₀|²``, each atom tethered where it was bound."""

    units = "real"
    cutoff = None

    def __init__(self, k: float) -> None:
        self.k = k
        self.binds = 0

    def bind(self, frame, skin):
        self.binds += 1
        self.anchor = np.column_stack([np.asarray(frame["atoms"][c]) for c in "xyz"])
        self.pos = np.zeros_like(self.anchor)
        self.force = np.zeros_like(self.anchor)
        return self.pos, self.force

    def compute(self, step, rebuild, box):
        d = self.pos - self.anchor
        self.force[:] = -self.k * d
        return 0.5 * self.k * float((d * d).sum())


def test_minimize_binds_a_python_potential_to_the_frame():
    # Tethered to the starting geometry, the LJ dimer relaxes to between its
    # own minimum and the start; the tether saw the frame at bind.
    frame = _dimer_frame()
    start = _xyz(frame).copy()
    tether = _Tether(k=0.5)
    pots = _lj_dimer()
    pots.push(tether)
    report = mp.optimize.Lbfgs(pots, fmax=1e-6, max_steps=500).minimize(frame)
    assert report.converged
    assert tether.binds == 1
    np.testing.assert_allclose(tether.anchor, start)
    assert not np.allclose(_xyz(frame), start)


def _mmff_ethoxybenzene() -> tuple[object, object]:
    template = mp.io.smiles.read_smiles("CCOc1ccc(Br)cc1").to_atomistic()
    molecule, _ = mp.conformer.Conformer(seed=1).generate(template)
    typifier = mp.ff.typifier.Mmff94sTypifier()
    frame = typifier.typify(molecule).to_frame()
    return mp.ff.PotentialCompiler(typifier.forcefield()).compile(frame), frame


def test_lammps_is_the_documented_minimize_example() -> None:
    potentials, frame = _mmff_ethoxybenzene()
    report = mp.optimize.Lbfgs.lammps(potentials).minimize(frame)
    assert report.stop_reason in {
        "energy_change",
        "force_norm",
        "max_iterations",
        "max_evaluations",
    }
    assert report.n_steps <= 100 and report.n_evals <= 1000 + 40


def test_a_budget_stop_is_not_convergence() -> None:
    potentials, frame = _mmff_ethoxybenzene()
    report = mp.optimize.Lbfgs(potentials, fmax=0.0, max_steps=1).minimize(frame)
    assert (report.stop_reason, report.converged) == ("max_iterations", False)
