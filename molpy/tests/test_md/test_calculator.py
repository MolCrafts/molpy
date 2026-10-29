"""The calculator seam: a force provider of its own, mounted on ``mp.md.MD``.

The calculator below is a harmonic bond written in numpy. It records what the
driver hands it, so the tests check the contract and not a model.
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp


class Bond:
    """``E = k (r - r0)**2`` between atoms 0 and 1, in ``real`` units."""

    units = "real"
    cutoff = 3.0

    def __init__(self, k: float = 100.0, r0: float = 1.0, fail_at: int | None = None):
        self.k = k
        self.r0 = r0
        self.fail_at = fail_at
        self.binds = 0
        self.calls: list[tuple[int, bool, bool]] = []
        self.seen_positions: list[np.ndarray] = []
        self.positions = np.empty((0, 3))
        self.forces = np.empty((0, 3))
        self.virial = np.zeros(6)

    def bind(self, frame: mp.core.Frame, skin: float):
        self.binds += 1
        n = frame["atoms"].n_rows
        self.positions = np.zeros((n, 3))
        self.forces = np.zeros((n, 3))
        return self.positions, self.forces, self.virial

    def compute(self, step: int, rebuild: bool, box: mp.core.Box | None) -> float:
        self.calls.append((step, rebuild, box is not None))
        self.seen_positions.append(self.positions.copy())
        if step == self.fail_at:
            raise FloatingPointError("the model diverged")
        d = self.positions[1] - self.positions[0]
        r = float(np.linalg.norm(d))
        g = 2.0 * self.k * (r - self.r0) / r
        self.forces[:] = 0.0
        self.forces[0] = g * d
        self.forces[1] = -g * d
        f1 = self.forces[1]
        self.virial[:] = [
            f1[0] * d[0],
            f1[1] * d[1],
            f1[2] * d[2],
            f1[0] * d[1],
            f1[0] * d[2],
            f1[1] * d[2],
        ]
        return self.k * (r - self.r0) ** 2


def _pair(x1: float = 0.8) -> mp.core.Frame:
    atoms = mp.core.Block()
    atoms.insert("x", np.array([0.2, x1]))
    atoms.insert("y", np.ones(2))
    atoms.insert("z", np.ones(2))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(2))
    atoms.insert("mass", np.full(2, 12.0))
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame.box = mp.core.Box(mp.core.Cell.lengths([10.0, 10.0, 10.0]))
    return frame


def _md(calculator, units: str = "real") -> mp.md.MD:
    sim = mp.md.MD(units)
    sim.set_integrator(mp.md.VelocityVerlet(0.5))
    sim.set_calculator(calculator, skin=0.5)
    return sim


def _coords(frame: mp.core.Frame) -> np.ndarray:
    return np.column_stack([np.asarray(frame["atoms"][k]) for k in "xyz"])


def test_the_bond_is_priced_by_hand():
    # 0.6 apart, compressed by 0.4: E = 100 * 0.16, f0 = 2k(r - r0) d/r = -80,
    # W_xx = f1 * d = 80 * 0.6.
    frame = _pair(0.8)
    _md(Bond()).run(frame, 0)
    assert frame.meta["energy"] == pytest.approx(16.0, abs=1e-9)
    assert np.asarray(frame["atoms"]["fx"]) == pytest.approx([-80.0, 80.0], abs=1e-9)
    assert frame.meta["virial"][0] == pytest.approx(48.0, abs=1e-9)


def test_the_positions_buffer_holds_the_frame_coordinates_at_each_call():
    frame = _pair(1.3)
    calc = Bond()
    sim = _md(calc)
    for _ in range(3):
        before = _coords(frame)
        sim.run(frame, 1)
        # The last call of this run evaluated at the coordinates the step
        # drifted to, which are the frame's now.
        np.testing.assert_array_equal(calc.seen_positions[-1], _coords(frame))
        assert not np.array_equal(before, _coords(frame))


def test_the_buffers_are_reused_not_reallocated():
    calc = Bond()
    sim = _md(calc)
    sim.run(_pair(1.3), 0)
    addresses = {
        name: getattr(calc, name).__array_interface__["data"][0]
        for name in ("positions", "forces", "virial")
    }
    frame = _pair(1.3)
    sim.run(frame, 5)
    for name, address in addresses.items():
        assert getattr(calc, name).__array_interface__["data"][0] == address, name


def test_bind_once_across_runs_and_again_after_a_setter():
    calc = Bond()
    sim = _md(calc)
    frame = _pair(1.2)
    sim.run(frame, 2)
    sim.run(frame, 2)
    assert calc.binds == 1
    sim.set_integrator(mp.md.VelocityVerlet(0.5))
    sim.run(frame, 1)
    assert calc.binds == 2


def test_an_exception_stops_the_run_at_its_step_unchanged():
    calc = Bond(fail_at=3)
    sim = _md(calc)
    with pytest.raises(FloatingPointError, match="the model diverged"):
        sim.run(_pair(1.1), 10)
    assert [call[0] for call in calc.calls] == [0, 1, 2, 3]
    assert sim.step == 3


def test_rebuild_and_box_arrive_when_they_change():
    calc = Bond()
    sim = _md(calc)
    frame = _pair(1.2)
    sim.run(frame, 2)
    frame.box = mp.core.Box(mp.core.Cell.lengths([10.01, 10.01, 10.01]))
    sim.run(frame, 1)
    # The first evaluation rebuilds and gets the box; the still pair never
    # moves half the skin; the changed box arrives once, at the next call.
    assert [c[1] for c in calc.calls] == [True, False, False, False]
    assert [c[2] for c in calc.calls] == [True, False, False, True]


def test_a_metal_calculator_is_converted_into_a_real_run():
    class MetalBond(Bond):
        units = "metal"

    frame = _pair(0.8)
    _md(MetalBond()).run(frame, 0)
    ev_in_kcal_per_mol = 23.060_548
    assert frame.meta["energy"] == pytest.approx(16.0 * ev_in_kcal_per_mol, rel=1e-6)


@pytest.mark.parametrize(
    ("buffers", "message"),
    [
        (lambda n: (np.zeros((n, 2)), np.zeros((n, 3))), "shape"),
        (lambda n: (np.zeros((n, 3), dtype=np.float32), np.zeros((n, 3))), "float64"),
        (lambda n: (np.zeros((3, n)).T, np.zeros((n, 3))), "C-contiguous"),
        (lambda n: (np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(5)), "shape"),
    ],
)
def test_a_wrong_buffer_is_refused_at_bind(buffers, message):
    class Wrong(Bond):
        def bind(self, frame, skin):
            return buffers(frame["atoms"].n_rows)

    with pytest.raises((TypeError, ValueError), match=message):
        _md(Wrong()).run(_pair(1.2), 0)


def test_a_read_only_buffer_is_refused_at_bind():
    class ReadOnly(Bond):
        def bind(self, frame, skin):
            positions = np.zeros((frame["atoms"].n_rows, 3))
            positions.flags.writeable = False
            return positions, np.zeros_like(positions)

    with pytest.raises(ValueError, match="writeable"):
        _md(ReadOnly()).run(_pair(1.2), 0)


def test_a_calculator_without_units_is_refused():
    class Undeclared:
        cutoff = None

        def bind(self, frame, skin): ...

        def compute(self, step, rebuild, box): ...

    with pytest.raises(TypeError, match="units"):
        mp.md.MD("real").set_calculator(Undeclared())


def test_set_evaluator_points_a_duck_typed_force_at_set_calculator():
    class Evaluate:
        def evaluate(self, pos):
            return 0.0, np.zeros_like(pos)

    with pytest.raises(TypeError, match="set_calculator"):
        mp.md.MD("real").set_evaluator(Evaluate())


def test_reading_the_stepped_frame_from_compute_is_refused_by_name():
    frame = _pair(1.2)

    class Peeking(Bond):
        def compute(self, step, rebuild, box):
            return float(np.asarray(frame["atoms"]["x"])[0])

    with pytest.raises(RuntimeError, match="in use"):
        _md(Peeking()).run(frame, 1)


def test_a_buffer_resized_in_place_is_refused():
    class Shrinking(Bond):
        def compute(self, step, rebuild, box):
            energy = super().compute(step, rebuild, box)
            self.forces.resize((1, 3), refcheck=False)
            return energy

    with pytest.raises(ValueError, match="never resize"):
        _md(Shrinking()).run(_pair(1.2), 1)


def test_buffers_that_share_memory_are_refused():
    class Shared(Bond):
        def bind(self, frame, skin):
            both = np.zeros((frame["atoms"].n_rows, 3))
            return both, both

    with pytest.raises(ValueError, match="share memory"):
        _md(Shared()).run(_pair(1.2), 0)


def test_a_non_finite_energy_stops_the_run():
    class Diverging(Bond):
        def compute(self, step, rebuild, box):
            super().compute(step, rebuild, box)
            return float("nan") if step == 2 else 0.0

    sim = _md(Diverging())
    with pytest.raises(ValueError, match="non-finite"):
        sim.run(_pair(1.2), 5)
    assert sim.step == 2


def test_a_python_evaluate_inside_potentials_is_refused_by_md():
    class Evaluate:
        def evaluate(self, pos):
            return 0.0, np.zeros_like(pos)

    holder = mp.ff.potential.Potentials()
    holder.push(Evaluate())
    with pytest.raises(TypeError, match="set_calculator"):
        mp.md.MD("real").set_evaluator(holder)


def test_a_barostat_rescales_the_cell_and_hands_it_to_the_calculator():
    calc = Bond()
    sim = _md(calc)
    sim.set_barostat(mp.md.BerendsenBarostat(1.0, 500.0, 2.2e4, 0.5))
    frame = _pair(0.8)
    before = frame.box.volume()
    sim.run(frame, 3)
    # A compressed bond pushes the cell out, and the new cell reaches the
    # calculator from the step after the first rescale.
    assert frame.box.volume() > before
    assert frame.meta["pressure"] > 1.0
    assert [c[2] for c in calc.calls] == [True, False, True, True]


def test_a_barostat_without_a_virial_is_refused():
    class NoVirial(Bond):
        def bind(self, frame, skin):
            positions, forces, _ = super().bind(frame, skin)
            return positions, forces

    sim = _md(NoVirial())
    sim.set_barostat(mp.md.BerendsenBarostat(1.0, 500.0, 2.2e4, 0.5))
    with pytest.raises(ValueError, match="virial"):
        sim.run(_pair(0.8), 1)
