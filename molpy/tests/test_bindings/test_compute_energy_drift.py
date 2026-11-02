"""``mp.compute.EnergyDrift``: the drift per atom of a conserved energy."""

import pytest

import molpy as mp


def test_a_line_gives_its_slope_per_atom():
    # E = -100 + 0.002 t over four samples of four atoms: 5e-4 per atom.
    acc = mp.compute.EnergyDrift(4).accumulator()
    for t in range(4):
        acc.push(time=float(t), energy=-100.0 + 0.002 * t)
    out = acc.finish()
    assert out.drift == pytest.approx(5e-4, rel=1e-10)
    assert out.n_samples == 4
    assert out.duration == 3.0


def test_a_symmetric_wobble_has_no_drift():
    acc = mp.compute.EnergyDrift(1).accumulator()
    for t, e in enumerate([0.25, -0.25, -0.25, 0.25]):
        acc.push(time=float(t), energy=e)
    out = acc.finish()
    assert out.drift == pytest.approx(0.0, abs=1e-15)
    assert out.rms == pytest.approx(0.25, abs=1e-15)


def test_time_that_does_not_advance_is_refused():
    acc = mp.compute.EnergyDrift(1).accumulator()
    acc.push(time=1.0, energy=0.0)
    with pytest.raises(ValueError, match="time"):
        acc.push(time=1.0, energy=0.0)
