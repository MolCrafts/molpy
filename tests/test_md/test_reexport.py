"""molpy.md is THE user-facing MD namespace — a verbatim molrs.md re-export."""

import numpy as np
import pytest

import molpy.md
import molrs.md


def test_every_molrs_md_public_name_is_identical_in_molpy_md():
    """Completeness: molpy.md re-exports the full molrs.md surface verbatim."""
    for name in molrs.md.__all__:
        assert hasattr(molpy.md, name), f"molpy.md misses {name}"
        assert getattr(molpy.md, name) is getattr(molrs.md, name), (
            f"molpy.md.{name} is not the molrs.md object"
        )
        assert name in molpy.md.__all__


def test_potential_subclass_runs_through_the_integrator():
    """``class MyPot(molpy.md.Potential)`` drives VelocityVerlet end to end."""

    class Tether(molpy.md.Potential):
        def __init__(self, k):
            self.k = float(k)
            self.calls = 0

        def calc_energy_forces(self, pos):
            self.calls += 1
            return 0.5 * self.k * float((pos * pos).sum()), -self.k * pos

    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    tether = Tether(1.0e-3)
    vv = molpy.md.VelocityVerlet(0.5, potential=tether, mass=np.ones(2))
    state = vv.advance_n(vv.initial(pos, np.zeros_like(pos)), 3)
    assert tether.calls > 0
    assert np.isfinite(state.energy)
    # The tether pulls the off-origin atom inward: the override's forces acted.
    assert abs(float(state.pos[1, 0])) < 1.5


def test_unoverridden_potential_base_raises():
    with pytest.raises((TypeError, NotImplementedError)):
        molpy.md.Potential()
