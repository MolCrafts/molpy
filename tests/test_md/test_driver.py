"""``mp.md`` is ``molrs.md``: the phase driver runs a frame through molpy names.

The driver's behaviour is molrs's and tested there; this pins the mirror and
one step of the seam molpy users call.
"""

import molrs
import numpy as np
import pytest

import molpy as mp
from molpy.ff.potential import PairLjCut
from molpy.md import MD, NeighborList, VelocityVerlet


def _pair() -> mp.Frame:
    atoms = mp.Block()
    pos = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d]))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(2))
    atoms.insert("mass", np.ones(2))
    frame = mp.Frame()
    frame["atoms"] = atoms
    frame.box = mp.Box(mp.core.Cell.lengths([10.0] * 3))
    return frame


def test_the_md_names_are_molrs_objects():
    for name in molrs.md.__all__:
        assert getattr(mp.md, name) is getattr(molrs.md, name)


def test_one_step_writes_energy_and_forces_onto_the_frame():
    frame = _pair()
    sim = MD()
    sim.set_integrator(VelocityVerlet(0.001))
    sim.set_neighbor(NeighborList(2.5, 0.5))
    sim.set_evaluator(PairLjCut(1.0, 1.0, 2.5))
    sim.run(frame, 1)
    assert np.isfinite(frame.meta["energy"]) and frame.meta["energy"] != 0.0
    assert np.isfinite(np.asarray(frame["atoms"]["fx"])).all()


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_a_non_positive_timestep_is_refused(dt):
    with pytest.raises(ValueError, match="dt"):
        VelocityVerlet(dt)
