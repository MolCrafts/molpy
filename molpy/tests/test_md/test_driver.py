"""``mp.md`` is ``molpy.md``: the phase driver runs a frame through molpy names.

The driver's behaviour is molrs's and tested there; this pins one step of the
seam molpy users call (the mirror's identity is ``test_public_paths.py``'s).
"""

import numpy as np
import pytest

import molpy as mp


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


def test_one_step_writes_energy_and_forces_onto_the_frame():
    frame = _pair()
    sim = mp.md.MD("real")
    sim.set_integrator(mp.md.VelocityVerlet(0.001))
    sim.set_neighbor(mp.core.NeighborList(3.0), skin=0.5)
    sim.set_evaluator(mp.ff.potential.PairLjCut(1.0, 1.0, 2.5))
    sim.run(frame, 1)
    assert np.isfinite(frame.meta["energy"]) and frame.meta["energy"] != 0.0
    assert np.isfinite(np.asarray(frame["atoms"]["fx"])).all()


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_a_non_positive_timestep_is_refused(dt):
    with pytest.raises(ValueError, match="dt"):
        mp.md.VelocityVerlet(dt)


def _argon() -> mp.Frame:
    # Eight argon atoms on a cubic lattice of spacing 4 Å in a 12 Å box.
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    pos = 4.0 * grid + 1.0
    atoms = mp.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d], dtype=float))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(len(pos)))
    atoms.insert("mass", np.full(len(pos), 39.948))
    frame = mp.Frame()
    frame["atoms"] = atoms
    frame.box = mp.Box(mp.core.Cell.lengths([12.0] * 3))
    return frame


def _langevin_md() -> mp.md.MD:
    sim = mp.md.MD("real")
    sim.set_integrator(mp.md.VelocityVerlet(2.0))
    sim.set_neighbor(mp.core.NeighborList(5.5), skin=1.0)
    sim.set_evaluator(mp.ff.potential.PairLjCut(0.238, 3.405, 4.5))
    sim.set_thermostat(mp.md.FixLangevin(120.0, 100.0, 2.0, seed=3))
    return sim


def test_a_checkpoint_through_mrec_resumes_the_run():
    whole = _argon()
    _langevin_md().run(whole, 10)

    first = _langevin_md()
    frame = _argon()
    first.run(frame, 5)
    record = bytearray()
    mp.io.write_mrec(record, frame=frame, meta={"step": first.step})
    stored = mp.io.mrec.MrecFile(bytes(record))
    restored = stored.frame()
    resumed = _langevin_md()
    resumed.resume(stored.meta()["step"])
    resumed.run(restored, 5)

    assert resumed.step == 10
    for key in ("x", "y", "z", "vx", "vy", "vz"):
        np.testing.assert_allclose(
            np.asarray(restored["atoms"][key]),
            np.asarray(whole["atoms"][key]),
            rtol=1e-10,
            atol=1e-12,
        )
