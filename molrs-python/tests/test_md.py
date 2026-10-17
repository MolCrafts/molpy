"""Seam tests for molrs.md — construction, types, move semantics, a few steps."""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.ff.potential import PairLjCut, Potential
from molrs.md import MdDriver, MaxwellBoltzmann, VelocityVerlet


class Harmonic:
    """Duck-typed potential: no base class."""

    def __init__(self, k: float = 1.0) -> None:
        self.k = k

    def evaluate(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        e = 0.5 * self.k * float(np.sum(pos * pos))
        return e, (-self.k * pos)


class TestPotentialProtocol:
    def test_structural_class_is_potential(self) -> None:
        assert isinstance(Harmonic(), Potential)

    def test_ljcut_is_pair_driven_not_coordinates_only(self) -> None:
        """``PairLjCut`` is priced over a pair table (``evaluate_pairs``), so
        it is not the coordinates-only ``Potential`` a duck-typed provider
        is; an integrator still takes it, feeding it the pairs."""
        lj = PairLjCut(1.0, 1.0, 2.5)
        assert not isinstance(lj, Potential)
        assert callable(lj.evaluate_pairs) and callable(lj.pair_energy_force)
        vv = VelocityVerlet(
            0.001,
            potential=PairLjCut(1.0, 1.0, 2.5),
            neighbors=molrs.core.VerletSkin(
                molrs.core.NeighborList(3.0),
                2.5,
                np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
                molrs.core.Box(molrs.core.Cell.lengths([10.0] * 3)),
                skin=0.5,
            ),
            mass=np.ones(2),
        )
        state = vv.initial(
            np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]), np.zeros((2, 3))
        )
        e, _ = lj.pair_energy_force(1.44, [1.2, 0.0, 0.0])
        assert state.energy == pytest.approx(e)

    def test_md_defines_no_potential(self) -> None:
        """MD integrates potentials; it does not define or re-export one."""
        from molrs import md

        for name in ("Potential", "Potentials", "PairLjCut"):
            assert not hasattr(md, name)
            assert not hasattr(molrs._native.md, name)


class TestMdDtype:
    def test_float64_is_accepted(self) -> None:
        md = MdDriver(dtype=np.float64)
        assert md.dtype == np.dtype(np.float64)

    def test_float32_is_rejected_with_rust_message(self) -> None:
        with pytest.raises(ValueError, match="Rust"):
            MdDriver(dtype=np.float32)


class TestDispatch:
    def test_duck_typed_potential_takes_coordinates_only(self) -> None:
        calls: list[int] = []

        class Probe:
            def evaluate(self, pos):
                calls.append(pos.ndim if hasattr(pos, "ndim") else -1)
                pos = np.asarray(pos, dtype=np.float64)
                return 0.0, np.zeros_like(pos)

        pos = np.zeros((2, 3))
        vel = np.zeros((2, 3))
        vv = VelocityVerlet(0.001, potential=Probe(), mass=np.ones(2))
        vv.initial(pos, vel)
        assert calls and all(c == 2 for c in calls)

    def test_potentials_take_the_native_fast_path(self) -> None:
        from molrs.ff.potential import Potentials

        pots = Potentials()
        # empty collection is a native type; take_potential must not duck-wrap it.
        vv = VelocityVerlet(0.001, potential=pots, mass=np.ones(1))
        assert vv is not None


class _Free:
    """No force: the atom drifts at constant velocity."""

    def evaluate(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        return 0.0, np.zeros_like(pos)


class TestImageFlags:
    def test_a_crossing_survives_later_advances(self) -> None:
        """``images`` accumulates the crossings over every ``advance``: the
        flag set when the atom leaves the cell is still there steps later,
        and ``pos + L·images`` is the straight unwrapped path."""
        edge, dt, v = 10.0, 0.5, 1.0
        vv = VelocityVerlet(
            dt,
            potential=_Free(),
            mass=np.ones(1),
            box=molrs.core.Box(molrs.core.Cell.lengths([edge] * 3)),
        )
        state = vv.initial(np.array([[9.2, 5.0, 5.0]]), np.array([[v, 0.0, 0.0]]))
        n_steps = 6  # crosses x = 10 on the second step
        for _ in range(n_steps):
            state = vv.advance(state)
        assert state.images.tolist() == [[1, 0, 0]]
        unwrapped = state.pos + edge * state.images
        np.testing.assert_allclose(unwrapped, [[9.2 + n_steps * dt * v, 5.0, 5.0]])
        state = vv.advance_n(state, 4)
        assert state.images.tolist() == [[1, 0, 0]]

    def test_a_duck_typed_state_carries_its_images(self) -> None:
        vv = VelocityVerlet(
            0.5,
            potential=_Free(),
            mass=np.ones(1),
            box=molrs.core.Box(molrs.core.Cell.lengths([10.0] * 3)),
        )
        seed = vv.initial(np.array([[5.0, 5.0, 5.0]]), np.array([[1.0, 0.0, 0.0]]))

        class State:
            pos = seed.pos
            vel = seed.vel
            forces = seed.forces
            energy = seed.energy
            images = np.array([[2, 0, -1]], dtype=np.int32)

        assert vv.advance(State()).images.tolist() == [[2, 0, -1]]


class TestMass:
    def test_a_scalar_mass_is_refused_with_the_array_spelling(self) -> None:
        """Integrators are built before they see ``N``; the mass is per atom."""
        with pytest.raises(TypeError, match=r"np\.full\(n_atoms, mass\)"):
            VelocityVerlet(0.001, potential=_Free(), mass=39.948)
        with pytest.raises(TypeError, match=r"mass.*1-D float64"):
            VelocityVerlet(0.001, potential=_Free(), mass=np.ones((2, 1)))
        with pytest.raises(TypeError, match="scalar"):
            MaxwellBoltzmann(1.0).velocities(np.zeros((2, 3)), 1.0)
        vv = VelocityVerlet(0.001, potential=_Free(), mass=np.full(2, 39.948))
        assert vv.initial(np.zeros((2, 3)), np.zeros((2, 3))).pos.shape == (2, 3)


class TestMaxwellBoltzmann:
    def test_kbt_constructor(self) -> None:
        mb = MaxwellBoltzmann(molrs.core.UnitPreset("real").boltzmann() * 300.0, seed=1)
        pos = np.zeros((4, 3))
        vel = mb.velocities(pos, np.ones(4))
        assert vel.shape == (4, 3)
        assert np.all(np.isfinite(vel))


class TestMdDriverThermo:
    def test_thermo_reads_kinetic_energy_in_the_callers_units(self) -> None:
        """``ke`` is ``½ Σ m v²`` and ``temp`` its equipartition temperature
        under the caller's ``kb`` — the core ``KineticEnergy`` with
        ``mvv2e = 1``, not a second formula."""
        atoms = molrs.core.Block()
        pos = np.array([[0.1, 0.0, 0.0], [0.0, -0.2, 0.3]])
        for d, key in enumerate("xyz"):
            atoms.insert(key, np.ascontiguousarray(pos[:, d]))
        for d, key in enumerate(("vx", "vy", "vz")):
            atoms.insert(key, np.array([0.5, -0.25]) * (d + 1))
        atoms.insert("mass", np.array([1.0, 2.0]))
        frame = molrs.core.Frame()
        frame["atoms"] = atoms

        kb = 0.5
        driver = MdDriver().set_potential(Harmonic())
        driver.run(frame, 4, dt=0.01, kb=kb, thermo=2)

        assert [row["step"] for row in driver.thermo] == [0, 2, 4]
        first = driver.thermo[0]
        vel = np.array([[0.5, 1.0, 1.5], [-0.25, -0.5, -0.75]])
        ke = 0.5 * float(np.sum(np.array([1.0, 2.0])[:, None] * vel**2))
        assert first["ke"] == pytest.approx(ke)
        assert first["etotal"] == pytest.approx(first["pe"] + ke)
        n_dof = 3 * 2 - int(driver._integrator.removed_dof)
        assert first["temp"] == pytest.approx(2.0 * ke / (n_dof * kb))
