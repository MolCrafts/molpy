"""ForceField + Frame MD — the one driver over the Rust integrators.

Public path: ``molrs.md.MdDriver``. Frame topology is compiled per
:meth:`MdDriver.run`; the Rust ``VelocityVerlet`` owns the neighbour loop (the
``VerletSkin`` rebuild policy and pair feeding) — Python never does pair
bookkeeping.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._native import (
    ForceField,
    KineticEnergy,
    NeighborList,
    PotentialCompiler,
    Potentials,
    VerletSkin,
)
from .._native import md as _md

_NEIGHBOR_DEFAULTS = {
    "cutoff": None,
    "skin": 2.0,
    "every": 1,
    "delay": 0,
    "check": True,
}


def _stack_vel(atoms: object, shape: tuple[int, int]) -> NDArray[np.float64]:
    if all(name in atoms for name in ("vx", "vy", "vz")):
        return np.stack(
            [
                np.asarray(atoms["vx"], dtype=np.float64),
                np.asarray(atoms["vy"], dtype=np.float64),
                np.asarray(atoms["vz"], dtype=np.float64),
            ],
            axis=1,
        )
    return np.zeros(shape, dtype=np.float64)


class MdDriver:
    """Run a ``ForceField`` (or pre-built potential) over a ``Frame``::

        from molrs import md

        driver = md.MdDriver()
        driver.set_forcefield(ff)
        driver.set_neighbors(cutoff=7.5, skin=2.0)
        state = driver.run(frame, 1000, dt=1.0, kb=molrs.core.UnitPreset("real").boltzmann())

    The driver is unit-agnostic: nothing here converts units.
    Force-field parameters must already be consistent with ``dt`` / ``mass``
    / velocities. ``thermo=N`` requires an explicit ``kb=``.

    Precision is ``MdDriver(dtype=np.float64)`` only; float32 / mixed belong in
    the Rust integrator.

    After :meth:`run`, :attr:`n_edges` / :attr:`n_rebuilds` /
    :attr:`ago` report the run's neighbour state (``None`` when the run had
    no neighbour list) and :attr:`thermo` holds the sampled observables.
    """

    def __init__(self, *, dtype=np.float64) -> None:
        dt = np.dtype(dtype)
        if dt != np.dtype(np.float64):
            raise ValueError(
                "MdDriver(dtype=) currently accepts only numpy.float64; "
                "float32 / mixed belong in the Rust integrator"
            )
        self.dtype = dt
        self._forcefield: object | None = None
        self._potential: object | None = None
        self._skin: VerletSkin | None = None
        self._skin_used = False
        self._neighbor_config: dict | None = None
        self._integrator: _md.VelocityVerlet | None = None
        self.thermo: list[dict[str, float]] = []

    # -- configuration ------------------------------------------------------

    def set_forcefield(self, forcefield: object) -> MdDriver:
        """Attach a ``ForceField``; each :meth:`run` compiles it per frame."""
        if not isinstance(forcefield, ForceField):
            raise TypeError(
                f"set_forcefield expects a ForceField, got {type(forcefield).__name__}"
            )
        self._forcefield = forcefield
        self._potential = None
        return self

    def set_potential(self, potential: object) -> MdDriver:
        """Attach a pre-built potential (advanced; replaces :meth:`set_forcefield`).

        Accepts a compiled ``Potentials`` collection (e.g. one
        :class:`molrs.ff.compile.ExplicitTerms` built), an ``PairLjCut``, or a
        ``Potential`` subclass instance. The caller owns units — and, when skipping :meth:`set_neighbors`, neighbor correctness too:
        compiled ``Potentials`` evaluate exactly the topology (any ``pairs``
        block included) they were bound to; nothing is rebuilt as coordinates
        move. A ``Potentials`` collection is **moved** into the run's
        integrator — one run per attach.
        """
        if isinstance(potential, Potentials) and len(potential) == 0:
            raise ValueError(
                "Potentials is still deferred (len==0); compile with "
                "PotentialCompiler(ff).compile(frame) first"
            )
        self._potential = potential
        self._forcefield = None
        return self

    def set_neighbors(
        self,
        neighbors: VerletSkin | None = None,
        *,
        cutoff: float | None = None,
        skin: float | None = None,
        every: int | None = None,
        delay: int | None = None,
        check: bool | None = None,
    ) -> MdDriver:
        """Configure the neighbour list the run's integrator will own.

        Two mutually exclusive forms:

        * **Prebuilt** — pass a ``VerletSkin`` for full control of the search
          engine and rebuild policy. It is **moved** into the next run's
          integrator (single-shot); its force ``cutoff`` must cover the pair
          style's own cutoff.
        * **Kwargs** — each run builds a fresh
          ``VerletSkin(NeighborList(cutoff + skin), cutoff, …)`` over the
          frame. ``cutoff`` is the force cutoff, in the coordinates' length unit (default: derived from
          the pair style's ``cutoff`` param); ``skin`` is the Verlet buffer
          (default 2.0); ``every`` / ``delay`` / ``check`` mirror the
          ``VerletSkin`` rebuild policy (defaults 1 / 0 / True).

        Only consulted when the run has a nonbond term (force-field pair
        styles, or a :meth:`set_potential` potential that needs pairs fed);
        bonded-only force fields need no neighbour configuration.
        """
        kwargs_given = any(v is not None for v in (cutoff, skin, every, delay, check))
        if neighbors is not None:
            if kwargs_given:
                raise ValueError(
                    "set_neighbors takes a prebuilt VerletSkin OR kwargs, not both"
                )
            if not isinstance(neighbors, VerletSkin):
                raise TypeError(
                    f"neighbors must be a VerletSkin, got {type(neighbors).__name__}"
                )
            self._skin = neighbors
            self._skin_used = False
            self._neighbor_config = None
        else:
            if cutoff is not None and float(cutoff) <= 0.0:
                raise ValueError("cutoff must be > 0")
            if skin is not None and float(skin) < 0.0:
                raise ValueError("skin must be >= 0")
            config = dict(_NEIGHBOR_DEFAULTS)
            if cutoff is not None:
                config["cutoff"] = float(cutoff)
            if skin is not None:
                config["skin"] = float(skin)
            if every is not None:
                config["every"] = int(every)
            if delay is not None:
                config["delay"] = int(delay)
            if check is not None:
                config["check"] = bool(check)
            self._skin = None
            self._neighbor_config = config
        return self

    # -- run ----------------------------------------------------------------

    @property
    def n_edges(self) -> int | None:
        """Pair edges in the last run's list (``None`` without neighbors)."""
        return None if self._integrator is None else self._integrator.n_edges

    @property
    def n_rebuilds(self) -> int | None:
        """Neighbour rebuilds during the last run (``None`` without neighbors)."""
        return None if self._integrator is None else self._integrator.n_rebuilds

    @property
    def ago(self) -> int | None:
        """Updates since the last rebuild (``None`` without neighbors)."""
        return None if self._integrator is None else self._integrator.ago

    def _force_cutoff(self, config: dict) -> float:
        """The force cutoff the neighbour list must cover.

        First of: ``set_neighbors(cutoff=…)``, the largest style-level
        ``cutoff`` the force field declares, a prebuilt skin's own cutoff.
        A neighbour-driven pair style must declare one — ``compile_typed``
        refuses it otherwise — so the second of those is normally the answer.
        """
        if config["cutoff"] is not None:
            return float(config["cutoff"])
        ff = self._forcefield
        declared = [style["cutoff"] for style in ff.get_styles("pair")]
        found = [float(c) for c in declared if c is not None]
        if found:
            return max(found)
        if self._skin is not None:
            return float(self._skin.cutoff)
        raise ValueError(
            "cannot derive a force cutoff: no pair style declares 'cutoff'. "
            "Call set_neighbors(cutoff=..., skin=...) before run."
        )

    def _build_skin(
        self, frame: object, pos: NDArray[np.float64], force_cutoff: float | None
    ) -> VerletSkin:
        """One fresh ``VerletSkin`` for this run (prebuilt skins pass through)."""
        if self._skin is not None:
            skin = self._skin
            if force_cutoff is not None and force_cutoff > float(skin.cutoff) + 1e-12:
                raise ValueError(
                    f"prebuilt VerletSkin cutoff {skin.cutoff} is smaller "
                    f"than the pair-style force cutoff {force_cutoff}; "
                    "pairs would be silently missed"
                )
            # Moved into the integrator below: single-shot by construction.
            self._skin = None
            self._skin_used = True
            return skin
        if self._skin_used and self._neighbor_config is None:
            raise ValueError(
                "the prebuilt VerletSkin from set_neighbors was moved into a "
                "previous run's integrator; call set_neighbors again with a "
                "fresh VerletSkin or kwargs"
            )
        config = self._neighbor_config or dict(_NEIGHBOR_DEFAULTS)
        cutoff = config["cutoff"] if config["cutoff"] is not None else force_cutoff
        if not cutoff:
            raise ValueError(
                "cannot derive a force cutoff for the neighbour list. Call "
                "set_neighbors(cutoff=..., skin=...) before run."
            )
        box = getattr(frame, "box", None)
        if box is None:
            raise ValueError(
                "a neighbour-driven run needs periodic minimum images: set frame.box"
            )
        skin_width = float(config["skin"])
        return VerletSkin(
            NeighborList(float(cutoff) + skin_width),
            float(cutoff),
            pos,
            box,
            skin=skin_width,
            every=int(config["every"]),
            delay=int(config["delay"]),
            check=bool(config["check"]),
        )

    def _assemble(
        self,
        frame: object,
        dt: float,
        pos: NDArray[np.float64],
        mass: NDArray[np.float64],
    ) -> _md.VelocityVerlet:
        """Wire one run. This single step does exactly:

        1. **Compile the potential.** ``set_forcefield`` path with a pair
           style: ``PotentialCompiler(ff).compile_typed(frame)`` — kernels
           keyed on the atoms rather than on a ``pairs`` block, each carrying
           the force field's own ``special_bonds`` weights. Without a pair
           style: ``PotentialCompiler(ff).compile(frame)``, which is the
           bonded-only case.
           ``set_potential`` path: adopt the attached potential as-is (caller
           owns units).
        2. **Build the neighbour state.** With a nonbond term: a fresh
           ``VerletSkin(NeighborList(rc + skin), rc, pos, frame.box, …)``
           from the :meth:`set_neighbors` kwargs (defaults otherwise), or
           the prebuilt skin (single-shot; cutoff checked against the force
           cutoff). Bonded-only runs carry no neighbour state.
        3. **Construct the integrator.** ``VelocityVerlet(dt,
           potential=…, neighbors=…, mass=…)`` — potential, skin and the
           loop's pair feeding are **moved** into Rust; nothing to bookkeep
           in Python.
        """
        if self._forcefield is not None:
            ff = self._forcefield
            if ff.get_styles("pair"):
                # One call decides which kernel each style needs and how its
                # close neighbours are scaled (special_bonds included), so the
                # driver re-derives neither.
                config = self._neighbor_config or dict(_NEIGHBOR_DEFAULTS)
                pots = PotentialCompiler(ff).compile_typed(frame)
                neighbors = self._build_skin(frame, pos, self._force_cutoff(config))
            else:
                pots = PotentialCompiler(ff).compile(frame)
                if len(pots) == 0:
                    raise ValueError(
                        "PotentialCompiler(forcefield).compile(frame) produced "
                        "empty Potentials"
                    )
                neighbors = None
        elif self._potential is not None:
            pots = self._potential
            if isinstance(pots, Potentials):
                self._potential = None  # moved into the integrator below
            neighbors = None
            if self._skin is not None or self._neighbor_config is not None:
                neighbors = self._build_skin(frame, pos, None)
        else:
            raise RuntimeError("set_forcefield or set_potential before run")
        return _md.VelocityVerlet(
            float(dt),
            potential=pots,
            neighbors=neighbors,
            mass=mass,
            # The cell the positions are folded into each step. Without it
            # `MdState.images` stays zero and the wrapped coordinates lose the
            # history that makes them readable as a trajectory.
            box=getattr(frame, "box", None),
        )

    def run(
        self,
        frame: object,
        n_steps: int,
        *,
        dt: float,
        mass: NDArray[np.floating] | float | None = None,
        temperature: float | None = None,
        seed: int = 0,
        thermo: int | None = None,
        kb: float | None = None,
    ) -> _md.MdState:
        """Integrate ``n_steps`` NVE steps; write pos and vel back to ``frame``.

        Assembly is per call (:meth:`_assemble`) — a driver configured via
        :meth:`set_forcefield` runs again and again. ``mass=`` overrides
        ``frame["atoms"]["mass"]`` (a scalar broadcasts). ``temperature=``
        draws initial velocities through ``MaxwellBoltzmann(kb * temperature,
        seed=seed)`` (LAMMPS ``velocity create``); otherwise the frame's
        ``vx``/``vy``/``vz`` (or zeros) are used. ``thermo=N`` samples
        ``step`` / ``pe`` / ``ke`` / ``etotal`` / ``temp`` every N steps into
        :attr:`thermo` (the ``temp`` column uses ``kb=``). Every quantity is in
        the caller's unit system — ``kb`` is Boltzmann's constant in it, e.g.
        ``molrs.core.UnitPreset("real").boltzmann()`` — and nothing is
        converted. Returns the final ``MdState``.
        """
        if self._forcefield is None and self._potential is None:
            raise RuntimeError("set_forcefield or set_potential before run")
        if (thermo is not None or temperature is not None) and kb is None:
            raise ValueError(
                "MdDriver.run(thermo=...) / temperature= requires an explicit kb="
            )
        atoms = frame["atoms"]
        pos = atoms.coords
        if mass is None:
            if "mass" not in atoms:
                raise ValueError(
                    "frame['atoms'] must carry a mass column, or pass mass="
                )
            mass_arr = np.asarray(atoms["mass"], dtype=np.float64)
        else:
            mass_arr = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        if mass_arr.size == 1:
            mass_arr = np.full(pos.shape[0], float(mass_arr[0]))
        if temperature is not None:
            vel = _md.MaxwellBoltzmann(
                float(kb) * float(temperature), seed=int(seed)
            ).velocities(pos, mass_arr)
        else:
            vel = _stack_vel(atoms, pos.shape)

        integrator = self._assemble(frame, dt, pos, mass_arr)
        self._integrator = integrator
        state = integrator.initial(pos, vel)
        self.thermo = []
        if thermo is None:
            if int(n_steps) > 0:
                state = integrator.advance_n(state, int(n_steps))
        else:
            interval = int(thermo)
            if interval < 1:
                raise ValueError(f"thermo must be >= 1, got {thermo}")
            n_dof = max(1, 3 * pos.shape[0] - int(integrator.removed_dof))
            # Unit-agnostic, like the integrators: the caller's units are
            # consistent (mvv2e = 1) and kb is Boltzmann's constant in them.
            # Both constants are overridden, so the preset (the core
            # constructor's required argument) contributes nothing.
            kinetic = KineticEnergy("real", mvv2e=1.0, boltzmann=float(kb))

            def record(step: int, state: _md.MdState) -> None:
                k = kinetic.compute(masses=mass_arr, velocities=state.vel)
                pe = float(state.energy)
                self.thermo.append(
                    {
                        "step": step,
                        "pe": pe,
                        "ke": k.energy,
                        "etotal": pe + k.energy,
                        "temp": k.temperature(n_dof),
                    }
                )

            record(0, state)
            done = 0
            while done < int(n_steps):
                n = min(interval, int(n_steps) - done)
                state = integrator.advance_n(state, n)
                done += n
                record(done, state)

        atoms["x"] = state.pos[:, 0]
        atoms["y"] = state.pos[:, 1]
        atoms["z"] = state.pos[:, 2]
        atoms["vx"] = state.vel[:, 0]
        atoms["vy"] = state.vel[:, 1]
        atoms["vz"] = state.vel[:, 2]
        return state
