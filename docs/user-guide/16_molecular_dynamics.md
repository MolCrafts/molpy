# Molecular Dynamics

Step a frame through time in-process: `mp.md.MD` integrates on the CPU, takes
its forces from a compiled force field or from a model of your own, and writes
the result back onto the same frame.

## What the driver is

A molecular-dynamics step is a fixed sequence of small jobs: half a kick, a
drift, a neighbour-list check, a force evaluation, a thermostat, the second
half-kick, and whatever is measured at the end. LAMMPS names these points of
the step *phases*, and `mp.md.MD` walks the same list (`mp.md.Phase`). Each
part of a simulation is mounted at the phase where it acts, so the driver
itself holds no physics:

- the integrator (`VelocityVerlet`);
- the evaluator (`set_evaluator`): a compiled force field, or a potential of
  your own;
- the neighbour list;
- an optional thermostat (`FixLangevin`) and barostat (`BerendsenBarostat`).

The state lives on the frame and nowhere else. Positions, velocities, masses,
forces and image flags are `atoms` columns. The potential and kinetic
energies, the virial and the pressure are written into `frame.meta`. Because
the driver keeps nothing back, a run can be stopped, inspected and continued
at any step.

## Units

A run is in one LAMMPS units style, chosen when `MD` is built:
`MD(units="real")`. The styles are LAMMPS's eight — `lj` (the default, as in
LAMMPS), `real`, `metal`, `si`, `cgs`, `electron`, `micro` and `nano` — and they
mean exactly what the LAMMPS `units` command makes them mean. Every input is
in the run's style: masses, positions and velocities on the frame, the
timestep, force-field parameters, temperatures, pressures and damping times.
The driver applies the style's conversion constants (`ftm2v`, `mvv2e`, `boltz`,
`nktv2p`) with LAMMPS's own values, so a step in `real` is the step LAMMPS
takes in `real`.

Nothing is converted between styles. A Python potential declares the style of
the numbers it returns (`units = "metal"`), a force field its `units`, and a
frame read from a LAMMPS data file the style its title states; if any of them
is not the run's style, the run refuses to start and names both.

The timestep belongs to the run, as LAMMPS's `timestep` does, and starts at the
style's LAMMPS default: 0.005 τ in `lj`, 1 fs in `real`, 0.001 ps in `metal`.
`set_timestep` changes it; the integrator, thermostat and barostat all step by
it.

## A Lennard-Jones liquid at constant temperature

The frame holds 216 argon atoms on a lattice in a periodic box of 22.8 Å.
The force field is a single `PairLjCut`, cut at the customary 2.5σ = 8.5 Å. A pair kernel reads the pairs of the run's neighbour
list, so a list is mounted beside it. The list searches at its own cutoff, and
`skin` is the margin that lets one search serve many steps.

```python
import numpy as np

import molpy as mp


def argon(n_side=6, spacing=3.8):
    grid = np.array([[i, j, k] for i in range(n_side) for j in range(n_side) for k in range(n_side)])
    pos = spacing * grid + 0.5 * spacing
    atoms = mp.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d], dtype=float))
    atoms.insert("mass", np.full(len(pos), 39.948))  # g/mol
    atoms.insert("element", np.array(["Ar"] * len(pos)))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(len(pos)))  # at rest until velocities are drawn
    frame = mp.Frame()
    frame["atoms"] = atoms
    frame.box = mp.Box(mp.core.Cell.lengths([n_side * spacing] * 3))
    return frame


frame = argon()
velocities = mp.md.MaxwellBoltzmann(120.0, units="real", seed=1).velocities(
    np.asarray(frame["atoms"]["x", "y", "z"]), np.asarray(frame["atoms"]["mass"])
)
frame["atoms"]["vx", "vy", "vz"] = np.ascontiguousarray(velocities)

sim = mp.md.MD(units="real")
sim.set_timestep(2.0)  # fs
sim.set_integrator(mp.md.VelocityVerlet())
sim.set_neighbor(mp.core.NeighborList(9.5), skin=1.0)  # pairs within 8.5 Å reach the kernel
sim.set_evaluator(mp.ff.potential.PairLjCut(0.238, 3.405, 8.5))  # kcal/mol, Å
sim.set_thermostat(mp.md.FixLangevin(120.0, 200.0, seed=7))  # K, fs
sim.run(frame, 200)

print(sim.step, frame.meta["energy"], frame.meta["kinetic"])
```

`MaxwellBoltzmann` sets velocities as LAMMPS `velocity all create` does: it
draws them in the style's velocity unit, removes the centre-of-mass motion, and
scales them so the temperature — counted over `3N − 3` degrees of freedom, as
LAMMPS `compute temp` counts it — is exactly the one asked for. `FixLangevin` adds a drag and a
random force at `post_force`. Its noise is keyed by the run's step and each
atom's id, so a run is reproducible from its seed. `sim.step` counts across
calls to `run`: two runs of 100 steps are one run of 200.

## Pressure

`BerendsenBarostat` rescales the cell and the positions about the cell's
centre at the end of each step, by `μ³ = 1 − (dt/τ)(P₀ − P)/B`, toward a target
pressure `P₀` (atm in `real`), as LAMMPS `fix press/berendsen iso` does. The
pressure is LAMMPS's, `(2K + tr W)/(3V)` times the style's `nktv2p`. It reads the step's virial, so every force provider must supply one;
a pair kernel always does. The instantaneous pressure is written to
`frame.meta["pressure"]`. `B` is the system's bulk modulus: the volume relaxes
with a time of about `τ·B_coupling/B_system`, so an overestimate only slows the
coupling down. Berendsen coupling relaxes the volume but does not sample the
isothermal–isobaric ensemble: use it to equilibrate a density.

```python
sim.set_barostat(mp.md.BerendsenBarostat(1.0, 1000.0, 1.0e3))  # atm, fs, atm
volume = frame.box.volume()
sim.run(frame, 100)
print(frame.meta["pressure"], frame.box.volume() / volume)
```

The neighbour list follows the cell. It searches again once the atoms' motion
together with the strain of the cell since the last search could have brought
a pair across the skin.

## Trajectories and checkpoints

A run is a sequence of calls to `run`, and the frame between two calls *is*
the state. A trajectory is therefore written by the caller, every `k` steps,
with the writer of the format it needs:

```python
trajectory = bytearray()
with mp.io.xyz.XyzWriter(trajectory) as writer:
    for _ in range(5):
        sim.run(frame, 20)
        writer.write(frame)
```

A checkpoint is the frame — positions, velocities, box, image flags and the
last step's forces — and the step number. A `*.mrec` record holds both.
`resume` continues from it: the next run counts on from that step and takes the
frame's forces as that step's, instead of evaluating them again.

```python
record = bytearray()
mp.io.write_mrec(record, frame=frame, meta={"step": sim.step})

stored = mp.io.mrec.MrecFile(bytes(record))
restart = mp.md.MD(units="real")
restart.set_timestep(2.0)
restart.set_integrator(mp.md.VelocityVerlet())
restart.set_neighbor(mp.core.NeighborList(9.5), skin=1.0)
restart.set_evaluator(mp.ff.potential.PairLjCut(0.238, 3.405, 8.5))
restart.set_thermostat(mp.md.FixLangevin(120.0, 200.0, seed=7))
restart.set_barostat(mp.md.BerendsenBarostat(1.0, 1000.0, 1.0e3))
restart.resume(stored.meta()["step"])
resumed = stored.frame()
restart.run(resumed, 50)
print(restart.step)
```

The restarted run draws the noise the uninterrupted one would have drawn. Its
neighbour list is searched afresh at the restored positions, so a pair sum may
add the same terms in another order and agree only to rounding.

## Energy conservation

Without a thermostat, the total energy of a velocity-Verlet run is conserved
up to a bounded fluctuation, and a slow drift measures how good the timestep
and the cutoff are. `mp.compute.EnergyDrift` fits a line through the total
energy and reports its slope per atom. Divide by the atom count, never by the
total energy: the zero of a potential is arbitrary, and a ratio to `E₀` can
hide a drift of meV per atom.

```python
nve = mp.md.MD(units="real")
nve.set_timestep(2.0)
nve.set_integrator(mp.md.VelocityVerlet())
nve.set_neighbor(mp.core.NeighborList(9.5), skin=1.0)
nve.set_evaluator(mp.ff.potential.PairLjCut(0.238, 3.405, 8.5))

n_atoms = frame["atoms"].n_rows
drift = mp.compute.EnergyDrift(n_atoms).accumulator()
for _ in range(20):
    nve.run(frame, 10)
    total = frame.meta["energy"] + frame.meta["kinetic"]
    drift.push(time=nve.step * 2.0, energy=total)  # fs, kcal/mol
result = drift.finish()
print(result.drift, result.rms)  # kcal/mol per atom per fs, kcal/mol per atom
```

## A potential of your own

A neural network, or any program that turns positions into forces by itself,
is a Python **potential**: an object satisfying `mp.ff.potential.Potential`.
It declares the LAMMPS units style it works in (which must be the run's) and
the cutoff of its own neighbour search, and implements two methods:

- `bind(frame, skin)` is called once before the first evaluation, and again
  after any change to the run. It reads what it needs from the frame (species,
  charges, the box) and returns its **exchange buffers**: positions and forces,
  `(N, 3)` float64, and optionally a `(6,)` virial `W = Σ_pairs f ⊗ d`
  (`−∂E/∂ε`): positive under compression, in the potential's energy unit,
  ordered `[xx, yy, zz, xy, xz, yz]`. An ASE stress is `−W/V` in
  `[xx, yy, zz, yz, xz, xy]` order; negate and reorder it.
- `compute(step, rebuild, box)` is called once per step. The driver has already
  written the positions into the buffer. The potential writes the forces (and
  the virial) into theirs in place, and returns the energy.

`set_evaluator` mounts it, with `skin` the margin of its own neighbour search
(left out, the style's LAMMPS default: 2 Å in `real` and `metal`).
The driver owns the neighbour-list cadence. `rebuild` is true when the list the
potential built at `cutoff + skin` may be stale, and `box` is the cell when it
changed since the last call. Raise from `compute` to stop the run: the
exception comes back from `run` unchanged, at the step it was raised.

```python
class Springs:
    """Each atom tethered to where it started: E = ½ k |x − x₀|², in `real`."""

    units = "real"
    cutoff = None  # searches no neighbours

    def __init__(self, k):
        self.k = k

    def bind(self, frame, skin):
        self.x0 = np.asarray(frame["atoms"]["x", "y", "z"]).copy()
        self.positions = np.zeros_like(self.x0)
        self.forces = np.zeros_like(self.x0)
        return self.positions, self.forces

    def compute(self, step, rebuild, box):
        d = self.positions - self.x0
        self.forces[:] = -self.k * d
        return 0.5 * self.k * float((d * d).sum())


tethered = mp.md.MD(units="real")
tethered.set_integrator(mp.md.VelocityVerlet())  # the default 1 fs
tethered.set_evaluator(Springs(k=2.0))
tethered.run(argon(), 50)
```

The buffers are what makes the exchange cheap. For a model on a GPU they are
numpy views of pinned host tensors, so each step costs one copy each way and
one synchronisation. Nothing is allocated, and only one call crosses into
Python:

```python
# docs: skip — needs torch and a CUDA device, not docs dependencies
import torch


class TorchPotential:
    units = "metal"
    cutoff = 5.0

    def __init__(self, model):
        self.model = model.cuda()

    def bind(self, frame, skin):
        n = frame["atoms"].n_rows
        self.pos_host = torch.empty((n, 3), dtype=torch.float64, pin_memory=True)
        self.force_host = torch.empty((n, 3), dtype=torch.float64, pin_memory=True)
        self.pos_dev = torch.empty((n, 3), dtype=torch.float64, device="cuda")
        self.skin = skin
        return self.pos_host.numpy(), self.force_host.numpy()

    # `set_cell`, `build_neighbors` and the call are the model's own API.
    def compute(self, step, rebuild, box):
        self.pos_dev.copy_(self.pos_host, non_blocking=True)
        if box is not None:
            self.model.set_cell(torch.as_tensor(box.matrix, device="cuda"))
        if rebuild:
            self.model.build_neighbors(self.pos_dev, self.cutoff + self.skin)
        energy, forces = self.model(self.pos_dev)
        self.force_host.copy_(forces, non_blocking=True)
        torch.cuda.synchronize()
        return float(energy)
```

The same object also joins a `Potentials` collection (`push`), whose
`evaluate` and `mp.optimize.Lbfgs` call it with `rebuild=True` every time,
since neither has a cadence. In MD, mount it directly: inside a collection it
would get no cell, no cadence and no error that stops the run, so MD refuses
a collection that holds one.
