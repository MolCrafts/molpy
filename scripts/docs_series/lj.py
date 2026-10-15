"""Lennard-Jones molecular dynamics for liquid argon.

Figures in ``docs/compute/`` are drawn from trajectories produced here, not from
hand-typed points. Argon is used because its Lennard-Jones parameters are
standard and its measured transport coefficients are in the literature, so a
reader can check the numbers the docs quote:

    sigma = 3.405 A, epsilon/kB = 119.8 K, m = 39.948 g/mol
    Rahman, A. (1964). Phys. Rev. 136, A405. DOI: 10.1103/PhysRev.136.A405

Units are LAMMPS *real*: length A, energy kcal/mol, mass g/mol, time fs.

The dynamics is MolPy's own: ``mp.md.VelocityVerlet`` integrates the cut,
shifted Lennard-Jones pair kernel ``PairLjCut`` over a Verlet-skin neighbour
list, and ``KineticEnergy`` reports the kinetic energy and temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import molpy as mp
from molpy.compute import KineticEnergy, Unwrap
from molpy.core import NeighborList, UnitRegistry, VerletSkin, constants
from molpy.ff.potential import PairLjCut

_UNITS = UnitRegistry()

#: Boltzmann constant per mole, kcal/(mol*K): k_B N_A (the gas constant)
#: from J to kcal.
KB = constants.GAS_CONSTANT * _UNITS.factor("J", "kcal")

#: (kcal/mol/A) / (g/mol) -> A/fs**2: a force over a mass, both per mole,
#: is kcal/g per A, and kcal/g is a squared speed.
ACCEL = _UNITS.factor("kcal/g", "angstrom^2/fs^2")

#: A^2/fs -> cm^2/s (diffusion coefficients).
ANGSTROM2_PER_FS_TO_CM2_PER_S = _UNITS.factor("angstrom^2/fs", "cm^2/s")

#: fs -> ps and fs -> s.
FS_TO_PS = _UNITS.factor("fs", "ps")
FS_TO_S = _UNITS.factor("fs", "s")

#: The speed of light in cm/s, for wavenumbers (cm^-1) from frequencies (s^-1).
SPEED_OF_LIGHT_CM_PER_S = constants.SPEED_OF_LIGHT * _UNITS.factor("m/s", "cm/s")


@dataclass(frozen=True)
class ArgonLj:
    """Lennard-Jones parameters for argon in LAMMPS *real* units."""

    sigma: float = 3.405
    epsilon_kelvin: float = 119.8
    mass: float = 39.948
    cutoff_sigma: float = 2.5

    @property
    def epsilon(self) -> float:
        """Well depth, kcal/mol."""
        return self.epsilon_kelvin * KB

    @property
    def cutoff(self) -> float:
        """Pair cutoff, A."""
        return self.cutoff_sigma * self.sigma

    def number_density(self, mass_density_g_cm3: float) -> float:
        """Convert a mass density (g/cm^3) to a number density (atoms/A^3)."""
        per_cm3 = mass_density_g_cm3 / self.mass * constants.AVOGADRO
        return per_cm3 * _UNITS.factor("angstrom^3", "cm^3")


@dataclass(frozen=True)
class Trajectory:
    """A finished run.

    Attributes:
        wrapped: (n_frames, n_atoms, 3) coordinates folded into the box, A.
        unwrapped: (n_frames, n_atoms, 3) continuous coordinates, A.
        velocities: (n_frames, n_atoms, 3) velocities, A/fs.
        box_length: cubic edge, A.
        dt: time between stored frames, fs.
        temperature: mean instantaneous temperature over the run, K.
        energy_drift: |E(t) - E(0)| / |E(0)| at the last frame, dimensionless.
    """

    wrapped: np.ndarray
    unwrapped: np.ndarray
    velocities: np.ndarray
    box_length: float
    dt: float
    temperature: float
    energy_drift: float


#: Verlet skin on top of the pair cutoff, A. A 10 fs argon step moves an atom
#: about 0.05 A, so the list is rebuilt every few tens of steps.
SKIN = 1.0

#: Steps between velocity rescalings while equilibrating.
RESCALE_EVERY = 20


class LennardJonesMd:
    """Velocity-Verlet MD of a cubic Lennard-Jones fluid on ``mp.md``.

    MD is unit-agnostic: it integrates ``a = F / m``. With forces in
    kcal/(mol A) and positions in A, an acceleration in A/fs^2 needs the mass
    in kcal fs^2/(mol A^2), which is the molar mass divided by :data:`ACCEL`.
    The kinetic energy is read with ``KineticEnergy(units="real")`` on the
    molar mass, so the two unit conventions never mix.
    """

    def __init__(
        self,
        n_atoms: int = 500,
        mass_density: float = 1.374,
        potential: ArgonLj | None = None,
        seed: int = 0,
        timestep: float = 10.0,
    ) -> None:
        self.potential = potential or ArgonLj()
        self.n_atoms = n_atoms
        density = self.potential.number_density(mass_density)
        self.box_length = float((n_atoms / density) ** (1.0 / 3.0))
        self.box = mp.Box.cube(self.box_length)
        self._rng = np.random.default_rng(seed)
        self._masses = np.full(n_atoms, self.potential.mass)
        self._kinetic = KineticEnergy(units="real")
        self.dt = timestep

        lj = self.potential
        xyz = self._fcc_lattice(n_atoms, self.box_length)
        skin = VerletSkin(
            NeighborList(lj.cutoff + SKIN), lj.cutoff, xyz, self.box, skin=SKIN
        )
        self._integrator = mp.md.VelocityVerlet(
            timestep,
            potential=PairLjCut(lj.epsilon, lj.sigma, lj.cutoff),
            neighbors=skin,
            mass=self._masses / ACCEL,
            box=self.box,
        )
        self._dof = 3 * n_atoms - self._integrator.removed_dof
        self.state = self._integrator.initial(xyz, np.zeros_like(xyz))

    @staticmethod
    def _fcc_lattice(n_atoms: int, box_length: float) -> np.ndarray:
        """Place ``n_atoms`` on an FCC lattice filling the cube.

        Starting from a lattice (rather than random points) avoids the
        overlapping pairs that would blow up the first force evaluation.
        """
        cells = int(np.ceil((n_atoms / 4.0) ** (1.0 / 3.0)))
        a = box_length / cells
        basis = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        )
        sites = [
            (np.array([i, j, k]) + b) * a
            for i in range(cells)
            for j in range(cells)
            for k in range(cells)
            for b in basis
        ]
        return np.ascontiguousarray(np.array(sites)[:n_atoms])

    def _kinetic_energy(self, velocities: np.ndarray):
        return self._kinetic.compute(masses=self._masses, velocities=velocities)

    def _instant_temperature(self) -> float:
        return self._kinetic_energy(self.state.vel).temperature(self._dof)

    def _total_energy(self) -> float:
        return self.state.energy + self._kinetic_energy(self.state.vel).energy

    def _rescale(self, temperature: float) -> None:
        current = self._instant_temperature()
        if current > 0.0:
            self.state.vel = self.state.vel * np.sqrt(temperature / current)

    def thermalize(self, temperature: float) -> None:
        """Draw Maxwell-Boltzmann velocities and remove net momentum."""
        sigma_v = np.sqrt(KB * temperature / self.potential.mass * ACCEL)
        velocities = self._rng.normal(0.0, sigma_v, size=(self.n_atoms, 3))
        self.state.vel = velocities - velocities.mean(axis=0)
        self._rescale(temperature)

    def equilibrate(self, temperature: float, steps: int) -> None:
        """Run with periodic velocity rescaling to reach ``temperature``.

        Rescaling is a crude thermostat: it fixes the mean kinetic energy but
        distorts the dynamics, which is exactly why sampling below is NVE.
        """
        done = 0
        while done < steps:
            chunk = min(RESCALE_EVERY, steps - done)
            self.state = self._integrator.advance_n(self.state, chunk)
            self._rescale(temperature)
            done += chunk

    def sample(self, steps: int, stride: int = 5) -> Trajectory:
        """Run constant-energy dynamics and store every ``stride``-th frame.

        ``stride`` must keep a frame-to-frame step well under half the box:
        the unwrapped path is rebuilt from the stored frames by continuity.
        """
        reference_energy = self._total_energy()

        wrapped: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        temperatures: list[float] = []

        for step in range(steps):
            self.state = self._integrator.advance(self.state)
            if step % stride == 0:
                wrapped.append(self.state.pos)
                velocities.append(self.state.vel)
                temperatures.append(self._instant_temperature())

        drift = abs(self._total_energy() - reference_energy) / abs(reference_energy)
        # The integrator folds every step into the cell. Frames stored every
        # few steps move far less than half a box between frames, so the
        # continuous path is recovered by minimum image between frames.
        folded = np.array(wrapped)
        unwrapped = Unwrap().compute(positions=folded, boxes=[self.box] * len(folded))
        return Trajectory(
            wrapped=folded,
            unwrapped=unwrapped,
            velocities=np.array(velocities),
            box_length=self.box_length,
            dt=self.dt * stride,
            temperature=float(np.mean(temperatures)),
            energy_drift=float(drift),
        )
