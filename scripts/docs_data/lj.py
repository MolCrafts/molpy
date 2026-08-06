"""Lennard-Jones molecular dynamics for liquid argon.

Figures in ``docs/compute/`` are drawn from trajectories produced here, not from
hand-typed points. Argon is used because its Lennard-Jones parameters are
standard and its measured transport coefficients are in the literature, so a
reader can check the numbers the docs quote:

    sigma = 3.405 A, epsilon/kB = 119.8 K, m = 39.948 g/mol
    Rahman, A. (1964). Phys. Rev. 136, A405. DOI: 10.1103/PhysRev.136.A405

Units are LAMMPS *real*: length A, energy kcal/mol, mass g/mol, time fs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Boltzmann constant, kcal/(mol*K).
KB = 0.0019872041

#: (kcal/mol/A) / (g/mol) -> A/fs**2.
ACCEL = 4.184e-4


@dataclass(frozen=True)
class ArgonLJ:
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
        avogadro = 6.02214076e23
        return mass_density_g_cm3 / self.mass * avogadro * 1e-24


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


class LennardJonesMD:
    """Velocity-Verlet MD of a cubic Lennard-Jones fluid.

    The pair loop is a dense minimum-image evaluation. That is O(N^2) and is
    only honest at the few-hundred-atom sizes used for documentation figures;
    production work belongs in a real engine.
    """

    def __init__(
        self,
        n_atoms: int = 500,
        mass_density: float = 1.374,
        potential: ArgonLJ | None = None,
        seed: int = 0,
    ) -> None:
        self.potential = potential or ArgonLJ()
        self.n_atoms = n_atoms
        density = self.potential.number_density(mass_density)
        self.box_length = float((n_atoms / density) ** (1.0 / 3.0))
        self._rng = np.random.default_rng(seed)
        self.xyz = self._fcc_lattice(n_atoms, self.box_length)
        self.velocities = np.zeros_like(self.xyz)

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

    def _forces(self, xyz: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (forces kcal/(mol*A), potential energy kcal/mol)."""
        lj = self.potential
        length = self.box_length
        delta = xyz[:, None, :] - xyz[None, :, :]
        delta -= length * np.round(delta / length)
        r2 = np.einsum("ijk,ijk->ij", delta, delta)
        np.fill_diagonal(r2, np.inf)

        inside = r2 < lj.cutoff**2
        inv_r2 = np.where(inside, 1.0 / r2, 0.0)
        s6 = lj.sigma**6
        s6_r6 = s6 * inv_r2**3
        s12_r12 = s6_r6**2

        # U(r) = 4 eps (s12/r12 - s6/r6), shifted so U(rc) = 0.
        shift = 4.0 * lj.epsilon * ((s6 / lj.cutoff**6) ** 2 - s6 / lj.cutoff**6)
        pair_energy = np.where(
            inside, 4.0 * lj.epsilon * (s12_r12 - s6_r6) - shift, 0.0
        )
        energy = 0.5 * float(pair_energy.sum())

        # F(r) = 24 eps (2 s12/r12 - s6/r6) / r^2 * r_vec
        scale = 24.0 * lj.epsilon * (2.0 * s12_r12 - s6_r6) * inv_r2
        forces = np.einsum("ij,ijk->ik", scale, delta)
        return forces, energy

    def _kinetic(self) -> float:
        """Kinetic energy, kcal/mol."""
        v2 = float(np.einsum("ij,ij->", self.velocities, self.velocities))
        return 0.5 * self.potential.mass * v2 / ACCEL

    def _instant_temperature(self) -> float:
        dof = 3 * self.n_atoms - 3
        return 2.0 * self._kinetic() / (dof * KB)

    def thermalize(self, temperature: float) -> None:
        """Draw Maxwell-Boltzmann velocities and remove net momentum."""
        sigma_v = np.sqrt(KB * temperature / self.potential.mass * ACCEL)
        self.velocities = self._rng.normal(0.0, sigma_v, size=self.xyz.shape)
        self.velocities -= self.velocities.mean(axis=0)
        self._rescale(temperature)

    def _rescale(self, temperature: float) -> None:
        current = self._instant_temperature()
        if current > 0.0:
            self.velocities *= np.sqrt(temperature / current)

    def equilibrate(self, temperature: float, steps: int, dt: float = 10.0) -> None:
        """Run with periodic velocity rescaling to reach ``temperature``.

        Rescaling is a crude thermostat: it fixes the mean kinetic energy but
        distorts the dynamics, which is exactly why sampling below is NVE.
        """
        forces, _ = self._forces(self.xyz)
        for step in range(steps):
            forces = self._step(forces, dt)
            if step % 20 == 0:
                self._rescale(temperature)
        self._rescale(temperature)

    def _step(self, forces: np.ndarray, dt: float) -> np.ndarray:
        accel = forces / self.potential.mass * ACCEL
        self.velocities += 0.5 * accel * dt
        self.xyz = self.xyz + self.velocities * dt
        forces, _ = self._forces(self.xyz)
        accel = forces / self.potential.mass * ACCEL
        self.velocities += 0.5 * accel * dt
        return forces

    def sample(self, steps: int, stride: int = 5, dt: float = 10.0) -> Trajectory:
        """Run constant-energy dynamics and store every ``stride``-th frame."""
        forces, potential = self._forces(self.xyz)
        reference_energy = potential + self._kinetic()

        length = self.box_length
        wrapped: list[np.ndarray] = []
        unwrapped: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        temperatures: list[float] = []
        last_energy = reference_energy

        for step in range(steps):
            forces = self._step(forces, dt)
            if step % stride == 0:
                # Integration never folds coordinates, so self.xyz is already
                # the continuous (unwrapped) path that displacement kernels need.
                unwrapped.append(self.xyz.copy())
                wrapped.append(self.xyz - length * np.floor(self.xyz / length))
                velocities.append(self.velocities.copy())
                temperatures.append(self._instant_temperature())
            if step == steps - 1:
                _, potential = self._forces(self.xyz)
                last_energy = potential + self._kinetic()

        drift = abs(last_energy - reference_energy) / abs(reference_energy)
        return Trajectory(
            wrapped=np.array(wrapped),
            unwrapped=np.array(unwrapped),
            velocities=np.array(velocities),
            box_length=length,
            dt=dt * stride,
            temperature=float(np.mean(temperatures)),
            energy_drift=float(drift),
        )
