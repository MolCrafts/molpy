"""Base classes for geometry optimization.

The optimizer operates directly on a :class:`molrs.Frame` — the universal
coordinate-plus-topology container. ``run(frame, ...)`` reads and writes the
frame's ``"atoms"`` coordinate columns in place and returns an
:class:`OptimizationResult` carrying the optimized frame. The potential
(:class:`~molpy.optimize.ForceFieldPotential`) is the Frame-consuming layer, so
both the optimizer and the potential share the same ``Frame`` object.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import molrs
import numpy as np


# Protocol for potentials (duck typing)
class PotentialLike(Protocol):
    """Protocol for potential functions evaluated on a :class:`molrs.Frame`."""

    def calc_energy(self, frame: "molrs.Frame") -> float: ...
    def calc_forces(self, frame: "molrs.Frame") -> np.ndarray: ...


@dataclass
class OptimizationResult:
    """Result of a geometry optimization.

    Attributes:
        frame: Final optimized ``molrs.Frame`` (same object when ``inplace=True``).
        energy: Final potential energy (force-field units, e.g. kcal/mol).
        fmax: Final maximum force component (energy units / angstrom).
        nsteps: Number of optimization steps taken.
        converged: Whether the convergence criterion was met.
        reason: Human-readable termination reason.
    """

    frame: "molrs.Frame"
    energy: float
    fmax: float
    nsteps: int
    converged: bool
    reason: str


class Optimizer(ABC):
    """Base class for Frame-native geometry optimizers.

    The optimizer reads/writes the coordinate columns of a ``molrs.Frame``'s
    ``"atoms"`` block and evaluates energy/forces by calling the potential on the
    same frame (``potential.calc_energy(frame)`` / ``calc_forces(frame)``). The
    frame *is* the optimized state — there is no structure-to-frame conversion.

    Args:
        potential: Potential exposing ``calc_energy(frame)`` / ``calc_forces(frame)``.

    Example:
        >>> from molpy.optimize import LBFGS, ForceFieldPotential
        >>>
        >>> potential = ForceFieldPotential(forcefield)  # molrs ForceField
        >>> opt = LBFGS(potential, maxstep=0.04, memory=20)
        >>> result = opt.run(frame, fmax=0.01, steps=500)  # frame: molrs.Frame
    """

    def __init__(self, potential: PotentialLike) -> None:
        self.potential = potential
        self._callbacks: list[tuple[Callable, int, dict]] = []

    # ===== Bridge methods: Frame coordinate columns <-> numpy arrays =====

    def get_positions(self, frame: "molrs.Frame") -> np.ndarray:
        """Return the frame's atom positions as an ``(N, 3)`` array."""
        atoms = frame["atoms"]
        x = np.asarray(atoms["x"], dtype=float)
        if x.size == 0:
            return np.empty((0, 3), dtype=float)
        y = np.asarray(atoms["y"], dtype=float)
        z = np.asarray(atoms["z"], dtype=float)
        return np.column_stack([x, y, z])

    def set_positions(self, frame: "molrs.Frame", positions: np.ndarray) -> None:
        """Write an ``(N, 3)`` (or flat ``3N``) position array into the frame.

        Uses ``Block.insert`` (an upsert), which both molrs Block variants
        support — the rich ``molrs.frame.Block`` (from ``Atomistic.to_frame``)
        and the raw core ``Block`` (from ``MMFFTypifier.typify``) — so the
        optimizer is agnostic to how the frame was built.
        """
        atoms = frame["atoms"]
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        atoms.insert("x", np.ascontiguousarray(positions[:, 0]))
        atoms.insert("y", np.ascontiguousarray(positions[:, 1]))
        atoms.insert("z", np.ascontiguousarray(positions[:, 2]))

    def get_energy_and_forces(self, frame: "molrs.Frame") -> tuple[float, np.ndarray]:
        """Evaluate energy and forces by calling the potential on the frame."""
        energy = self.potential.calc_energy(frame)
        forces = self.potential.calc_forces(frame)
        return energy, forces

    def get_energy(self, frame: "molrs.Frame") -> float:
        """Return the potential energy of the frame."""
        energy, _ = self.get_energy_and_forces(frame)
        return energy

    def get_forces(self, frame: "molrs.Frame") -> np.ndarray:
        """Return the forces on the frame's atoms."""
        _, forces = self.get_energy_and_forces(frame)
        return forces

    # ===== Abstract method for subclasses =====

    @abstractmethod
    def step(self, frame: "molrs.Frame") -> tuple[float, float]:
        """Perform one optimization step, mutating the frame's coordinates.

        Args:
            frame: ``molrs.Frame`` to optimize (modified in-place).

        Returns:
            ``(energy, fmax)`` after the step.
        """
        ...

    # ===== Public API =====

    def run(
        self,
        frame: "molrs.Frame",
        fmax: float = 0.01,
        steps: int = 1000,
        *,
        inplace: bool = True,
    ) -> OptimizationResult:
        """Optimize ``frame`` until convergence or ``steps`` is reached.

        Args:
            frame: ``molrs.Frame`` to optimize.
            fmax: Convergence threshold on the maximum force component.
            steps: Maximum number of steps.
            inplace: If True, mutate ``frame``; if False, optimize a ``frame.copy()``.

        Returns:
            :class:`OptimizationResult` whose ``frame`` is the optimized frame.
        """
        working_frame = frame if inplace else frame.copy()

        energy = 0.0
        current_fmax = float("inf")
        nsteps = 0

        for _ in range(steps):
            energy, current_fmax = self.step(working_frame)
            nsteps += 1

            for callback, interval, kwargs in self._callbacks:
                if nsteps % interval == 0:
                    callback(self, working_frame, **kwargs)

            if current_fmax < fmax:
                return OptimizationResult(
                    frame=working_frame,
                    energy=energy,
                    fmax=current_fmax,
                    nsteps=nsteps,
                    converged=True,
                    reason=f"Converged: fmax={current_fmax:.6f} < {fmax}",
                )

        return OptimizationResult(
            frame=working_frame,
            energy=energy,
            fmax=current_fmax,
            nsteps=nsteps,
            converged=False,
            reason=f"Max steps reached: {steps}",
        )

    def attach(self, func: Callable, interval: int = 1, **kwargs: Any) -> None:
        """Attach a callback ``func(optimizer, frame, **kwargs)`` called every ``interval`` steps."""
        self._callbacks.append((func, interval, kwargs))
