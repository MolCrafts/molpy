"""Base classes for geometry optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

import numpy as np

from molpy.core.entity import Entity


# Protocol for potentials (duck typing)
class PotentialLike(Protocol):
    """Protocol for potential functions compatible with calc_energy_from_frame."""

    def calc_energy(self, *args: Any, **kwargs: Any) -> float: ...
    def calc_forces(self, *args: Any, **kwargs: Any) -> np.ndarray: ...


S = TypeVar("S")  # Generic structure type (StructLike)


@dataclass
class OptimizationResult(Generic[S]):
    """Result of a geometry optimization.

    Attributes:
        structure: Final optimized structure (same object if inplace=True)
        energy: Final potential energy
        fmax: Final maximum force component
        nsteps: Number of optimization steps taken
        converged: Whether convergence criteria were met
        reason: Human-readable termination reason
    """

    structure: S
    energy: float
    fmax: float
    nsteps: int
    converged: bool
    reason: str


class Optimizer(ABC, Generic[S]):
    """Base class for structure optimizers.

    Works with any StructLike (Struct, Atomistic, CoarseGrain, etc.) that:
    - Has `.entities` (TypeBucket) containing entities with "xyz" field
    - Has `.to_frame()` method to convert to Frame format

    The optimizer calls potential.calc_energy(frame) and potential.calc_forces(frame)
    directly - each potential is responsible for extracting what it needs from Frame.

    Args:
        potential: Potential with calc_energy/calc_forces methods
        entity_type: Type of entity to optimize (default: Entity for all)

    Example:
        >>> from molpy.optimize import LBFGS
        >>> from molpy.potential.bond import Harmonic
        >>>
        >>> potential = Harmonic(k=100.0, r0=1.5)
        >>> opt = LBFGS(potential, maxstep=0.04, memory=20)
        >>> result = opt.run(struct, fmax=0.01, steps=500)
    """

    def __init__(
        self,
        potential: PotentialLike,
        *,
        entity_type: type[Entity] = Entity,
    ) -> None:
        self.potential = potential
        self.entity_type = entity_type
        self._callbacks: list[tuple[Callable, int, dict]] = []

    # ===== Bridge methods: StructLike ↔ numpy arrays =====

    def get_positions(self, structure: S) -> np.ndarray:
        """Extract positions as (N, 3) array from entities.

        Args:
            structure: Structure to extract positions from

        Returns:
            (N, 3) numpy array of positions
        """
        entities = structure.entities.all()
        if not entities:
            return np.empty((0, 3), dtype=float)
        
        # Use x, y, z fields (never use xyz)
        x_list = entities["x"]
        y_list = entities["y"]
        z_list = entities["z"]
        positions = np.column_stack([x_list, y_list, z_list])
        
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        return positions

    def set_positions(self, structure: S, positions: np.ndarray) -> None:
        """Write positions back into structure entities.

        Args:
            structure: Structure to update
            positions: (N, 3) array of new positions
        """
        entities = structure.entities.all()
        positions = positions.reshape(-1, 3)
        for i, entity in enumerate(entities):
            pos = positions[i]
            # Update x, y, z fields (never use xyz)
            entity["x"] = float(pos[0])
            entity["y"] = float(pos[1])
            entity["z"] = float(pos[2])

    def get_energy_and_forces(self, structure: S) -> tuple[float, np.ndarray]:
        """Compute energy and forces via Frame interface.

        Converts structure to Frame and calls potential methods directly.
        Each potential is responsible for extracting what it needs from Frame.

        Args:
            structure: Structure to evaluate

        Returns:
            (energy, forces) where forces is (N, 3) array
        """
        # Convert structure to Frame
        frame = structure.to_frame()

        # Call potential methods directly - they handle Frame extraction
        energy = self.potential.calc_energy(frame)
        forces = self.potential.calc_forces(frame)

        return energy, forces

    def get_energy(self, structure: S) -> float:
        """Compute energy for structure.

        Args:
            structure: Structure to evaluate

        Returns:
            Potential energy
        """
        energy, _ = self.get_energy_and_forces(structure)
        return energy

    def get_forces(self, structure: S) -> np.ndarray:
        """Compute forces for structure as (N, 3) array.

        Args:
            structure: Structure to evaluate

        Returns:
            (N, 3) array of forces
        """
        _, forces = self.get_energy_and_forces(structure)
        return forces

    # ===== Abstract method for subclasses =====

    @abstractmethod
    def step(self, structure: S) -> tuple[float, float]:
        """Perform one optimization step.

        Args:
            structure: Structure to optimize (modified in-place)

        Returns:
            (energy, fmax) tuple where:
                energy: potential energy after step
                fmax: maximum force component after step
        """
        pass

    # ===== Public API =====

    def run(
        self,
        structure: S,
        fmax: float = 0.01,
        steps: int = 1000,
        *,
        inplace: bool = True,
    ) -> OptimizationResult[S]:
        """Run optimization until convergence or max steps.

        Args:
            structure: Structure to optimize
            fmax: Convergence threshold (max force component)
            steps: Maximum number of steps
            inplace: If True, modify structure in-place; if False, work on copy

        Returns:
            OptimizationResult with final state
        """
        # Make copy if needed
        if inplace:
            working_structure = structure
        else:
            from copy import deepcopy

            working_structure = deepcopy(structure)

        energy = 0.0
        current_fmax = float("inf")
        nsteps = 0

        for i in range(steps):
            energy, current_fmax = self.step(working_structure)
            nsteps += 1

            # Call callbacks
            for callback, interval, kwargs in self._callbacks:
                if nsteps % interval == 0:
                    callback(self, working_structure, **kwargs)

            # Check convergence
            if current_fmax < fmax:
                return OptimizationResult(
                    structure=working_structure,
                    energy=energy,
                    fmax=current_fmax,
                    nsteps=nsteps,
                    converged=True,
                    reason=f"Converged: fmax={current_fmax:.6f} < {fmax}",
                )

        # Max steps reached
        return OptimizationResult(
            structure=working_structure,
            energy=energy,
            fmax=current_fmax,
            nsteps=nsteps,
            converged=False,
            reason=f"Max steps reached: {steps}",
        )

    def attach(self, func: Callable, interval: int = 1, **kwargs: Any) -> None:
        """Attach a callback function.

        The callback will be called every `interval` steps with the optimizer
        instance and current structure as arguments.

        Args:
            func: Callback function(optimizer, structure, **kwargs)
            interval: Call every N steps
            **kwargs: Additional arguments for callback
        """
        self._callbacks.append((func, interval, kwargs))
