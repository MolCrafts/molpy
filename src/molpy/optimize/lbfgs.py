"""L-BFGS geometry optimizer."""

from typing import TypeVar

import numpy as np

from .base import Optimizer

S = TypeVar("S")  # Generic structure type


class LBFGS(Optimizer[S]):
    """Limited-memory BFGS geometry optimizer.

    Implements the L-BFGS algorithm for quasi-Newton optimization with
    limited memory storage. Uses two-loop recursion to compute search
    directions efficiently.

    Args:
        potential: Potential with calc_energy/calc_forces methods
        maxstep: Maximum step size (as displacement norm)
        memory: Number of previous steps to store for Hessian approximation
        damping: Damping factor for step size
        entity_type: Type of entity to optimize

    Attributes:
        maxstep: Maximum allowed step size
        memory: LBFGS memory size
        damping: Step damping factor
        s_history: Position difference history
        y_history: Gradient difference history
        rho_history: Curvature history (1 / y·s)

    Example:
        >>> from molpy.core.atomistic import Atomistic
        >>> from molpy.potential.bond import Harmonic
        >>> from molpy.optimize import LBFGS
        >>>
        >>> struct = Atomistic()
        >>> # ... add atoms and bonds ...
        >>> potential = Harmonic(k=100.0, r0=1.5)
        >>> opt = LBFGS(potential, maxstep=0.04, memory=20)
        >>> result = opt.run(struct, fmax=0.01, steps=500)
    """

    def __init__(
        self,
        potential,
        *,
        maxstep: float = 0.04,
        memory: int = 20,
        damping: float = 1.0,
        entity_type=None,
    ) -> None:
        # Handle entity_type default
        if entity_type is None:
            from molpy.core.entity import Entity

            entity_type = Entity

        super().__init__(potential, entity_type=entity_type)
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping

        # LBFGS state (reset for each new structure/run)
        self._current_structure_id = None
        self.s_history: list[np.ndarray] = []  # position differences
        self.y_history: list[np.ndarray] = []  # gradient differences
        self.rho_history: list[float] = []  # 1 / (y · s)

        self._prev_positions: np.ndarray | None = None
        self._prev_gradient: np.ndarray | None = None

    def _reset_state(self, structure_id) -> None:
        """Reset LBFGS state for a new structure."""
        if self._current_structure_id != structure_id:
            self._current_structure_id = structure_id
            self.s_history = []
            self.y_history = []
            self.rho_history = []
            self._prev_positions = None
            self._prev_gradient = None

    def step(self, structure: S) -> tuple[float, float]:
        """Perform one L-BFGS optimization step.

         Args:
             structure: Structure to optimize (modified in-place)

        Returns:
             (energy, fmax) tuple where:
                 energy: potential energy after step
                 fmax: maximum force component after step
        """
        # Reset state if this is a new structure
        self._reset_state(id(structure))

        # Get current state
        positions = self.get_positions(structure)
        forces = self.get_forces(structure)
        gradient = -forces  # Forces are -∇E, we need ∇E for minimization
        energy = self.get_energy(structure)

        # Flatten for linear algebra
        x = positions.reshape(-1)
        g = gradient.reshape(-1)

        # Update LBFGS history if we have a previous step
        if self._prev_positions is not None:
            s = x - self._prev_positions
            y = g - self._prev_gradient

            sy = np.dot(s, y)
            if sy > 1e-10:  # Only update if curvature condition holds
                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(1.0 / sy)

                # Keep only last 'memory' updates
                if len(self.s_history) > self.memory:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                    self.rho_history.pop(0)

        # Compute search direction using two-loop recursion
        search_dir = self._lbfgs_direction(g)

        # Apply step-size control
        step_size = self._compute_step_size(search_dir)

        # Update positions
        new_x = x - step_size * search_dir
        new_positions = new_x.reshape(positions.shape)
        self.set_positions(structure, new_positions)

        # Store for next iteration
        self._prev_positions = x.copy()
        self._prev_gradient = g.copy()

        # Compute fmax (maximum force component)
        fmax = float(np.max(np.abs(forces)))

        return energy, fmax

    def _lbfgs_direction(self, g: np.ndarray) -> np.ndarray:
        """Compute LBFGS search direction via two-loop recursion.

        Args:
            g: Current gradient (flattened)

        Returns:
            Search direction (same shape as g)
        """
        if not self.s_history:
            # First iteration: use steepest descent
            return g

        q = g.copy()
        alphas = []

        # First loop (backward)
        for s, y, rho in zip(
            reversed(self.s_history),
            reversed(self.y_history),
            reversed(self.rho_history),
        ):
            alpha = rho * np.dot(s, q)
            q -= alpha * y
            alphas.append(alpha)

        alphas.reverse()

        # Initial Hessian approximation (γI)
        # Use scaling factor from most recent curvature
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
        z = gamma * q

        # Second loop (forward)
        for s, y, rho, alpha in zip(
            self.s_history, self.y_history, self.rho_history, alphas
        ):
            beta = rho * np.dot(y, z)
            z += s * (alpha - beta)

        return z

    def _compute_step_size(self, search_dir: np.ndarray) -> float:
        """Compute step size with maxstep constraint.

        Args:
            search_dir: Search direction (flattened)

        Returns:
            Step size (scalar)
        """
        # Limit step by maxstep
        dr_norm = np.linalg.norm(search_dir)
        if dr_norm > self.maxstep:
            return self.damping * self.maxstep / dr_norm
        return self.damping
