"""Adapter exposing a molrs ForceField as an optimizer ``potential``.

The optimizer base (:class:`molpy.optimize.base.Optimizer`) drives any object
that answers ``calc_energy(frame)`` / ``calc_forces(frame)``. molrs's native
:class:`molrs.Potentials` instead consumes flat coordinate vectors and must be
recompiled against the current frame each step (geometry changes while the
topology/types stay fixed). This thin adapter bridges the two: it compiles the
force field against the live frame and evaluates energy/forces from the frame's
coordinates.
"""

from __future__ import annotations

import numpy as np

import molrs


class ForceFieldPotential:
    """Evaluate a molrs :class:`~molrs.ForceField` from a :class:`~molrs.Frame`.

    Args:
        forcefield: A molrs ``ForceField`` whose styles/types are fully defined.
            The frame passed to :meth:`calc_energy` / :meth:`calc_forces` must
            carry ``type`` columns whose values match the force field's type
            names (e.g. a bond labelled ``"OW-HW"``).
    """

    def __init__(self, forcefield: molrs.ForceField) -> None:
        self.forcefield = forcefield

    def calc_energy(self, frame: molrs.Frame) -> float:
        """Return the total potential energy for *frame*."""
        pots = self.forcefield.to_potentials(frame)
        return float(pots.calc_energy(molrs.extract_coords(frame)))

    def calc_forces(self, frame: molrs.Frame) -> np.ndarray:
        """Return the per-atom forces for *frame* as an ``(N, 3)`` array."""
        pots = self.forcefield.to_potentials(frame)
        forces = np.asarray(pots.calc_forces(molrs.extract_coords(frame)), dtype=float)
        return forces.reshape(-1, 3)
