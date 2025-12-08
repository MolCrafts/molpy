from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..target import Target

if TYPE_CHECKING:
    import molpy as mp
    from ..constraint import Constraint


class Packer(ABC):
    """Base class for all packer implementations."""

    def __init__(self):
        self.targets: list[Target] = []

    def add_target(self, target: Target) -> None:
        """Add a target to the packer."""
        self.targets.append(target)

    def def_target(
        self,
        frame: "mp.Frame",
        number: int,
        constraint: "Constraint",
        is_fixed: bool = False,
        name: str = "",
    ) -> Target:
        """
        Define a target for packing.

        Args:
            frame: Frame containing the molecule structure
            number: Number of copies to pack
            constraint: Spatial constraint for packing
            is_fixed: Whether this target is fixed
            name: Optional name for this target

        Returns:
            Created Target object
        """
        target = Target(frame, number, constraint, is_fixed, name)
        self.add_target(target)
        return target

    @abstractmethod
    def pack(
        self,
        targets: list[Target] | None = None,
        max_steps: int = 1000,
        seed: int | None = None,
    ) -> "mp.Frame":
        """
        Pack molecules according to targets.

        Args:
            targets: List of packing targets. If None, uses stored targets.
            max_steps: Maximum optimization steps
            seed: Random seed for packing

        Returns:
            Packed Frame
        """
        ...

    def __call__(
        self,
        targets: list[Target] | None = None,
        max_steps: int = 1000,
        seed: int | None = None,
        **kwargs,
    ) -> "mp.Frame":
        """
        Call packer as a function. Delegates to pack() method.

        This allows packers to be called directly: packer(targets, max_steps=1000)
        """
        return self.pack(targets=targets, max_steps=max_steps, seed=seed)

    @property
    def n_points(self):
        return sum([t.n_points for t in self.targets])

    @property
    def points(self):
        if not self.targets:
            return np.empty((0, 3))

        target_points = [t.points for t in self.targets]
        # Filter out empty arrays
        non_empty_points = [p for p in target_points if p.size > 0]

        if not non_empty_points:
            return np.empty((0, 3))

        return np.concatenate(non_empty_points, axis=0)
