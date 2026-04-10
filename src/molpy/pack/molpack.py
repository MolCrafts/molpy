"""
Molpack - High-level molecular packing interface.

This module provides the main entry point for molecular packing using Packmol.
"""

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molpy.core.frame import Frame

from .packer import get_packer
from .target import Target

if TYPE_CHECKING:
    from .constraint import Constraint


class Molpack:
    """
    High-level molecular packing interface.

    This class provides a clean API for molecular packing using Packmol backend.

    Usage:
        packer = Molpack(workdir=Path("packing"))
        packer.add_target(frame, number=100, constraint=box_constraint)
        result = packer.optimize(max_steps=1000, seed=42)
    """

    def __init__(
        self,
        workdir: Path,
    ):
        """
        Initialize Molpack.

        Args:
            workdir: Working directory for packing operations
        """
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        self.workdir = workdir
        self.targets = []
        self.packer = get_packer(workdir=workdir)

    def add_target(self, frame: Frame, number: int, constraint: "Constraint") -> Target:
        """
        Add a packing target.

        Args:
            frame: Frame containing the molecule structure
            number: Number of copies to pack
            constraint: Spatial constraint for packing

        Returns:
            Target object
        """
        target = Target(frame, number, constraint)
        self.targets.append(target)
        self.packer.add_target(target)
        return target

    def optimize(
        self,
        max_steps: int = 1000,
        seed: int | None = None,
        pbc: np.ndarray | list[float] | None = None,
    ) -> Frame:
        """
        Run packing optimization.

        Args:
            max_steps: Maximum optimization steps
            seed: Random seed. If None, uses random seed.
            pbc: Periodic boundary conditions. 3 values ``[lx, ly, lz]`` for
                 a box with origin at zero, or 6 values
                 ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
                 Requires Packmol >= 20.15.0.

        Returns:
            Packed Frame
        """
        if seed is None:
            seed = random.randint(1, 10000)
        return self.packer(self.targets, max_steps=max_steps, seed=seed, pbc=pbc)
