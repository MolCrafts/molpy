"""
Molpack - High-level molecular packing interface.

This module provides the main entry point for molecular packing.
It allows selecting different packer backends (Packmol, nlopt, etc.).
"""

import random
from pathlib import Path
from typing import Any, Literal

from molpy import Frame

from .packer import get_packer
from .target import Target


class Molpack:
    """
    High-level molecular packing interface.

    This class provides a clean API for molecular packing with the ability
    to choose different packer backends.

    Usage:
        packer = Molpack(workdir=Path("packing"), packer="packmol")
        packer.add_target(frame, number=100, constraint=box_constraint)
        result = packer.optimize(max_steps=1000, seed=42)
    """

    def __init__(
        self,
        workdir: Path,
        packer: Literal["packmol", "nlopt"] = "packmol",
    ):
        """
        Initialize Molpack.

        Args:
            workdir: Working directory for packing operations
            packer: Packer backend to use ("packmol" or "nlopt")
        """
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        self.workdir = workdir
        self.targets = []
        self.packer = get_packer(packer, workdir=workdir)

    def add_target(self, frame: Frame, number: int, constraint: Any) -> Target:
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

    def optimize(self, max_steps: int = 1000, seed: int | None = None) -> Frame:
        """
        Run packing optimization.

        Args:
            max_steps: Maximum optimization steps
            seed: Random seed. If None, uses random seed.

        Returns:
            Packed Frame
        """
        if seed is None:
            seed = random.randint(1, 10000)
        # Use __call__ method instead of legacy pack() method
        return self.packer(self.targets, max_steps=max_steps, seed=seed)
