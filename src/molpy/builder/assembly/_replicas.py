"""Replicate a strand into a multi-molecule world for statistical assembly.

Packing production boxes belongs in :mod:`molpy.pack`. This class only does what
crosslinking demos need: copy one strand onto a grid, give each copy a
``mol_id``, and return one :class:`~molpy.core.atomistic.Atomistic` that a
proximity selector can edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import molpy as mp
from molpy.core import fields

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic


class Replicas:
    """Copies of one strand arranged for a melt or gel precursor.

    Example::

        melt = Replicas(strand).grid(3, spacing=9.5, jitter=1.0, seed=7)
        gel = GraphAssembler(xlink).assemble(
            melt, ExhaustiveSelector(cutoff=6.5, exclude_same_molecule=True)
        )
    """

    def __init__(self, strand: Atomistic) -> None:
        self._strand = strand

    @property
    def strand(self) -> Atomistic:
        return self._strand

    def grid(
        self,
        n: int,
        spacing: float,
        *,
        jitter: float = 0.0,
        seed: int = 0,
        rotate: bool = True,
    ) -> Atomistic:
        """Return ``n³`` copies on a cubic lattice with spacing in Å.

        Each copy gets ``mol_id`` ``0 .. n³-1`` so
        ``exclude_same_molecule=True`` on a proximity selector forbids
        intra-chain pairs. Optional rigid rotation and position jitter break
        grid artifacts.

        Raises:
            ValueError: if ``n < 1`` or ``spacing <= 0``.
        """
        if n < 1:
            raise ValueError(f"grid size n must be >= 1, got {n}")
        if spacing <= 0:
            raise ValueError(f"spacing must be positive (Å), got {spacing}")

        rng = np.random.default_rng(seed)
        world = mp.Atomistic()
        mol_id = 1  # fields.MOL_ID is 1-indexed
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    copy = self._strand.copy()
                    if rotate:
                        axis = rng.normal(size=3)
                        norm = float(np.linalg.norm(axis))
                        if norm > 0:
                            copy.rotate(
                                list(axis / norm),
                                float(rng.uniform(0, 2 * np.pi)),
                            )
                    origin = np.array([i, j, k], dtype=float) * spacing
                    if jitter:
                        origin = origin + rng.uniform(-jitter, jitter, 3)
                    copy.move(list(origin), entity_type=mp.Atom)
                    for atom in copy.atoms:
                        atom[fields.MOL_ID] = mol_id
                    world.merge(copy)
                    mol_id += 1
        return world

    def times(self, count: int, *, spacing: float = 10.0) -> Atomistic:
        """Return ``count`` copies along x with the given spacing (Å).

        Simpler than :meth:`grid` when you only need a few chains for a demo.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        world = mp.Atomistic()
        for index in range(count):
            copy = self._strand.copy()
            copy.move([index * spacing, 0.0, 0.0], entity_type=mp.Atom)
            for atom in copy.atoms:
                atom[fields.MOL_ID] = index + 1
            world.merge(copy)
        return world
