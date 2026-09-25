"""Replicate a strand into a multi-molecule world for statistical assembly.

Packing production boxes belongs to the external molpack package
(``molcrafts-molpack``). This class only does what crosslinking demos need:
copy one strand onto a grid, give each copy a 1-based ``mol_id``, and return
one :class:`~molpy.core.atomistic.Atomistic` that a proximity selector can edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.core import fields

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic


class Replicas:
    """Copies of one strand arranged for a melt or gel precursor.

    Example::

        melt = Replicas(strand).grid(3, spacing=9.5, jitter=1.0, seed=7)
        gel = GraphAssembler(xlink).apply(
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
        """Return ``n³`` copies on a cubic lattice.

        Each copy gets ``mol_id`` ``1 .. n³`` (``fields.MOL_ID`` is 1-based).
        The ``mol_id`` labels molecules for output only: a proximity
        selector's ``exclude_same_molecule=True`` decides "same molecule" from
        bond connectivity, so it forbids intra-chain pairs because the copies
        are not bonded to each other, not because of ``mol_id``. Optional rigid
        rotation and position jitter break grid artifacts.

        Args:
            n: Copies per lattice edge; ``n³`` copies in total.
            spacing: Lattice spacing (Å).
            jitter: Half-width (Å) of the uniform random offset added to each
                lattice origin along x, y and z; ``0.0`` disables it.
            seed: Seed for the random rotations and jitter.
            rotate: Before translating, rotate each copy by a uniformly random
                angle in ``[0, 2π)`` rad about a random axis through the
                origin (the axis direction is uniform on the sphere).

        Returns:
            One world holding every copy.

        Raises:
            ValueError: if ``n < 1`` or ``spacing <= 0``.
        """
        if n < 1:
            raise ValueError(f"grid size n must be >= 1, got {n}")
        if spacing <= 0:
            raise ValueError(f"spacing must be positive (Å), got {spacing}")

        rng = np.random.default_rng(seed)
        world = Atomistic()
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
                    copy.translate(list(origin))
                    for atom in copy.atoms:
                        atom[fields.MOL_ID] = mol_id
                    world.merge(copy)
                    mol_id += 1
        return world

    def times(self, count: int, *, spacing: float = 10.0) -> Atomistic:
        """Return ``count`` copies along x with the given spacing (Å).

        Simpler than :meth:`grid` when you only need a few chains for a demo.
        Copy ``k`` (0-based) is translated by ``k * spacing`` along x and gets
        ``mol_id`` ``k + 1``.

        Args:
            count: Number of copies.
            spacing: Distance (Å) between consecutive copies along x.

        Returns:
            One world holding every copy.

        Raises:
            ValueError: if ``count < 1``.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        world = Atomistic()
        for index in range(count):
            copy = self._strand.copy()
            copy.translate([index * spacing, 0.0, 0.0])
            for atom in copy.atoms:
                atom[fields.MOL_ID] = index + 1
            world.merge(copy)
        return world
