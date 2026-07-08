"""Force-field-free soft potential for geometry relaxation.

:class:`SoftPotential` is a coordinate-only :class:`~molpy.optimize.PotentialLike`
implementation — it needs neither a force field nor atom types, only the
coordinates and bond list carried by a :class:`molrs.Frame`. It exists to relax
the freshly formed, often over-stretched bonds produced by graph edits such as
crosslinking, where a fully parameterized force field is not yet available.

Two energy terms are combined:

* **Harmonic bonds** pull every bond toward a rest length ``r0`` (by default the
  sum of the two atoms' covalent radii), removing the long crosslink bonds a
  reaction leaves behind.
* **Soft-core repulsion** is a purely repulsive, finite-everywhere spring between
  close non-bonded pairs, so a relaxing bond cannot drag atoms into overlap.

Both terms are ordinary quadratics, so the energy is smooth and bounded and the
gradient is well behaved for a quasi-Newton optimizer such as
:class:`~molpy.optimize.LBFGS`.
"""

from __future__ import annotations

import numpy as np

import molrs

from molpy.core.element import Element

# Padding (Angstrom) so every atom sits strictly inside the synthesized
# non-periodic box used for the neighbor search.
_BOX_MARGIN = 1.0


class SoftPotential:
    """Coordinate-only harmonic-bond + soft-core-repulsion potential.

    Args:
        k_bond: Harmonic bond force constant (energy / Angstrom²).
        r0: Bond rest length (Angstrom). ``None`` (default) derives a per-bond
            rest length from the sum of the two atoms' covalent radii; a float
            applies the same rest length to every bond.
        k_rep: Soft-core repulsion force constant (energy / Angstrom²).
        rc: Repulsion cutoff (Angstrom); non-bonded pairs closer than ``rc``
            repel, pairs at or beyond ``rc`` contribute nothing.
        repulsion: Enable the soft-core repulsion term. When ``False`` (or when
            a neighbor query cannot be built for the frame) only the bond term
            is evaluated.

    Example:
        >>> from molpy.optimize import LBFGS, SoftPotential
        >>> result = LBFGS(SoftPotential()).run(frame)  # frame: molrs.Frame
    """

    def __init__(
        self,
        *,
        k_bond: float = 10.0,
        r0: float | None = None,
        k_rep: float = 1.0,
        rc: float = 2.0,
        repulsion: bool = True,
    ) -> None:
        self.k_bond = k_bond
        self.r0 = r0
        self.k_rep = k_rep
        self.rc = rc
        self.repulsion = repulsion

    # ===== PotentialLike API =====

    def calc_energy(self, frame: molrs.Frame) -> float:
        """Return the total soft-potential energy of *frame*."""
        energy, _ = self._evaluate(frame)
        return energy

    def calc_forces(self, frame: molrs.Frame) -> np.ndarray:
        """Return the per-atom forces on *frame* as an ``(N, 3)`` array."""
        _, forces = self._evaluate(frame)
        return forces

    # ===== Core evaluation =====

    def _evaluate(self, frame: molrs.Frame) -> tuple[float, np.ndarray]:
        """Compute ``(energy, forces)`` for the bond + repulsion terms."""
        coords = np.asarray(molrs.extract_coords(frame), dtype=float).reshape(-1, 3)
        forces = np.zeros_like(coords)
        energy = 0.0

        bond_i, bond_j = self._bond_indices(frame)
        if bond_i.size:
            rest = self._rest_lengths(frame, bond_i, bond_j)
            energy += self._accumulate_bonds(coords, bond_i, bond_j, rest, forces)

        if self.repulsion:
            energy += self._accumulate_repulsion(coords, bond_i, bond_j, forces)

        return energy, forces

    def _accumulate_bonds(
        self,
        coords: np.ndarray,
        bond_i: np.ndarray,
        bond_j: np.ndarray,
        rest: np.ndarray,
        forces: np.ndarray,
    ) -> float:
        """Harmonic-bond energy; accumulates forces into ``forces`` in place."""
        delta = coords[bond_i] - coords[bond_j]
        dist = np.linalg.norm(delta, axis=1)
        valid = dist > 1e-12
        energy = float(0.5 * self.k_bond * np.sum((dist[valid] - rest[valid]) ** 2))

        # Force on i: -k_bond * (r - r0) * (ri - rj) / r  (and its negative on j).
        scale = np.zeros_like(dist)
        scale[valid] = -self.k_bond * (dist[valid] - rest[valid]) / dist[valid]
        pair_force = scale[:, None] * delta
        np.add.at(forces, bond_i, pair_force)
        np.add.at(forces, bond_j, -pair_force)
        return energy

    def _accumulate_repulsion(
        self,
        coords: np.ndarray,
        bond_i: np.ndarray,
        bond_j: np.ndarray,
        forces: np.ndarray,
    ) -> float:
        """Soft-core repulsion energy over close non-bonded pairs.

        Accumulates forces into ``forces`` in place. Degrades gracefully to
        zero (bonds-only) when no neighbor query can be built for the frame.
        """
        pairs = self._nonbonded_pairs(coords, bond_i, bond_j)
        if pairs.size == 0:
            return 0.0

        pi, pj = pairs[:, 0], pairs[:, 1]
        delta = coords[pi] - coords[pj]
        dist = np.linalg.norm(delta, axis=1)
        active = (dist < self.rc) & (dist > 1e-12)
        if not np.any(active):
            return 0.0

        pi, pj = pi[active], pj[active]
        delta, dist = delta[active], dist[active]
        overlap = self.rc - dist
        energy = float(0.5 * self.k_rep * np.sum(overlap**2))

        # Force pushes the pair apart: +k_rep * (rc - r) * (ri - rj) / r on i.
        scale = self.k_rep * overlap / dist
        pair_force = scale[:, None] * delta
        np.add.at(forces, pi, pair_force)
        np.add.at(forces, pj, -pair_force)
        return energy

    # ===== Topology / geometry helpers =====

    @staticmethod
    def _bond_indices(frame: molrs.Frame) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(atomi, atomj)`` 0-based row indices of every bond."""
        if "bonds" not in frame:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        bonds = frame["bonds"]
        if bonds.nrows == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        return (
            np.asarray(bonds["atomi"], dtype=int),
            np.asarray(bonds["atomj"], dtype=int),
        )

    def _rest_lengths(
        self, frame: molrs.Frame, bond_i: np.ndarray, bond_j: np.ndarray
    ) -> np.ndarray:
        """Per-bond rest length: explicit ``r0`` or the covalent-radii sum."""
        if self.r0 is not None:
            return np.full(bond_i.shape, float(self.r0))
        radii = self._atom_radii(frame)
        return radii[bond_i] + radii[bond_j]

    @staticmethod
    def _atom_radii(frame: molrs.Frame) -> np.ndarray:
        """Covalent radius per atom via ``Element`` (from ``element``/``symbol``)."""
        atoms = frame["atoms"]
        if "element" in atoms:
            symbols = np.asarray(atoms["element"])
        elif "symbol" in atoms:
            symbols = np.asarray(atoms["symbol"])
        else:
            symbols = None
        n_atoms = molrs.extract_coords(frame).reshape(-1, 3).shape[0]
        unknown = Element(0).covalent  # "unknown" element fallback, still via Element
        if symbols is None:
            return np.full(n_atoms, unknown)
        return np.array([SoftPotential._covalent(str(s), unknown) for s in symbols])

    @staticmethod
    def _covalent(symbol: str, default: float) -> float:
        """Covalent radius for *symbol* via ``Element``; *default* if unresolved."""
        try:
            return Element(symbol).covalent
        except KeyError:
            return default

    def _nonbonded_pairs(
        self, coords: np.ndarray, bond_i: np.ndarray, bond_j: np.ndarray
    ) -> np.ndarray:
        """Close non-bonded ``(i, j)`` index pairs (``i < j``) within ``rc``.

        Uses a molrs :class:`~molrs.NeighborQuery` over a synthesized
        non-periodic box (the crosslinker's pattern). Returns an empty array if
        the structure carries no usable coordinates or the query cannot be built.
        """
        if coords.shape[0] < 2:
            return np.empty((0, 2), dtype=int)
        try:
            box = self._bounding_box(coords)
            nlist = molrs.NeighborQuery(box, coords, self.rc).query(coords)
            pairs = np.asarray(nlist.pairs(), dtype=int)
        except Exception:
            # Degrade to bonds-only if a neighbor query cannot be built.
            return np.empty((0, 2), dtype=int)

        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)

        # Keep each unordered pair once (the self-query yields both directions
        # and i==i self-pairs).
        upper = pairs[:, 0] < pairs[:, 1]
        pairs = pairs[upper]
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)

        bonded = {(int(min(i, j)), int(max(i, j))) for i, j in zip(bond_i, bond_j)}
        if bonded:
            keep = np.array(
                [(int(i), int(j)) not in bonded for i, j in pairs], dtype=bool
            )
            pairs = pairs[keep]
        return pairs

    def _bounding_box(self, coords: np.ndarray) -> molrs.Box:
        margin = self.rc + _BOX_MARGIN
        lo = coords.min(axis=0) - margin
        hi = coords.max(axis=0) + margin
        return molrs.Box.ortho(hi - lo, lo, np.array([False, False, False]))
