"""Space-group symmetry — native, zero-dependency expansion for the crystal builder.

A crystal structure is usually published as an *asymmetric unit* (a few basis
sites) plus a space group: the full unit cell is recovered by applying every
symmetry operator to each site. This module supplies that machinery with nothing
but :mod:`numpy` and the standard library — no spglib, ASE, or pymatgen.

The primitives are deliberately small:

- :func:`parse_triplet` turns a Jones-faithful coordinate triplet
  (``"-y+1/2, x, z+1/4"`` — the form CIFs use in ``_symmetry_equiv_pos_as_xyz``)
  into an affine operator ``(R, t)`` with integer/half-integer entries kept exact
  via :class:`fractions.Fraction`.
- :class:`SpaceGroup` holds a list of such operators. Build it from an explicit
  operator list (:meth:`SpaceGroup.from_triplets`, e.g. pasted straight from a
  CIF) or from a handful of generators closed into the full group
  (:meth:`SpaceGroup.from_generators`).
- :meth:`SpaceGroup.equivalent_positions` expands one fractional site into all
  symmetry images, de-duplicated under the periodic boundary.

:meth:`molpy.builder.crystal.Lattice.from_spacegroup` wires this into the crystal
builder so a published structure becomes a tiling-ready :class:`~molpy.builder.crystal.Lattice`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["SpaceGroup", "parse_triplet"]


def _parse_component(comp: str) -> tuple[list[float], float]:
    """Parse one component (e.g. ``"1/4-y"``) into ``(row, translation)``.

    ``row`` is the ``[cx, cy, cz]`` coefficients of ``x, y, z``; ``translation``
    is the constant term reduced into ``[0, 1)``.
    """
    comp = comp.strip().replace(" ", "")
    if not comp:
        raise ValueError("empty coordinate component")
    # Make every term sign-prefixed so a simple split on '+' isolates them.
    terms = [t for t in comp.replace("-", "+-").split("+") if t]
    row = [0.0, 0.0, 0.0]
    trans = Fraction(0)
    for term in terms:
        sign = 1
        if term[0] == "-":
            sign, term = -1, term[1:]
        if term and term[-1] in "xyz":
            coeff = term[:-1]
            value = Fraction(coeff) if coeff not in ("", "+") else Fraction(1)
            row["xyz".index(term[-1])] += float(sign * value)
        else:
            trans += sign * Fraction(term)
    return row, float(trans % 1)


def parse_triplet(triplet: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a coordinate triplet into an affine operator ``(R, t)``.

    Args:
        triplet: A Jones-faithful symmetry string with three comma-separated
            components, e.g. ``"x,y,z"`` or ``"-y+1/2, x+1/2, z+1/2"`` — exactly
            the form found in a CIF ``_symmetry_equiv_pos_as_xyz`` loop.

    Returns:
        ``(R, t)`` where ``R`` is the ``(3, 3)`` rotation/reflection matrix and
        ``t`` the ``(3,)`` translation, with each ``t`` component in ``[0, 1)``.

    Raises:
        ValueError: If ``triplet`` does not have exactly three components.
    """
    comps = triplet.split(",")
    if len(comps) != 3:
        raise ValueError(f"expected 3 components, got {len(comps)}: {triplet!r}")
    rows, trans = zip(*(_parse_component(c) for c in comps))
    return np.array(rows, dtype=float), np.array(trans, dtype=float)


@dataclass(frozen=True)
class SpaceGroup:
    """A space group as its list of affine symmetry operators.

    Attributes:
        operators: Tuple of ``(R, t)`` pairs (``R`` is ``(3, 3)``, ``t`` is
            ``(3,)``). The identity is included; order equals the group order.
    """

    operators: tuple[tuple[np.ndarray, np.ndarray], ...]

    @property
    def order(self) -> int:
        """Number of symmetry operators in the group."""
        return len(self.operators)

    @classmethod
    def from_triplets(cls, triplets: list[str]) -> SpaceGroup:
        """Build from an explicit, complete operator list.

        Use this with the full ``_symmetry_equiv_pos_as_xyz`` loop copied from a
        CIF — every operator is taken verbatim, nothing is generated.
        """
        return cls(tuple(parse_triplet(t) for t in triplets))

    @classmethod
    def from_generators(cls, generators: list[str], *, max_order: int = 1536) -> SpaceGroup:
        """Build by closing a set of generator triplets into the full group.

        Repeatedly composes operators until the set is closed under
        multiplication (translations reduced mod 1). ``max_order`` guards against
        a non-crystallographic generator set that would never close.
        """
        ops = [parse_triplet(t) for t in generators]
        if not any(np.allclose(R, np.eye(3)) and not t.any() for R, t in ops):
            ops.insert(0, (np.eye(3), np.zeros(3)))
        seen = {_op_key(R, t) for R, t in ops}
        i = 0
        while i < len(ops):
            Ri, ti = ops[i]
            for Rj, tj in list(ops):
                R = Rj @ Ri
                t = (Rj @ ti + tj) % 1.0
                key = _op_key(R, t)
                if key not in seen:
                    seen.add(key)
                    ops.append((R, t))
                    if len(ops) > max_order:
                        raise ValueError("generator set did not close (>max_order ops)")
            i += 1
        return cls(tuple(ops))

    def equivalent_positions(self, frac: ArrayLike, *, symprec: float = 1e-5) -> np.ndarray:
        """All symmetry images of fractional site ``frac``, de-duplicated in ``[0, 1)``.

        Args:
            frac: A single ``(3,)`` fractional coordinate (the asymmetric-unit site).
            symprec: Distance below which two images (under the periodic boundary)
                are treated as the same atom — collapses sites that sit on a
                special position.

        Returns:
            ``(M, 3)`` array of unique fractional coordinates, ``M`` the site
            multiplicity.
        """
        frac = np.asarray(frac, dtype=float)
        out: list[np.ndarray] = []
        for R, t in self.operators:
            p = (R @ frac + t) % 1.0
            if not any(_same_site(p, q, symprec) for q in out):
                out.append(p)
        return np.array(out, dtype=float)


def _op_key(R: np.ndarray, t: np.ndarray) -> tuple:
    """Hashable key for an operator (translations reduced mod 1, rounded)."""
    return (tuple(np.round(R, 6).ravel()), tuple(np.round(t % 1.0, 6)))


def _same_site(p: np.ndarray, q: np.ndarray, symprec: float) -> bool:
    """True if fractional points ``p`` and ``q`` coincide under the periodic boundary."""
    d = p - q
    d -= np.round(d)  # minimum image in fractional space
    return bool(np.all(np.abs(d) < symprec))
