"""Space-group symmetry — native, zero-dependency expansion for the crystal builder.

A crystal structure is usually published as an *asymmetric unit* (a few basis
sites) plus a space group: the full unit cell is recovered by applying every
symmetry operator to each site. This module supplies that machinery with nothing
but :mod:`numpy` and the standard library — no spglib, ASE, or pymatgen.

The primitives are deliberately small:

- :func:`parse_triplet` turns a Jones-faithful coordinate triplet
  (``"-y+1/2, x, z+1/4"`` — the form CIFs use in ``_symmetry_equiv_pos_as_xyz``)
  into an affine operator ``(R, t)`` of ``float`` arrays. Each fraction is read
  exactly (:class:`fractions.Fraction`) and reduced into ``[0, 1)`` before it
  becomes a float.
- :class:`SpaceGroup` holds a list of such operators. Build it from an explicit
  operator list (:meth:`SpaceGroup.from_triplets`, e.g. the ``symops`` that
  :func:`molpy.io.read_cif` keeps in ``frame.meta``) or from a handful of
  generators closed into the full group (:meth:`SpaceGroup.from_generators`).
- :meth:`SpaceGroup.equivalent_positions` expands one fractional site into all
  symmetry images, de-duplicated under the periodic boundary.

Operators are composed in floating point, so a translation such as
``1/3 + 2/3`` comes out as ``0.9999999999999999``. Every translation is wrapped
into ``[0, 1)`` with a tolerance (:data:`_WRAP_TOL`), so it reads as the
lattice translation ``0``, and operators are compared on keys rounded to
:data:`_KEY_DECIMALS` places.

:meth:`molpy.builder.Lattice.from_cif` wires this into the crystal builder, so
a CIF read by :func:`molpy.io.read_cif` becomes a tiling-ready
:class:`~molpy.builder.Lattice`; :meth:`~molpy.builder.Lattice.from_spacegroup`
takes the cell, sites and group directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["SpaceGroup", "parse_triplet"]

#: A wrapped coordinate this close below 1 is the lattice translation 0.
_WRAP_TOL = 1e-9
#: Decimal places an operator is rounded to when two operators are compared.
_KEY_DECIMALS = 6


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
    def from_triplets(cls, triplets: Iterable[str]) -> SpaceGroup:
        """Build from an explicit, complete operator list.

        Use this with a CIF's full operator loop, as :func:`molpy.io.read_cif`
        keeps it in ``frame.meta["symops"]``. Every operator is taken verbatim;
        nothing is generated. :meth:`molpy.builder.Lattice.from_cif` does this
        for a CIF frame.
        """
        return cls(tuple(parse_triplet(t) for t in triplets))

    @classmethod
    def from_generators(
        cls, generators: list[str], *, max_order: int = 1536
    ) -> SpaceGroup:
        """Build by closing a set of generator triplets into the full group.

        Repeatedly composes operators until the set is closed under
        multiplication (translations wrapped into ``[0, 1)``). ``max_order``
        guards against a non-crystallographic generator set that would never
        close.
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
                t = _wrap(Rj @ ti + tj)
                key = _op_key(R, t)
                if key not in seen:
                    seen.add(key)
                    ops.append((R, t))
                    if len(ops) > max_order:
                        raise ValueError("generator set did not close (>max_order ops)")
            i += 1
        return cls(tuple(ops))

    def equivalent_positions(
        self, frac: ArrayLike, *, symprec: float = 1e-5
    ) -> np.ndarray:
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
            p = _wrap(R @ frac + t, tol=symprec)
            if not any(_same_site(p, q, symprec) for q in out):
                out.append(p)
        return np.array(out, dtype=float)


def _wrap(x: np.ndarray, tol: float = _WRAP_TOL) -> np.ndarray:
    """Reduce ``x`` into ``[0, 1)``; a value within ``tol`` below 1 becomes 0."""
    w = np.mod(x, 1.0)
    return np.where(w >= 1.0 - tol, 0.0, w)


def _op_key(R: np.ndarray, t: np.ndarray) -> tuple:
    """Hashable key for an operator, equal for operators that differ by rounding.

    The translation is rounded *before* it is reduced mod 1, so
    ``0.9999999999999999`` and ``0.0`` both key as ``0.0``.
    """
    t_key = np.mod(np.round(t, _KEY_DECIMALS), 1.0) + 0.0  # + 0.0 drops -0.0
    return (tuple(np.round(R, _KEY_DECIMALS).ravel()), tuple(t_key))


def _same_site(p: np.ndarray, q: np.ndarray, symprec: float) -> bool:
    """True if fractional points ``p`` and ``q`` coincide under the periodic boundary."""
    d = p - q
    d -= np.round(d)  # minimum image in fractional space
    return bool(np.all(np.abs(d) < symprec))
