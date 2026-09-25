"""Intermediate representation (IR) dataclasses for MolTemplate parsing.

The parser produces a tree of these nodes; the builder walks the tree to
materialise ``ForceField`` and ``Atomistic`` objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic

# Argument counts of the transform forms the builder can apply:
# ``move(dx, dy, dz)``, ``rot(deg, ax, ay, az[, cx, cy, cz])``, ``scale(s)``.
_SUPPORTED_ARITIES: dict[str, tuple[int, ...]] = {
    "move": (3,),
    "rot": (4, 7),
    "scale": (1,),
}


@dataclass
class Transform:
    """A chained transform call on a ``new`` instance (``.move(...)``, etc.).

    The parser records any ``op`` it reads (``rotvv``, ``quat``, ...); only
    ``move(dx,dy,dz)``, ``rot(deg,ax,ay,az[,cx,cy,cz])`` and ``scale(s)`` can
    be applied or repeated — any other form raises instead of being dropped.
    """

    op: str  # "move" | "rot" | "rotvv" | "scale" | "quat"
    args: list[float] = field(default_factory=list)

    def repeated(self, times: int) -> Transform:
        """Return this transform composed with itself ``times`` times.

        Copy ``k`` of an array ``[N].op(...)`` carries ``repeated(k)``: a move
        by ``k * d``, a rotation by ``k * angle`` about the same axis and
        centre, a scale by ``s ** k``.

        Args:
            times: Number of applications (``0`` is the identity).

        Returns:
            A new transform equivalent to ``times`` successive applications.

        Raises:
            ValueError: If the transform form is not supported.
        """
        self._require_supported()
        if self.op == "move":
            return Transform(self.op, [times * a for a in self.args])
        if self.op == "rot":
            return Transform(self.op, [times * self.args[0], *self.args[1:]])
        return Transform(self.op, [self.args[0] ** times])

    def apply(self, mol: Atomistic) -> None:
        """Apply this transform to ``mol`` in place.

        Angles are in degrees (moltemplate convention); the rotation centre
        defaults to the origin, and ``scale`` is uniform about the origin.

        Args:
            mol: Structure whose coordinates are transformed.

        Raises:
            ValueError: If the transform form is not supported.
        """
        self._require_supported()
        a = self.args
        if self.op == "move":
            mol.translate([a[0], a[1], a[2]])
        elif self.op == "rot":
            about = [a[4], a[5], a[6]] if len(a) == 7 else None
            mol.rotate([a[1], a[2], a[3]], math.radians(a[0]), about)
        else:
            mol.scale([a[0], a[0], a[0]])

    def _require_supported(self) -> None:
        if len(self.args) not in _SUPPORTED_ARITIES.get(self.op, ()):
            raise ValueError(
                f"unsupported moltemplate transform {self.op}({self.args}); "
                "supported: move(dx,dy,dz), rot(deg,ax,ay,az[,cx,cy,cz]), scale(s)"
            )


@dataclass
class ArrayDim:
    """One dimension of the ``new Cls [N].move(dx, dy, dz)`` array form.

    ``count`` is N (number of copies along this dimension); ``transform`` is
    applied repeatedly — copy ``k`` carries ``transform.repeated(k)`` and
    copy ``0`` is the untransformed template. If ``transform`` is ``None``
    the copies are placed at the same position (rare but legal per
    moltemplate).
    """

    count: int
    transform: Transform | None = None


@dataclass
class RandomChoice:
    """One entry inside ``new random([...], [...])``.

    ``class_name`` is the class to instantiate and ``transforms`` are the
    per-choice transforms written as ``.move(...)``/``.rot(...)``/``.scale(...)``
    chains attached directly to the class name inside the ``random`` list.
    """

    class_name: str
    transforms: list[Transform] = field(default_factory=list)


@dataclass
class NewStmt:
    """``inst = new ClassName[.move(...)...]`` — molecule instantiation.

    Supports post-class multi-dimensional arrays, e.g.::

        m = new Butane [12].move(0, 0, 5.2) [12].move(0, 5.2, 0) [6].move(10.4, 0, 0)

    which produces ``12 * 12 * 6 = 864`` copies on a regular grid.
    ``transforms`` is the per-instance transform chain applied *before*
    array expansion. ``arrays`` is the list of dimensions (empty when no
    ``[N]`` form was written).

    When ``class_name == "random"`` the statement is a weighted sampler
    of the form ``new random([Cls1, Cls2], [w1, w2])``. ``random_choices``
    holds the class list (with per-choice transforms) and
    ``random_weights`` their relative weights. Positive-integer weights
    that sum to the array grid size are treated as exact counts;
    otherwise the weights are normalised and used as probabilities.
    ``random_seed`` pins the PRNG for reproducibility when set.
    """

    instance_name: str
    class_name: str
    count: int = 1  # legacy single-count form: `new [N] Cls`
    transforms: list[Transform] = field(default_factory=list)
    arrays: list[ArrayDim] = field(default_factory=list)
    random_choices: list[RandomChoice] = field(default_factory=list)
    random_weights: list[float] = field(default_factory=list)
    random_seed: int | None = None


@dataclass
class WriteBlock:
    """``write("Section Name") { ... body lines ... }``."""

    section: str
    body_lines: list[str] = field(default_factory=list)


@dataclass
class WriteOnceBlock:
    """``write_once("Section Name") { ... }`` — identical to WriteBlock semantically."""

    section: str
    body_lines: list[str] = field(default_factory=list)


@dataclass
class ImportStmt:
    """``import "file.lt"``."""

    path: str


@dataclass
class ReplaceStmt:
    """``replace{ @atom:A @atom:B }`` — alias one atom/bond/… type name to another.

    Each ``pair`` is ``(old_raw, new_raw)`` where both retain their ``@atom:``
    / ``@bond:`` / ``@angle:`` / … prefix so downstream code can tell apart
    per-kind replace maps. Used heavily by ``oplsaa*.lt`` to decorate atom
    types with their bond / angle / dihedral / improper partners.
    """

    pairs: list[tuple[str, str]] = field(default_factory=list)


# Forward declaration for recursive ClassDef.statements
Statement = Union[
    "ClassDef", NewStmt, WriteBlock, WriteOnceBlock, ImportStmt, ReplaceStmt
]


@dataclass
class ClassDef:
    """``ClassName [inherits Base1, Base2] { ... statements ... }``."""

    name: str
    bases: list[str] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)


@dataclass
class Document:
    """Top-level parsed document — a sequence of statements."""

    statements: list[Statement] = field(default_factory=list)
