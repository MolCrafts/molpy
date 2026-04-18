"""Intermediate representation (IR) dataclasses for MolTemplate parsing.

The parser produces a tree of these nodes; the builder walks the tree to
materialise ``ForceField`` and ``Atomistic`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Transform:
    """A chained transform call on a ``new`` instance (``.move(...)``, etc.)."""

    op: str  # "move" | "rot" | "rotvv" | "scale" | "quat"
    args: list[float] = field(default_factory=list)


@dataclass
class ArrayDim:
    """One dimension of the ``new Cls [N].move(dx, dy, dz)`` array form.

    ``count`` is N (number of copies along this dimension); ``transform`` is
    applied repeatedly — copy ``k`` is shifted by ``k * args`` of the
    transform. If ``transform`` is ``None`` the copies are placed at the
    same position (rare but legal per moltemplate).
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
