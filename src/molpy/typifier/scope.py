"""Pattern-derived receptive field for region typing.

``TypeScope`` is the **system-side** synthesis of molrs syntax facts
(``max_bond_depth``, ``ring_primitives``). molrs never names ``reach`` or
``TypeScope`` — iron law 2: the engine reports grammar; the force-field layer
decides whether a finite neighbourhood exists and how large it is.

Bounded pattern sets yield a frozen ``TypeScope(reach=…)``. Any untyped ring
primitive (``[R]``, ``[R2]``, ``[x2]``, …) raises :class:`UnboundedPatternSet`
so a :class:`~molpy.typifier.smarts.SmartsTypifier` can refuse construction with
a named primitive rather than silently fall back to whole-graph typing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Self

import molrs

# Ring-primitive kinds that require global SSSR / whole-graph context.
# Sized [rN] is local; the rest are not.
_UNBOUNDED_KINDS = frozenset({"membership", "ring_count", "ring_bond_count"})


class UnboundedPatternSet(Exception):
    """A SMARTS pattern set has no finite receptive field.

    Attributes:
        primitive: Human-readable name of the offending ring token
            (e.g. ``"membership"``, ``"ring_count(2)"``, ``"ring_bond_count(2)"``).
    """

    def __init__(self, primitive: str) -> None:
        self.primitive = primitive
        super().__init__(
            f"pattern set contains {primitive}; ring membership / ring count / "
            f"ring-bond count are global properties, so this set has no finite "
            f"receptive field for region typing"
        )


def _compile_pattern(item: str | molrs.SmartsPattern) -> molrs.SmartsPattern:
    if isinstance(item, molrs.SmartsPattern):
        return item
    try:
        return molrs.SmartsPattern(item)
    except Exception:
        # OPLS-style bare atom expressions often need brackets.
        return molrs.SmartsPattern(f"[{item}]")


def _format_primitive(kind: str, n: int | None) -> str:
    if n is None:
        return kind
    return f"{kind}({n})"


@dataclass(frozen=True, slots=True)
class TypeScope:
    """Finite receptive field derived from a SMARTS pattern set.

    ``TERM_REACH`` is dihedral arity minus two (4 − 2), not a magic radius: every
    atom that can appear as an endpoint of a bonded term decided inside the
    write-back ball must itself be typed, so the write-back radius is at least
    this large.

    Attributes:
        reach: Neighbourhood radius in bonds that decides one atom's type.
    """

    TERM_REACH: ClassVar[int] = 2
    reach: int

    def __post_init__(self) -> None:
        if self.reach < 1:
            raise ValueError(f"reach must be >= 1, got {self.reach}")

    @property
    def interior_reach(self) -> int:
        """Write-back radius: ``max(reach, TERM_REACH)``."""
        return max(self.reach, self.TERM_REACH)

    @property
    def extract_radius(self) -> int:
        """Extracted-ball radius: ``interior_reach + reach``."""
        return self.interior_reach + self.reach

    @classmethod
    def from_patterns(cls, patterns: Iterable[str | molrs.SmartsPattern]) -> Self:
        """Synthesise a scope from molrs syntax facts on each pattern.

        Raises:
            UnboundedPatternSet: if any pattern uses an untyped ring primitive.
            ValueError: if ``patterns`` is empty.
        """
        compiled = [_compile_pattern(p) for p in patterns]
        if not compiled:
            raise ValueError("TypeScope.from_patterns requires at least one pattern")

        max_depth = 0
        max_sized_contrib = 0
        for pat in compiled:
            max_depth = max(max_depth, int(pat.max_bond_depth))
            for kind, n in pat.ring_primitives:
                if kind in _UNBOUNDED_KINDS:
                    raise UnboundedPatternSet(_format_primitive(kind, n))
                if kind == "sized" and n is not None:
                    # ⌊N/2⌋+1 — sized ring membership is local to that ring size.
                    max_sized_contrib = max(max_sized_contrib, n // 2 + 1)

        reach = max(cls.TERM_REACH, max_depth, max_sized_contrib)
        return cls(reach=reach)
