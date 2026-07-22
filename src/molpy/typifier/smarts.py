"""SMARTS pattern-set typifier base with a pattern-derived :class:`TypeScope`.

A :class:`SmartsTypifier` owns the patterns that decide types and refuses to
construct when those patterns have no finite receptive field (untyped ring
predicates). Subclasses implement :meth:`match`; this base only owns
``patterns`` + ``scope`` and a safe no-annotation default match for scope-only
tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import molrs

from molpy.core.atomistic import Atomistic
from molpy.typifier.base import Match, Typifier
from molpy.typifier.scope import TypeScope, UnboundedPatternSet, _compile_pattern

if TYPE_CHECKING:
    pass


class LocalTypifier(Typifier[Atomistic]):
    """Typifier that exposes a finite :class:`TypeScope` for region typing."""

    @property
    def scope(self) -> TypeScope:
        raise NotImplementedError


class SmartsTypifier(LocalTypifier):
    """Typifier whose receptive field is derived from a SMARTS pattern set.

    Args:
        patterns: SMARTS strings or compiled :class:`molrs.SmartsPattern`
            instances. At least one pattern is required.

    Raises:
        TypeError: if the pattern set is unbounded (message names the primitive).
        ValueError: if ``patterns`` is empty.
    """

    def __init__(self, patterns: Iterable[str | molrs.SmartsPattern]) -> None:
        compiled: list[molrs.SmartsPattern] = [_compile_pattern(p) for p in patterns]
        if not compiled:
            raise ValueError("SmartsTypifier requires at least one pattern")
        try:
            self._scope = TypeScope.from_patterns(compiled)
        except UnboundedPatternSet as exc:
            raise TypeError(
                f"pattern set contains {exc.primitive}; ring membership is a global "
                f"property, so this typifier has no finite receptive field and cannot "
                f"be used for region typing. Replace [R] with a sized [r6]."
            ) from exc
        self._patterns: tuple[molrs.SmartsPattern, ...] = tuple(compiled)

    @property
    def scope(self) -> TypeScope:
        """Finite receptive field synthesised from the pattern set."""
        return self._scope

    @property
    def patterns(self) -> Sequence[molrs.SmartsPattern]:
        """Compiled SMARTS patterns owned by this typifier."""
        return self._patterns

    def match(self, graph: Atomistic) -> Match:
        """Default: no annotations — subclasses override with real matching.

        Scope-bearing construction is the contract of this base class; atom
        typing remains force-field specific.
        """
        nodes = tuple({} for _ in graph.nodes)
        return Match(nodes=nodes)
