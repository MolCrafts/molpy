"""What a typifier is: ``MolGraph -> MolGraph``.

One pipeline, one hook. Every typifier copies the graph, matches it, and writes
the annotations back — and the only step that differs between an OPLS typifier, an
antechamber typifier and a coarse-grained typifier is the *match*. So
:meth:`Typifier.typify` is concrete and shared, and :meth:`Typifier.match` is the
single abstract method.

A typifier does **not** ask whether the graph it was handed is a fragment. Whether
a graph was cut out of a larger one is a fact about its provenance, not something
readable off its valences: a radical is a perfectly good molecule, and a
connectivity graph with no bond orders looks under-coordinated everywhere. Guessing
it would be guessing an identity. The party that *cut* the graph knows, and it is
the one that completes it — see
:meth:`~molpy.typifier.region.RegionTypes.of`, which caps every region it types
because every region is, by construction, a cut.

The pipeline is generic over the graph: an :class:`~molpy.core.atomistic.Atomistic`
and a :class:`~molpy.core.cg.CoarseGrain` are both ``molrs.Graph`` leaves, and a
concrete typifier specialises ``G`` to the one it types. Nothing here knows what
a bond, an angle or a dihedral is — that is a fact about a *force field*, and it
lives in :mod:`molpy.typifier.forcefield`.

Typifiers are named after the force field or the tool that decides the types:
:class:`~molpy.typifier.clp.ClpTypifier`,
:class:`~molpy.typifier.ambertools.AmberToolsTypifier`, ``OPLSAATypifier``,
``MMFFTypifier``. "Assign parameters to a graph whose types are already known" is
not a typifier — it is the second half of one, and it is
:class:`~molpy.typifier.forcefield.ForceFieldParams`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import molrs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from molpy.core.entity import Link

#: A value a typifier may write onto a graph element. molrs Block columns are
#: scalar-typed, so an annotation value is always one of these.
type Annotation = str | int | float | bool


@dataclass(frozen=True)
class Match:
    """The annotations matching a graph produced, ready to be written back.

    ``nodes`` is positional against ``graph.nodes``; ``links`` maps each link
    class to a tuple positional against ``graph.links.bucket(cls)``. A match taken
    from a valence-completed graph may be written onto the graph it completed:
    the completion keeps the original elements first and appends the caps, so the
    write is a prefix copy and the caps fall off the end.

    An empty mapping in either sequence means "this element got nothing", which
    is how an unparameterised term stays untyped instead of being stamped with
    ``None``.
    """

    nodes: tuple[Mapping[str, Annotation], ...]
    links: Mapping[type[Link], tuple[Mapping[str, Annotation], ...]] = field(
        default_factory=dict
    )

    def write_onto(self, graph: Any) -> None:
        """Write this match onto ``graph``, whose elements are a prefix of the
        matched graph's.

        Raises:
            ValueError: if the match is shorter than the graph it is written
                onto — the two came from different graphs, and a silent
                truncation would leave elements untyped for no stated reason.
        """
        nodes = list(graph.nodes)
        self._require_covers("nodes", len(self.nodes), len(nodes))
        for node, annotation in zip(nodes, self.nodes, strict=False):
            node.update(**annotation)

        for link_cls, decided in self.links.items():
            links = list(graph.links.bucket(link_cls))
            self._require_covers(link_cls.__name__, len(decided), len(links))
            for link, annotation in zip(links, decided, strict=False):
                link.update(**annotation)

    @staticmethod
    def _require_covers(what: str, matched: int, target: int) -> None:
        if matched < target:
            raise ValueError(
                f"match covers {matched} {what} but the graph has {target}; "
                f"the match was produced from a different graph"
            )


class Typifier[G: molrs.Graph](molrs.Typifier, ABC):
    """Assign force-field types and parameters to a molecular graph.

    Subclasses implement :meth:`match` and nothing else. Specialise ``G`` to the
    graph kind the typifier understands.
    """

    def typify(self, graph: G) -> G:
        """Return a new graph carrying types and parameters; ``graph`` is untouched.

        ``graph`` is taken at face value: whatever it is, that is the molecule
        being typed. A caller holding a *fragment* completes its valences first —
        a matcher shown a raw slice sees radicals where the cut fell, and types
        the interior against them.
        """
        typed = graph.copy()
        self.match(graph).write_onto(typed)
        return typed

    @abstractmethod
    def match(self, graph: G) -> Match:
        """Decide what every element of ``graph`` should be annotated with.

        ``graph`` arrives valence-completed and is a private copy, so an
        implementation may write intermediate results onto it — writing the node
        types it just decided, for instance, so that the bonded terms can be
        matched against them.
        """
