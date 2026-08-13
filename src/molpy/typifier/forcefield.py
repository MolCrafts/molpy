"""Turn node types into force-field parameters.

:class:`ForceFieldParams` is **not a typifier** — it decides nothing about what
an atom *is*. It is the second half of every force-field typifier: given a graph
whose nodes already carry a ``type``, it looks each pair and bonded term up in a
force field and annotates it. ``ClpTypifier`` and ``AmberToolsTypifier`` differ
only in how they obtain those node types; both hand the result to this class.

It is also the one place in molpy that knows a :class:`~molpy.core.atomistic.Bond`
is parameterised by a :class:`~molpy.core.forcefield.BondType`. That mapping
cannot be derived from arity — a dihedral and an improper both span four atoms —
so it is written down once, in :data:`_FF_TYPE_OF`. When a coarse-grained force
field grows a ``CGBondType``, one line is added there and nothing else changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molpy.core import fields
from molpy.core.atomistic import Angle, Bond, Dihedral, Improper
from molpy.core.forcefield import (
    AngleType,
    AtomType,
    BondType,
    DihedralType,
    ImproperType,
    PairType,
)
from molpy.typifier._matching import TypeClassIndex
from molpy.typifier.base import Match

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from molpy.core.entity import Entity, Link
    from molpy.core.forcefield import ForceField, Type
    from molpy.typifier.base import Annotation

#: Which force-field type parameterises which kind of link. Arity cannot decide
#: this: ``Dihedral`` and ``Improper`` both have four endpoints.
_FF_TYPE_OF: dict[type[Link], type[Type]] = {
    Bond: BondType,
    Angle: AngleType,
    Dihedral: DihedralType,
    Improper: ImproperType,
}

#: Endpoint attribute names of a force-field bonded type, in order. A type of
#: arity N carries exactly the first N of these.
_ENDPOINT_ATTRS = ("itom", "jtom", "ktom", "ltom")


class _TermMatcher:
    """Find the force-field type that best describes one bonded term.

    Matching resolves each endpoint's type to its class and compares against the
    force-field type's component names, in both directions. Among all matches the
    most specific wins; ties break toward the higher overlay layer, so a
    CL&P/CL&Pol term overrides the OPLS-AA term it shadows.
    """

    def __init__(
        self,
        forcefield: ForceField,
        ff_type: type[Type],
        index: TypeClassIndex,
        *,
        strict: bool,
    ) -> None:
        self._ff_type = ff_type
        self._index = index
        self._strict = strict
        self._table = [
            (self._pattern(candidate), candidate)
            for candidate in forcefield.get_types(ff_type)
        ]

    @property
    def parameterizes(self) -> bool:
        """Whether the force field declares any type of this kind at all."""
        return bool(self._table)

    @staticmethod
    def _pattern(ff_type: Type) -> tuple[str, ...]:
        """The component names of a force-field type, in endpoint order."""
        return tuple(
            getattr(ff_type, attr).name
            for attr in _ENDPOINT_ATTRS
            if hasattr(ff_type, attr)
        )

    def annotate(self, link: Link) -> Mapping[str, Annotation]:
        """The type name and parameters ``link`` should carry, or ``{}``.

        An empty mapping means "undecided": either an endpoint is untyped, or no
        force-field type matches and ``strict`` is off. Nothing is ever written
        as ``None`` — an undecided term keeps whatever it already had.

        Raises:
            ValueError: if no force-field type matches and ``strict`` is on.
        """
        atom_types = [endpoint.get(fields.TYPE) for endpoint in link.endpoints]
        if any(atom_type is None for atom_type in atom_types):
            return {}
        resolved = [str(atom_type) for atom_type in atom_types]

        best = self._best(resolved)
        if best is not None:
            # A force-field type's parameters describe the *interaction*. Its
            # own bookkeeping — the `id` a reader gave it to key its tables —
            # is not a parameter, and stamping it here would overwrite the
            # link's identity with a force-field type index.
            params = {
                key: value
                for key, value in best.params.kwargs.items()
                if key != fields.ID
            }
            return {fields.TYPE: best.name, **params}
        if not self._strict:
            return {}
        raise ValueError(
            f"No {self._ff_type.__name__} found for atom types: {' - '.join(resolved)}"
        )

    def _best(self, atom_types: Sequence[str]) -> Type | None:
        best_key: tuple[int, int] | None = None
        best: Type | None = None
        for pattern, ff_type in self._table:
            score = self._index.score(pattern, atom_types)
            if score is None:
                continue
            key = (score, self._index.layer_of(pattern))
            if best_key is None or key > best_key:
                best_key, best = key, ff_type
        return best


class _PairMatcher:
    """Look a node's nonbonded parameters up by its type.

    Unlike a bonded term, these are keyed by the type alone: no pattern, no
    direction, no specificity contest. A lookup, not a match.

    molrs OPLS keeps charge/mass on the AtomType and LJ (ε/σ) on the PairType;
    this matcher merges both so annotate produces a complete nonbonded view.
    """

    # AtomType params that are nonbonded / identity metadata for the node.
    _ATOM_KEYS = frozenset({"charge", "mass", "element"})

    def __init__(
        self,
        forcefield: ForceField,
        *,
        strict: bool,
        atom_types: Sequence[AtomType] | None = None,
    ) -> None:
        self._strict = strict
        self._table = {
            pair_type.name: pair_type for pair_type in forcefield.get_types(PairType)
        }
        # Real atom types only (skip class-wildcard placeholders type_="*").
        types = atom_types if atom_types is not None else forcefield.get_types(AtomType)
        self._atoms: dict[str, AtomType] = {
            at.name: at for at in types if at.params.kwargs.get("type_") != "*"
        }

    @property
    def parameterizes(self) -> bool:
        return bool(self._table) or bool(self._atoms)

    def annotate(self, node: Entity) -> Mapping[str, Annotation]:
        """The nonbonded parameters ``node`` should carry, or ``{}``.

        Raises:
            ValueError: if the node carries no type, or its type has no pair
                parameters, and ``strict`` is on.
        """
        node_type = node.get(fields.TYPE)
        if node_type is None:
            if self._strict:
                raise ValueError(f"node must carry a type before pair typing: {node}")
            return {}

        out: dict[str, Annotation] = {}
        atom_type = self._atoms.get(str(node_type))
        if atom_type is not None:
            for key, value in atom_type.params.kwargs.items():
                if key in self._ATOM_KEYS:
                    out[key] = value

        pair_type = self._table.get(node_type)
        if pair_type is not None:
            # As for bonded terms: the type's own `id` is bookkeeping, not a
            # nonbonded parameter, and the node already has an identity.
            for key, value in pair_type.params.kwargs.items():
                if key != fields.ID:
                    out[key] = value
        elif atom_type is None and self._strict:
            raise ValueError(f"No PairType found for node type: {node_type}")
        return out


class ForceFieldParams:
    """Annotate a graph's pair and bonded terms from its node types.

    Not a :class:`~molpy.typifier.base.Typifier`: it never decides a node's type,
    it only spends one. Use it as the tail of a typifier's ``match`` (via
    :meth:`match`), or on its own when the node types are already on the graph
    (via :meth:`assign`).

    A kind the force field declares no types for is not parameterised by it, and
    its terms are left alone — that is a fact about the force field, not a
    failure. A term the force field *should* cover but does not is an error when
    ``strict`` is on.

    Args:
        forcefield: The force field to look types and parameters up in.
        strict: Raise on a term this force field ought to parameterise but does
            not, rather than leaving it untyped.
    """

    def __init__(self, forcefield: ForceField, *, strict: bool = True) -> None:
        self._ff = forcefield
        self._strict = strict
        # Scan AtomTypes once; share with TypeClassIndex + _PairMatcher.
        atom_types = list(forcefield.get_types(AtomType))
        index = TypeClassIndex(forcefield, atom_types=atom_types)
        self._pair = _PairMatcher(forcefield, atom_types=atom_types, strict=strict)
        self._terms = {
            link_cls: _TermMatcher(forcefield, ff_type, index, strict=strict)
            for link_cls, ff_type in _FF_TYPE_OF.items()
        }

    @property
    def forcefield(self) -> ForceField:
        return self._ff

    def match(
        self, graph: Any, node_types: Sequence[Mapping[str, Annotation]] | None = None
    ) -> Match:
        """Annotate ``graph``'s nodes and links.

        Args:
            graph: The graph to annotate. It is written to: the node types are
                stamped on so the bonded terms can be matched against them.
                Callers pass the private valence-completed copy a typifier owns.
            node_types: Per-node annotations the caller's matcher decided
                (``{"type": ...}``, possibly with ``class``). ``None`` means the
                types are already on ``graph``.

        Raises:
            TypeError: if ``graph`` carries a link kind no force field in molpy
                knows how to parameterise. Adding one is a line in
                :data:`_FF_TYPE_OF`, never a silent skip.
        """
        nodes = list(graph.nodes)
        node_out: list[dict[str, Annotation]] = [dict() for _ in nodes]
        if node_types is not None:
            for node, out, decided in zip(nodes, node_out, node_types, strict=True):
                out.update(decided)
                node.update(decided)

        if self._pair.parameterizes:
            for node, out in zip(nodes, node_out, strict=True):
                out.update(self._pair.annotate(node))

        links: dict[type[Link], tuple[Mapping[str, Annotation], ...]] = {}
        for link_cls in graph.links.classes():
            matcher = self._terms.get(link_cls)
            if matcher is None:
                raise TypeError(
                    f"{type(self).__name__} does not know which force-field type "
                    f"parameterizes {link_cls.__name__}; register it in _FF_TYPE_OF"
                )
            if not matcher.parameterizes:
                continue
            links[link_cls] = tuple(
                matcher.annotate(link) for link in graph.links.bucket(link_cls)
            )

        return Match(nodes=tuple(node_out), links=links)

    def assign(self, graph: Any) -> Any:
        """Return a copy of ``graph`` with its pair and bonded terms parameterised.

        For a graph whose node types are already known — the output of an
        AmberTools run, say, whose topology was then regenerated. Replaces the
        old ``Atomistic.assign_bonded_types``, which matched force-field type
        names by splitting them on ``"-"`` (no wildcards, no classes, no overlay
        layers) and left a term it could not match silently unlabelled.
        """
        typed = graph.copy()
        self.match(typed).write_onto(typed)
        return typed
