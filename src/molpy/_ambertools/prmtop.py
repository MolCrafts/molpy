"""tleap's prmtop, read back as the assignment of the graph it was built from.

antechamber and tleap keep atom order, so prmtop atom *i* is graph atom *i*;
the AmberTools typifiers and :class:`molpy.builder.AmberPolymerBuilder`
assign a prmtop through :class:`Prmtop`, so their outputs are alike and
their force fields merge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from molrs.core import Angle, Atomistic, Bond, Dihedral, Improper
from molrs.core.constants import AMBER_SCEE, AMBER_SCNB
from molrs.ff.forcefield import ForceField, Style
from molrs.ff.typifier import TypeAssignment, Typifier
from molrs.io.amber import AmberPrmtopForcefieldReader

if TYPE_CHECKING:
    from pathlib import Path


class Prmtop:
    """A tleap prmtop, read back as the TypeAssignment it assigns to its source graph.

    antechamber and tleap keep atom order, so prmtop atom *i* is graph atom *i*;
    bonded terms are matched by their endpoint rows.
    """

    def __init__(self, prmtop: Path) -> None:
        self.forcefield, self.frame = AmberPrmtopForcefieldReader().read_system(prmtop)

    def assignment(self, graph: Atomistic, charges: Sequence[float]) -> TypeAssignment:
        """Regenerate ``graph``'s terms to the prmtop's set and annotate them.

        Args:
            graph: The graph the prmtop was built from (the typifier's private
                copy; its angles, dihedrals and impropers are rewritten).
            charges: The charge to stamp on each atom, in row order.

        Raises:
            ValueError: A graph term the prmtop lacks, or the reverse; an atom
                count or element that disagrees with the prmtop.
        """
        atoms = self.frame["atoms"]
        types = [str(name) for name in atoms["type"]]
        elements = [int(number) for number in atoms["atomic_number"]]
        graph_atoms = list(graph.atoms)
        if len(graph_atoms) != len(types):
            raise ValueError(
                f"the prmtop has {len(types)} atoms, the graph {len(graph_atoms)}"
            )
        for index, atom in enumerate(graph_atoms):
            if atom.get("atomic_number") != elements[index]:
                raise ValueError(
                    f"atom {index} is {atom.get('atomic_number')} in the graph but "
                    f"{elements[index]} in the prmtop"
                )
        row = {atom.handle: index for index, atom in enumerate(graph_atoms)}

        prmtop_terms = self._terms()
        graph.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
        graph.del_impropers(list(graph.impropers))
        for term in prmtop_terms[Improper]:
            graph.add_improper(*(graph_atoms[i] for i in term))

        links: dict[type, list[dict]] = {}
        for kind in (Bond, Angle, Dihedral, Improper):
            relations = list(
                {
                    Bond: graph.bonds,
                    Angle: graph.angles,
                    Dihedral: graph.dihedrals,
                    Improper: graph.impropers,
                }[kind]
            )
            ends = [tuple(row[a.handle] for a in r.endpoints) for r in relations]
            self._require_same(kind, ends, prmtop_terms[kind])
            lookup = self._lookup(kind.__name__.lower())
            links[kind] = [
                {"type": lookup[tuple(types[i] for i in term)]} for term in ends
            ]

        (atom_style,) = self.forcefield.get_styles("atom")
        mass = {t.name: t.params["mass"] for t in atom_style.get_types()}
        nodes = [
            {
                "type": (atom_style.name, name, (), {"mass": mass[name]}),
                "charge": float(charge),
            }
            for name, charge in zip(types, charges, strict=True)
        ]
        styles = [
            (s.category, s.name, s.params) for s in self.forcefield.get_styles(Style)
        ]
        pairs = [
            (
                style.name,
                t.name,
                [e.name for e in t.endpoints],
                dict(t.params.items()),
            )
            for style in self.forcefield.get_styles("pair")
            for t in style.get_types()
        ]
        return TypeAssignment(nodes, links, styles=styles, pairs=pairs)

    def _terms(self) -> dict[type, list[tuple[int, ...]]]:
        """The prmtop's terms per kind, as atom-row tuples in file order.

        The structure reader already gives one proper dihedral per atom quartet
        (a multi-term torsion is one term) and keeps impropers in their own
        block.
        """
        four = ("atomi", "atomj", "atomk", "atoml")
        columns = {
            Bond: ("bonds", four[:2]),
            Angle: ("angles", four[:3]),
            Dihedral: ("dihedrals", four),
            Improper: ("impropers", four),
        }
        rows: dict[type, list[tuple[int, ...]]] = {}
        for kind, (block, keys) in columns.items():
            rows[kind] = []
            if block in self.frame:
                cols = [self.frame[block][key] for key in keys]
                rows[kind] = [tuple(int(v) for v in term) for term in zip(*cols)]
        return rows

    @staticmethod
    def _require_same(
        kind: type,
        graph_terms: list[tuple[int, ...]],
        prmtop_terms: list[tuple[int, ...]],
    ) -> None:
        """Raise unless the graph and the prmtop carry the same terms of ``kind``."""
        graph_keys = {min(t, t[::-1]) for t in graph_terms}
        prmtop_keys = {min(t, t[::-1]) for t in prmtop_terms}
        for key in sorted(graph_keys - prmtop_keys):
            raise ValueError(f"{kind.__name__.lower()} {key} is not in the prmtop")
        for key in sorted(prmtop_keys - graph_keys):
            raise ValueError(
                f"prmtop {kind.__name__.lower()} {key} is not in the graph"
            )

    def _lookup(self, category: str) -> dict[tuple[str, ...], tuple]:
        """Endpoint atom types (either orientation) → the term's annotation."""
        lookup: dict[tuple[str, ...], tuple] = {}
        for style in self.forcefield.get_styles(category):
            for t in style.get_types():
                ends = [e.name for e in t.endpoints]
                annotation = (style.name, t.name, ends, dict(t.params.items()))
                lookup[tuple(ends)] = annotation
                lookup[tuple(reversed(ends))] = annotation
        return lookup


class AmberTypifier(Typifier):
    """The seed every AmberTools typifier shares: AMBER's declared settings.

    A typifier's output force field starts as its ``source_forcefield()``'s empty
    likeness, so declaring units ``real`` and the AMBER 1-4 scaling here is
    what makes every typed output carry them (tleap's prmtop defaults:
    ``coul_14 = 1 / SCEE``, ``lj_14 = 1 / SCNB``).
    """

    def source_forcefield(self) -> ForceField:
        """An empty force field declaring AMBER units and 1-4 scaling."""
        ff = ForceField("amber", units="real")
        ff.set_special_bonds(
            [0.0, 0.0, 1.0 / AMBER_SCNB],
            [0.0, 0.0, 1.0 / AMBER_SCEE],
        )
        return ff


class PrmtopTypifier(AmberTypifier):
    """Assign a tleap prmtop to the graph it was written for.

    No program runs: ``assign`` regenerates the graph's terms to the prmtop's
    set and stamps the prmtop's types and charges on it, exactly as the
    AmberTools typifiers do after tleap. The output force field is their
    likeness (AMBER units and 1-4 scaling, types without the prmtop's row
    ids), so it merges with theirs. :class:`molpy.builder.AmberPolymerBuilder`
    assigns its chain's prmtop with it.

    Args:
        prmtop: The prmtop; its atom *i* is graph atom *i*.
    """

    def __init__(self, prmtop: Path) -> None:
        super().__init__()
        self._prmtop = Prmtop(prmtop)

    def assign(self, graph: Atomistic) -> TypeAssignment:
        """The prmtop's types, charges and terms for ``graph``.

        Raises:
            ValueError: The graph and the prmtop disagree on an atom or term.
        """
        return self._prmtop.assignment(graph, self._prmtop.frame["atoms"]["charge"])
