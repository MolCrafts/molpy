"""tleap's prmtop, read back as the assignment of the graph it was built from.

antechamber and tleap keep atom order, so prmtop atom *i* is graph atom *i*;
the AmberTools typifiers and :class:`molpy.builder.AmberPolymerBuilder`
assign a prmtop through :class:`Prmtop`. Their outputs merge when the files
declare the same 1-4 weights. Each typed graph keeps that file's exclusions
and any 1-4 scale that differs from those weights.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from molpy.core import Angle, Atom, Atomistic, Bond, Dihedral, Improper, RelationRef
from molpy._core import ForceField, Style, TypeAssignment, Typifier
from molpy.io.amber import AmberPrmtopForcefieldReader

if TYPE_CHECKING:
    from pathlib import Path

    from molpy._core import Annotation, AssignmentParamValue

    #: ``(style, name, endpoints, params)``: a term's type annotation.
    type TermAnnotation = tuple[str, str, list[str], dict[str, AssignmentParamValue]]

    #: One prmtop row: atom indices, its type name, and ``exclude_14`` when
    #: the block carries that column.
    type TermRow = tuple[tuple[int, ...], str, bool | None]


def assignment_params(params: Mapping[str, object]) -> dict[str, AssignmentParamValue]:
    """``params`` as a :class:`TypeAssignment` carries them.

    A number or string is stamped on the graph row. An array (a cmap
    ``grid``) stays on the type.

    Raises:
        ValueError: A param that is neither a number, a string nor an array.
    """
    out: dict[str, AssignmentParamValue] = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            out[key] = np.asarray(value, dtype=np.float64)
        elif isinstance(value, (int, float, str)):
            out[key] = value
        else:
            raise ValueError(
                f"param {key!r} is {type(value).__name__}; a TypeAssignment "
                "carries numbers, strings and arrays"
            )
    return out


class Prmtop:
    """A tleap prmtop, read back as the TypeAssignment of its source graph.

    antechamber and tleap keep atom order, so prmtop atom *i* is graph atom
    *i*. A bond or an angle matches either orientation of its atoms. A proper
    dihedral is stored in the prmtop's order — the order its type name is
    spelled in, which is the order the signed dihedral is priced in — and an
    improper keeps the prmtop's order, because reversing it is a different
    term. Each row is annotated with that row's type — a multi-term improper
    keeps every ``<quartet>@<periodicity>``, and a cmap keeps its grid.
    ``exclude_14`` is stamped on the dihedral or improper it flags. The
    assignment carries the file's 1-4 weights. A 1-4 whose own scale differs
    is written on the graph as ``pair_overrides``; the file's ``exclusions``
    replace the graph's. Neither block is a pair type. A graph that already
    holds ``pairs`` is refused: that block is what a later compile prices,
    and it would hide these rows. Impropers and cmaps are replaced from the
    file. Any refusal leaves the graph unchanged.
    """

    def __init__(self, prmtop: Path) -> None:
        self.forcefield, self.frame = AmberPrmtopForcefieldReader().read_system(prmtop)

    def assignment(
        self, graph: Atomistic, charges: Sequence[float] | npt.NDArray[np.float64]
    ) -> TypeAssignment:
        """Regenerate ``graph``'s terms to the prmtop's set and annotate them.

        Proper dihedrals are stored in the prmtop's order. Perception's
        central-atom order is not the order the signed dihedral is priced in.

        Args:
            graph: The graph the prmtop was built from. Its angles, dihedrals,
                impropers, cmaps, exclusions and pair_overrides are rewritten
                to the file. A refusal leaves the graph unchanged.
            charges: The charge to stamp on each atom, in row order. The
                length has to equal the atom count.

        Raises:
            ValueError: The graph and the prmtop disagree on atom count,
                element, or charge length; the prmtop force field declares no
                ``special_bonds``; the graph already has a ``pairs`` block; a
                bond, angle, or proper dihedral the two do not share; or a
                type name the force field does not declare. Impropers and
                cmaps are replaced from the file, so a perceived one the file
                lacks is not an error.
        """
        # Bonds, angles and propers are matched on the perceived topology.
        # The copy takes a refusal, so ``graph`` stays as the caller built it.
        self._rewrite(copy.copy(graph), charges)
        return self._rewrite(graph, charges)

    def _rewrite(
        self, graph: Atomistic, charges: Sequence[float] | npt.NDArray[np.float64]
    ) -> TypeAssignment:
        """Rewrite ``graph`` to the prmtop and return its assignment."""
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
        if graph.n_terms("pairs"):
            raise ValueError(
                "the graph already has a pairs block, which is priced instead of "
                "pair_overrides; drop it before assignment"
            )
        if len(charges) != len(graph_atoms):
            raise ValueError(
                f"the prmtop has {len(graph_atoms)} atoms, the charges {len(charges)}"
            )
        weights = self.forcefield.special_bonds
        if weights is None:
            raise ValueError(
                f"prmtop force field {self.forcefield.name!r} declares no special_bonds"
            )
        (atom_style,) = self.forcefield.get_styles("atom")
        mass = {
            t.name: assignment_params(t.params)["mass"] for t in atom_style.get_types()
        }
        for name in types:
            if name not in mass:
                raise ValueError(f"atom type {name!r} is not in the prmtop force field")
        nodes: list[dict[str, Annotation]] = [
            {
                "type": (atom_style.name, name, (), {"mass": mass[name]}),
                "charge": float(charge),
            }
            for name, charge in zip(types, charges, strict=True)
        ]
        styles = [
            (s.category, s.name, assignment_params(s.params))
            for s in self.forcefield.get_styles(Style)
        ]
        pairs = [
            (
                style.name,
                t.name,
                [e.name for e in t.endpoints],
                assignment_params(t.params),
            )
            for style in self.forcefield.get_styles("pair")
            for t in style.get_types()
        ]
        row = {atom.handle: index for index, atom in enumerate(graph_atoms)}
        four = ("atomi", "atomj", "atomk", "atoml")
        frame_rows = {
            Bond: self._rows("bonds", four[:2]),
            Angle: self._rows("angles", four[:3]),
            Dihedral: self._rows("dihedrals", four),
            Improper: self._rows("impropers", four),
        }

        graph.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
        graph.del_impropers(list(graph.impropers))
        for atoms_at, _, _ in frame_rows[Improper]:
            graph.add_improper(*(graph_atoms[i] for i in atoms_at))
        self._store_proper_order(graph, graph_atoms, frame_rows[Dihedral], row)

        relations = {
            Bond: graph.bonds,
            Angle: graph.angles,
            Dihedral: graph.dihedrals,
            Improper: graph.impropers,
        }
        links: dict[type | str, list[dict[str, Annotation]]] = {}
        for kind in (Bond, Angle, Dihedral, Improper):
            links[kind] = self._annotate(
                kind,
                list(relations[kind]),
                frame_rows[kind],
                reversible=kind is not Improper,
                row=row,
            )
        cmaps = self._cmaps(graph, graph_atoms)
        if cmaps:
            links["cmaps"] = cmaps

        self._install_block(
            graph,
            graph_atoms,
            "pair_overrides",
            ("epsilon", "sigma", "charge_product", "lj_scale", "coul_scale"),
        )
        self._install_block(graph, graph_atoms, "exclusions", ())
        return TypeAssignment(
            nodes, links, styles=styles, pairs=pairs, special_bonds=weights
        )

    def _store_proper_order(
        self,
        graph: Atomistic,
        atoms: Sequence[Atom],
        rows: list[TermRow],
        index: Mapping[int, int],
    ) -> None:
        """Store the prmtop's proper order when it names the perceived set.

        Perception lists a quartet with the smaller central atom first. The
        prmtop row is already in the order its type is spelled, and that is
        the order ``nφ − phase`` is priced in. A set the two disagree on is
        left as perceived, so :meth:`_annotate` still rejects it.
        """

        def key(quartet: tuple[int, ...]) -> tuple[int, ...]:
            return min(quartet, quartet[::-1])

        perceived = [
            key(tuple(index[atom.handle] for atom in relation.endpoints))
            for relation in graph.dihedrals
        ]
        filed = [key(quartet) for quartet, _, _ in rows]
        if sorted(perceived) != sorted(filed):
            return
        graph.del_dihedrals(list(graph.dihedrals))
        for quartet, _, _ in rows:
            graph.add_dihedral(*(atoms[i] for i in quartet))

    def _install_block(
        self,
        graph: Atomistic,
        atoms: Sequence[Atom],
        block: str,
        props: tuple[str, ...],
    ) -> None:
        """Replace ``graph``'s ``block`` with the prmtop frame's rows.

        A cell the frame marks null is left unset, so the typed frame keeps
        that hole. A prmtop with no ``block`` clears the graph's rows.
        """
        graph.del_terms(list(graph.terms(block)))
        if block not in self.frame:
            return
        data = self.frame[block]
        present = [key for key in props if key in data]
        columns = {key: data[key] for key in present}
        masks = {key: data.validity(key) for key in present}
        for index, (left, right) in enumerate(
            zip(data["atomi"], data["atomj"], strict=True)
        ):
            stated: dict[str, float] = {}
            for key in present:
                mask = masks[key]
                if mask is not None and not bool(mask[index]):
                    continue
                stated[key] = float(columns[key][index])
            graph.add_term(block, [atoms[int(left)], atoms[int(right)]], **stated)

    def _rows(self, block: str, keys: tuple[str, ...]) -> list[TermRow]:
        """``block``'s rows in file order, or none when the prmtop has no block."""
        if block not in self.frame:
            return []
        data = self.frame[block]
        columns = [data[key] for key in keys]
        names = [str(name) for name in data["type"]]
        flags = (
            [bool(flag) for flag in data["exclude_14"]]
            if "exclude_14" in data
            else None
        )
        rows: list[TermRow] = []
        for index, atoms in enumerate(zip(*columns, strict=True)):
            flag = None if flags is None else flags[index]
            rows.append((tuple(int(atom) for atom in atoms), names[index], flag))
        return rows

    def _annotate(
        self,
        kind: type[Bond] | type[Angle] | type[Dihedral] | type[Improper],
        relations: Sequence[RelationRef],
        rows: list[TermRow],
        *,
        reversible: bool,
        row: Mapping[int, int],
    ) -> list[dict[str, Annotation]]:
        """One annotation per ``relations`` row, from that prmtop row's type.

        ``reversible`` kinds match either orientation. An improper matches
        its stored order. Rows of one key are consumed in file order, which
        is the order the impropers were inserted.
        """
        label = kind.__name__.lower()
        pending: dict[tuple[int, ...], deque[tuple[str, bool | None]]] = {}
        for atoms, name, flag in rows:
            key = min(atoms, atoms[::-1]) if reversible else atoms
            pending.setdefault(key, deque()).append((name, flag))
        types = self._types(label)
        annotations: list[dict[str, Annotation]] = []
        for relation in relations:
            atoms = tuple(row[atom.handle] for atom in relation.endpoints)
            key = min(atoms, atoms[::-1]) if reversible else atoms
            queue = pending.get(key)
            if not queue:
                raise ValueError(f"{label} {key} is not in the prmtop")
            name, flag = queue.popleft()
            annotation = types.get(name)
            if annotation is None:
                raise ValueError(
                    f"{label} type {name!r} is not in the prmtop force field"
                )
            stamped: dict[str, Annotation] = {"type": annotation}
            if flag is not None:
                stamped["exclude_14"] = flag
            annotations.append(stamped)
        for key in sorted(key for key, queue in pending.items() if queue):
            raise ValueError(f"prmtop {label} {key} is not in the graph")
        return annotations

    def _cmaps(
        self, graph: Atomistic, atoms: Sequence[Atom]
    ) -> list[dict[str, Annotation]]:
        """Replace ``graph``'s cmaps with the prmtop's, in file order."""
        if "cmaps" in graph.term_blocks():
            graph.del_terms(list(graph.terms("cmaps")))
        rows = self._rows("cmaps", ("atomi", "atomj", "atomk", "atoml", "atomm"))
        types = self._types("cmap")
        annotations: list[dict[str, Annotation]] = []
        for indices, name, _ in rows:
            graph.add_term("cmaps", [atoms[i] for i in indices])
            annotation = types.get(name)
            if annotation is None:
                raise ValueError(f"cmap type {name!r} is not in the prmtop force field")
            annotations.append({"type": annotation})
        return annotations

    def _types(self, category: str) -> dict[str, TermAnnotation]:
        """Type name → its annotation, params included (a cmap grid too)."""
        found: dict[str, TermAnnotation] = {}
        for style in self.forcefield.get_styles(category):
            for term in style.get_types():
                ends = [endpoint.name for endpoint in term.endpoints]
                found[term.name] = (
                    style.name,
                    term.name,
                    ends,
                    assignment_params(term.params),
                )
        return found


class AmberTypifier(Typifier[Atomistic]):
    """The seed every AmberTools typifier shares.

    A typifier's output starts as this force field's empty likeness: units
    ``real``, and no 1-4 weights. A prmtop assignment declares the file's
    weights. A later file whose weights differ is an error, and the output
    is left as it was.
    """

    def source_forcefield(self) -> ForceField:
        """An empty force field declaring AMBER units."""
        return ForceField("amber", units="real")


class PrmtopTypifier(AmberTypifier):
    """Assign a tleap prmtop to the graph it was written for.

    No program runs: ``assign`` regenerates the graph's terms to the prmtop's
    set and stamps the prmtop's types and charges on it, exactly as the
    AmberTools typifiers do after tleap. Proper dihedrals are stored in the
    prmtop's order. The force field carries the file's
    1-4 weights and its types, without the prmtop's atom-type row ids. The
    typed graph carries the file's exclusions and, where a 1-4 scale differs
    from those weights, its ``pair_overrides``. A GAFF prmtop therefore merges
    with :class:`~molpy.ff.typifier.GaffTypifier`.
    :class:`molpy.builder.AmberPolymerBuilder` assigns its chain's prmtop
    with it.

    Args:
        prmtop: The prmtop; its atom *i* is graph atom *i*.
    """

    def __init__(self, prmtop: Path) -> None:
        super().__init__()
        self._prmtop = Prmtop(prmtop)

    def assign(self, graph: Atomistic) -> TypeAssignment:
        """The prmtop's types, charges and terms for ``graph``.

        Raises:
            ValueError: The graph and the prmtop disagree on atom count,
                element, or charge length; the prmtop declares no
                ``special_bonds``; the graph already has a ``pairs`` block; a
                bond, angle, or proper the two do not share; or a type the
                force field does not declare. Impropers and cmaps are
                replaced from the file. The graph is left unchanged.
        """
        return self._prmtop.assignment(graph, self._prmtop.frame["atoms"]["charge"])
