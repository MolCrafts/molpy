"""LAMMPS fix bond/react template serialization.

A ``fix bond/react`` template is a pre-reaction subgraph, the same subgraph after
the reaction, and the atom map between them. That is an **IO artifact**, not
reaction machinery: it is one serialization of the local environment a graph edit
disturbed. It used to live in ``molpy.reacter`` because a ``Reacter`` subclass
produced it; nothing produces it now but the caller, so it lives with the writer
that consumes it.

File format references:
    - LAMMPS ``fix bond/react``: https://docs.lammps.org/fix_bond_react.html
    - REACTER methodology: https://www.reacter.org
      (Gissinger, Jensen & Wise, Polymer 128, 211-217 (2017);
      Macromolecules 53, 9953-9961 (2020))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molpy.core.atomistic import Atom, Atomistic
    from molrs import Frame

#: Total charge must be conserved by a reaction template to within this
#: tolerance (elementary charge); a larger drift means an inconsistent template.
CHARGE_CONSERVATION_TOL: float = 1e-6

# Maps the metadata label key used by the LAMMPS data writer to the
# corresponding frame section name.
TYPE_LABEL_SECTIONS: dict[str, str] = {
    "atom_types": "atoms",
    "bond_types": "bonds",
    "angle_types": "angles",
    "dihedral_types": "dihedrals",
    "improper_types": "impropers",
}


@dataclass
class BondReactTemplate:
    """The pre/post subgraph pair ``fix bond/react`` needs, plus its atom map.

    Serialized into ``{name}_pre.mol``, ``{name}_post.mol`` and ``{name}.map``.

    Attributes:
        pre: Pre-reaction subgraph (the local environment before the edit).
        post: Post-reaction subgraph (same atoms, new topology).
        initiator_atoms: The pair of atoms that trigger the reaction
            (LAMMPS ``InitiatorIDs``). Exactly two.
        edge_atoms: Boundary atoms bonded to topology outside the template
            (LAMMPS ``EdgeIDs``).
        deleted_atoms: Atoms the reaction removes (LAMMPS ``DeleteIDs``).
        pre_react_id_to_atom: ``react_id`` → atom in ``pre``.
        post_react_id_to_atom: ``react_id`` → atom in ``post``.
    """

    pre: Atomistic
    post: Atomistic
    initiator_atoms: list[Atom]
    edge_atoms: list[Atom]
    deleted_atoms: list[Atom]
    pre_react_id_to_atom: dict
    post_react_id_to_atom: dict

    def assign_atom_ids(self) -> None:
        """Assign deterministic 1-based ``id`` values to the pre/post atoms.

        Insertion order defines the template-local indices the ``.map`` file
        uses, so writers call this before serializing.
        """
        for i, atom in enumerate(self.pre.atoms, start=1):
            atom["id"] = i
        for i, atom in enumerate(self.post.atoms, start=1):
            atom["id"] = i


class LammpsBondReactWriter:
    """Serialize a :class:`BondReactTemplate` into the files LAMMPS reads.

    The ``.map`` file is purely topological and independent of type numbering.
    The unified type maps, by contrast, must be shared between the system data
    file and every template, which is what lets ``fix bond/react`` match template
    atoms against the system.
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)

    # -- unified type numbering ---------------------------------------------

    @staticmethod
    def collect_type_maps(
        frames: Sequence[Frame],
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
        """Build unified string-type → 1-based ID mappings across ``frames``.

        Scans the ``type`` column of every topology section in every frame, keeps
        named types (skipping empty strings, ``"None"`` placeholders, and purely
        numeric labels that are already IDs), sorts them, and assigns 1-based
        integer IDs.

        Returns:
            ``(labels, type_maps)``: ``labels`` maps data-writer label keys
            (``"atom_types"``, …) to sorted type-name lists; ``type_maps`` maps
            section names (``"atoms"``, …) to ``{type_name: 1-based id}``.
        """
        labels: dict[str, list[str]] = {}
        type_maps: dict[str, dict[str, int]] = {}

        for label_key, section in TYPE_LABEL_SECTIONS.items():
            all_types: set[str] = set()
            for f in frames:
                if section in f and f[section].nrows > 0 and "type" in f[section]:
                    for t in f[section]["type"]:
                        s = str(t)
                        if s and s != "None" and not s.isdigit():
                            all_types.add(s)
            sorted_types = sorted(all_types)
            labels[label_key] = sorted_types
            type_maps[section] = {
                name: idx + 1 for idx, name in enumerate(sorted_types)
            }

        return labels, type_maps

    @staticmethod
    def apply_type_maps(
        frame: Frame,
        type_maps: dict[str, dict[str, int]],
        template_name: str = "",
    ) -> None:
        """Convert string ``type`` columns to unified numeric IDs, in place.

        Rows whose type is absent from the mapping (boundary topology with
        untyped terms) are dropped with a warning. The numeric IDs must match the
        system data file for template matching.
        """
        for section, tmap in type_maps.items():
            if section not in frame or frame[section].nrows == 0:
                continue
            block = frame[section]
            if "type" not in block:
                continue
            keep = [i for i in range(block.nrows) if str(block["type"][i]) in tmap]
            if len(keep) < block.nrows:
                dropped = block.nrows - len(keep)
                warnings.warn(
                    f"Dropped {dropped} {section} entries with unrecognized types "
                    f"from template '{template_name}'. Ensure all template "
                    f"topology is typed (pass a typifier to the assembler).",
                    stacklevel=2,
                )
                for key in list(block.keys()):
                    block[key] = block[key][keep]
            block["type"] = np.array(
                [tmap[str(block["type"][i])] for i in range(block.nrows)],
                dtype=np.int64,
            )

    # -- the .map file -------------------------------------------------------

    def write_map(self, template: BondReactTemplate) -> None:
        """Write ``{base_path}.map``.

        Raises:
            ValueError: if ``pre`` and ``post`` do not carry the same set of
                ``react_id`` atoms, if an initiator atom is unresolvable in
                ``pre``, or if there are not exactly two initiators.
        """
        template.assign_atom_ids()

        pre_index = {
            atom["react_id"]: i for i, atom in enumerate(template.pre.atoms, start=1)
        }
        post_index = {
            atom["react_id"]: i for i, atom in enumerate(template.post.atoms, start=1)
        }

        pre_rids, post_rids = set(pre_index), set(post_index)
        if pre_rids != post_rids:
            raise ValueError(
                f"Pre and post have different atoms!\n"
                f"  Missing in post: {pre_rids - post_rids}\n"
                f"  Missing in pre: {post_rids - pre_rids}"
            )

        equivalences = [(pre_index[rid], post_index[rid]) for rid in pre_rids]
        initiator_ids = self._initiator_ids(template, pre_index)
        initiator_rids = {a.get("react_id") for a in template.initiator_atoms}
        edge_ids = [
            pre_index[a.get("react_id")]
            for a in template.edge_atoms
            if a.get("react_id")
            and a.get("react_id") in pre_index
            and a.get("react_id") not in initiator_rids
        ]
        deleted_ids = [
            pre_index[a.get("react_id")]
            for a in template.deleted_atoms
            if a.get("react_id") and a.get("react_id") in pre_index
        ]

        map_path = Path(f"{self._base_path}.map")
        with map_path.open("w", encoding="utf-8") as f:
            f.write("# auto-generated map file for fix bond/react\n\n")
            f.write(f"{len(equivalences)} equivalences\n")
            f.write(f"{len(edge_ids)} edgeIDs\n")
            f.write(f"{len(deleted_ids)} deleteIDs\n\n")
            f.write("InitiatorIDs\n\n")
            for idx in initiator_ids:
                f.write(f"{idx}\n")
            f.write("\nEdgeIDs\n\n")
            for idx in edge_ids:
                f.write(f"{idx}\n")
            f.write("\nDeleteIDs\n\n")
            for idx in deleted_ids:
                f.write(f"{idx}\n")
            f.write("\nEquivalences\n\n")
            for pre_id, post_id in sorted(equivalences):
                f.write(f"{pre_id}   {post_id}\n")

    @staticmethod
    def _initiator_ids(
        template: BondReactTemplate, pre_index: dict[object, int]
    ) -> list[int]:
        """Template-local ids of the two initiators, in template order.

        Order is preserved (never a set) so the ``.map`` output is deterministic.
        """
        ids: list[int] = []
        for anchor in template.initiator_atoms:
            rid = anchor.get("react_id")
            if rid is None or rid not in pre_index:
                raise ValueError(
                    f"Initiator atom (react_id={rid!r}, "
                    f"element={anchor.get('element')}) is not resolvable in the "
                    f"pre template; extract a wider local environment."
                )
            ids.append(pre_index[rid])
        if len(ids) != 2:
            raise ValueError(
                f"fix bond/react requires exactly 2 initiator atoms, got {len(ids)}."
            )
        return ids
