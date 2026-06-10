"""LAMMPS fix bond/react template serialization.

Writers and helpers for the files consumed by the LAMMPS
``fix bond/react`` command: the ``.map`` atom-mapping file and the
unified string-type → numeric-ID mappings shared between the system
data file and the pre/post molecule templates.

The single user-facing entry point for a complete reactive system is
:func:`molpy.io.write_lammps_bond_react_system`; this module holds the
serialization building blocks it delegates to.

File format references:
    - LAMMPS ``fix bond/react``: https://docs.lammps.org/fix_bond_react.html
    - REACTER methodology: https://www.reacter.org
      (Gissinger, Jensen & Wise, Polymer 128, 211-217 (2017);
      Macromolecules 53, 9953-9961 (2020))
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molpy.core.frame import Frame
    from molpy.reacter.bond_react import BondReactTemplate

# Maps the metadata label key used by the LAMMPS data writer to the
# corresponding frame section name.
TYPE_LABEL_SECTIONS: dict[str, str] = {
    "atom_types": "atoms",
    "bond_types": "bonds",
    "angle_types": "angles",
    "dihedral_types": "dihedrals",
    "improper_types": "impropers",
}


def collect_type_maps(
    frames: Sequence[Frame],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Build unified string-type → 1-based ID mappings across frames.

    Scans the ``type`` column of every topology section in every frame,
    keeps named types (skipping empty strings, ``"None"`` placeholders,
    and purely numeric labels that are already IDs), sorts them, and
    assigns 1-based integer IDs. Using one mapping for the system frame
    and all template frames is what lets ``fix bond/react`` match
    template atoms against the system data file (see
    https://docs.lammps.org/fix_bond_react.html).

    Args:
        frames: Frames to scan (system frame plus pre/post template
            frames).

    Returns:
        A ``(labels, type_maps)`` pair. ``labels`` maps data-writer
        label keys (``"atom_types"``, ``"bond_types"``, …) to sorted
        type-name lists; ``type_maps`` maps section names (``"atoms"``,
        ``"bonds"``, …) to ``{type_name: 1-based id}`` dicts.
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
        type_maps[section] = {name: idx + 1 for idx, name in enumerate(sorted_types)}

    return labels, type_maps


def apply_type_maps(
    frame: Frame,
    type_maps: dict[str, dict[str, int]],
    template_name: str = "",
) -> None:
    """Convert string ``type`` columns to unified numeric IDs in place.

    Rows whose type is not present in the mapping (e.g. boundary
    topology with ``None`` types) are dropped with a warning, matching
    the historical behavior of
    :func:`molpy.io.write_lammps_bond_react_system`. The numeric IDs
    must match the system data file for ``fix bond/react`` template
    matching (see https://docs.lammps.org/fix_bond_react.html).

    Args:
        frame: Template frame to rewrite (mutated in place).
        type_maps: Section → ``{type_name: id}`` mapping from
            :func:`collect_type_maps`.
        template_name: Template name used in the warning message.
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
                f"Dropped {dropped} {section} entries with "
                f"unrecognized types from template '{template_name}'. "
                f"Ensure all template topology is typed "
                f"(pass typifier to reacter.run()).",
                stacklevel=2,
            )
            for key in list(block.keys()):
                block[key] = block[key][keep]
        block["type"] = np.array(
            [tmap[str(block["type"][i])] for i in range(block.nrows)],
            dtype=np.int64,
        )


def write_bond_react_map(template: BondReactTemplate, base_path: str | Path) -> None:
    """Write the ``.map`` file for LAMMPS ``fix bond/react``.

    The ``.map`` file is purely topological and independent of type
    numbering. It contains a header with section counts followed by the
    ``InitiatorIDs``, ``EdgeIDs``, ``DeleteIDs``, and ``Equivalences``
    sections; atom IDs are 1-based template-local indices (see
    https://docs.lammps.org/fix_bond_react.html).

    Args:
        template: Bond/react template carrying pre/post subgraphs and
            the initiator/edge/delete atom lists.
        base_path: Stem path; produces ``{base_path}.map``.

    Raises:
        ValueError: If the pre and post subgraphs do not contain the
            same set of ``react_id`` atoms.
    """
    template.assign_atom_ids()

    pre_rid_to_idx = {
        atom["react_id"]: i for i, atom in enumerate(template.pre.atoms, start=1)
    }
    post_rid_to_idx = {
        atom["react_id"]: i for i, atom in enumerate(template.post.atoms, start=1)
    }

    pre_rids = set(pre_rid_to_idx)
    post_rids = set(post_rid_to_idx)
    if pre_rids != post_rids:
        raise ValueError(
            f"Pre and post have different atoms!\n"
            f"  Missing in post: {pre_rids - post_rids}\n"
            f"  Missing in pre: {post_rids - pre_rids}"
        )

    equiv = [(pre_rid_to_idx[rid], post_rid_to_idx[rid]) for rid in pre_rids]

    initiator_rids = {a.get("react_id") for a in template.initiator_atoms}
    initiator_ids = [
        pre_rid_to_idx[rid] for rid in initiator_rids if rid and rid in pre_rid_to_idx
    ]
    edge_ids = [
        pre_rid_to_idx[a.get("react_id")]
        for a in template.edge_atoms
        if a.get("react_id")
        and a.get("react_id") in pre_rid_to_idx
        and a.get("react_id") not in initiator_rids
    ]
    deleted_ids = [
        pre_rid_to_idx[a.get("react_id")]
        for a in template.deleted_atoms
        if a.get("react_id") and a.get("react_id") in pre_rid_to_idx
    ]

    map_path = Path(f"{base_path}.map")
    with map_path.open("w", encoding="utf-8") as f:
        f.write("# auto-generated map file for fix bond/react\n\n")
        f.write(f"{len(equiv)} equivalences\n")
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
        for pre_id, post_id in sorted(equiv):
            f.write(f"{pre_id}   {post_id}\n")
