"""Tests for LAMMPS fix bond/react serialization (semantic, not byte-golden).

``TestWriteBondReactMap`` unit-tests ``write_bond_react_map``: header counts,
section order, 1-based IDs, and a ValueError on a pre/post atom-set mismatch.

The template couples two propanes (three carbons, so the radius-2 environment
has genuine edge atoms): ``pre`` is the committed ``mol2/bond_react_pre.mol2``,
``post`` the same atoms after the edit, and every topology type name is derived
from endpoint atom types.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molpy as mp
from molpy.core import Atomistic, RelationRef
from molpy.io.lammps_bond_react import BondReactTemplate


# ===================================================================
# Deterministic reaction builder (helpers, not tests)
# ===================================================================

# Rows (``react_id``) of ``mol2/bond_react_pre.mol2``: the radius-2 environment
# of a C-C coupling between two propanes. Rows 1-8 are the left propane (its
# terminal carbon 3 reacts), rows 9-16 the right one (its first carbon 9 reacts).
INITIATORS = (3, 9)
LEAVING = (6, 12)  # one hydrogen on each initiator
EDGES = (1, 11)  # carbons whose remaining hydrogens lie outside the template


def _canonical_link_type(link: RelationRef) -> str:
    """Orientation-independent topology type name from endpoint atom types."""
    names = tuple(str(ep["type"]) for ep in link.endpoints)
    return "-".join(min(names, names[::-1]))


def _assign_link_types(struct: Atomistic) -> None:
    """Deterministically (re)type every bond/angle/dihedral in ``struct``."""
    for link in struct.bonds:
        link["type"] = _canonical_link_type(link)
    for link in struct.angles:
        link["type"] = _canonical_link_type(link)
    for link in struct.dihedrals:
        link["type"] = _canonical_link_type(link)


def _build_reaction(test_data_dir: Path) -> BondReactTemplate:
    """Deterministic C-C coupling template, built as data rather than run by an engine.

    A bond/react template *is* a description of an edit: the radius-2 environment
    before it (the committed ``pre`` fixture), the same atoms after it, and which
    of them are initiators, edges and deletions.
    """
    frame = mp.io.read_mol2(test_data_dir / "mol2" / "bond_react_pre.mol2")
    frame["atoms"]["react_id"] = frame["atoms"]["id"]
    pre = Atomistic.from_frame(frame)
    pre.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
    by_react_id = {a["react_id"]: a for a in pre.atoms}

    # the same atoms, after the edit: drop the two C-H bonds, add the C-C bond
    post = pre.copy()
    post_by_react_id = {a["react_id"]: a for a in post.atoms}
    for react_id in LEAVING:
        target = post_by_react_id[react_id]
        for bond in list(post.bonds):
            if target in bond.endpoints:
                post.remove_bond(bond.handle)
    post.def_bond(*(post_by_react_id[react_id] for react_id in INITIATORS))
    # clear_existing: `post` inherited `pre`'s angles/dihedrals, including the ones
    # running through the hydrogens the reaction deletes. LAMMPS would then try to
    # build an angle on a deleted atom ("Angle atoms 2 3 9 missing").
    post.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)

    template = BondReactTemplate(
        pre=pre,
        post=post,
        initiator_atoms=[by_react_id[react_id] for react_id in INITIATORS],
        edge_atoms=[by_react_id[react_id] for react_id in EDGES],
        deleted_atoms=[by_react_id[react_id] for react_id in LEAVING],
    )
    for struct in (template.pre, template.post):
        _assign_link_types(struct)
    return template


def _parse_equivalences(content: str) -> list[tuple[int, int]]:
    lines = content.splitlines()
    pairs: list[tuple[int, int]] = []
    for line in lines[lines.index("Equivalences") + 1 :]:
        parts = line.split()
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


class TestWriteBondReactMap:
    """Unit tests for write_bond_react_map (module does not exist yet → RED)."""

    def _write_map(
        self, tmp_path: Path, test_data_dir: Path
    ) -> tuple[str, BondReactTemplate]:
        from molpy.io import write_bond_react_map

        template = _build_reaction(test_data_dir)
        write_bond_react_map(template, tmp_path / "rxn1")
        content = (tmp_path / "rxn1.map").read_text(encoding="utf-8")
        return content, template

    def test_map_header_counts(self, tmp_path: Path, TEST_DATA_DIR: Path) -> None:
        """Header lines carry the equivalence/edge/delete counts of the template."""
        content, template = self._write_map(tmp_path, TEST_DATA_DIR)

        pre_rids = {a["react_id"] for a in template.pre.atoms}
        initiator_rids = {a.get("react_id") for a in template.initiator_atoms}
        n_equiv = len(list(template.pre.atoms))
        n_edge = len(
            [
                a
                for a in template.edge_atoms
                if a.get("react_id") in pre_rids
                and a.get("react_id") not in initiator_rids
            ]
        )
        n_delete = len(
            [a for a in template.deleted_atoms if a.get("react_id") in pre_rids]
        )

        lines = content.splitlines()
        assert f"{n_equiv} equivalences" in lines
        assert f"{n_edge} edgeIDs" in lines
        assert f"{n_delete} deleteIDs" in lines

    def test_map_sections_present_in_order(
        self, tmp_path: Path, TEST_DATA_DIR: Path
    ) -> None:
        """InitiatorIDs, EdgeIDs, DeleteIDs, Equivalences appear in that order."""
        content, _ = self._write_map(tmp_path, TEST_DATA_DIR)
        positions = [
            content.index("InitiatorIDs"),
            content.index("EdgeIDs"),
            content.index("DeleteIDs"),
            content.index("Equivalences"),
        ]
        assert positions == sorted(positions)

    def test_map_ids_are_1based(self, tmp_path: Path, TEST_DATA_DIR: Path) -> None:
        """Equivalence IDs are >= 1 and pre-side IDs cover 1..n_atoms exactly."""
        content, template = self._write_map(tmp_path, TEST_DATA_DIR)
        pairs = _parse_equivalences(content)
        n_atoms = len(list(template.pre.atoms))

        assert all(pre >= 1 and post >= 1 for pre, post in pairs)
        assert sorted(pre for pre, _ in pairs) == list(range(1, n_atoms + 1))

    def test_map_mismatched_pre_post_raises(self, tmp_path: Path) -> None:
        """Post missing one react_id must raise ValueError, not write silently."""
        from molpy.io import write_bond_react_map

        pre = Atomistic()
        a1 = pre.def_atom(element="C", type="c3", react_id=1)
        a2 = pre.def_atom(element="C", type="c3", react_id=2)
        pre.def_bond(a1, a2, type="c3-c3")

        post = Atomistic()
        post.def_atom(element="C", type="c3", react_id=1)

        template = BondReactTemplate(
            pre=pre,
            post=post,
            initiator_atoms=[a1, a2],
            edge_atoms=[],
            deleted_atoms=[],
        )
        with pytest.raises(ValueError):
            write_bond_react_map(template, tmp_path / "bad")
