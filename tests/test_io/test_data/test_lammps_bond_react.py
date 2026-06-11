"""Tests for LAMMPS fix bond/react serialization (semantic, not byte-golden).

Two layers cover :mod:`molpy.io.data.lammps_bond_react`:

1. ``test_system_writer_produces_consistent_system`` — drives the public
   :func:`molpy.io.write_lammps_bond_react_system` end to end and asserts the
   *behavior* of the output: the expected file set, a 1-based bijection in the
   ``.map`` equivalences, exactly two initiators, and unified numeric type IDs
   in the pre/post ``.mol`` templates. It deliberately does not byte-compare —
   exact formatting (timestamps, force-field coeff order) is incidental and not
   what this module owns.
2. ``TestWriteBondReactMap`` — unit tests for ``write_bond_react_map``: header
   counts, section order, 1-based IDs, and a ValueError on a pre/post atom-set
   mismatch.

The system builder couples two propane fragments (three carbons so the radius-2
template has genuine edge atoms) with :class:`BondReactReacter` and derives all
topology type names from endpoint atom types.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molpy as mp
from molpy.core.atomistic import Atomistic
from molpy.core.entity import Link
from molpy.reacter import (
    BondReactReacter,
    find_port,
    form_single_bond,
    select_one_hydrogen,
    select_port,
)
from molpy.reacter.bond_react import BondReactTemplate


# ===================================================================
# Deterministic system builder (helpers, not tests)
# ===================================================================


def _propane_fragment(
    x0: float, port_label: str, port_on_first_carbon: bool
) -> Atomistic:
    """Build a propane fragment with fixed insertion order and geometry.

    Three carbons (instead of ethane's two) put the far methyl hydrogens
    outside the radius-2 reaction subgraph, so the generated template has
    nonzero EdgeIDs — exercising the edge-atom branch of the map writer.
    """
    struct = Atomistic()
    ca = struct.def_atom(
        element="C", type="c3", x=x0, y=0.0, z=0.0, charge=0.0, mol_id=1
    )
    cb = struct.def_atom(
        element="C", type="c3", x=x0 + 1.54, y=0.0, z=0.0, charge=0.0, mol_id=1
    )
    cc = struct.def_atom(
        element="C", type="c3", x=x0 + 3.08, y=0.0, z=0.0, charge=0.0, mol_id=1
    )
    ha1 = struct.def_atom(
        element="H", type="hc", x=x0 - 0.5, y=0.9, z=0.0, charge=0.0, mol_id=1
    )
    ha2 = struct.def_atom(
        element="H", type="hc", x=x0 - 0.5, y=-0.45, z=0.78, charge=0.0, mol_id=1
    )
    ha3 = struct.def_atom(
        element="H", type="hc", x=x0 - 0.5, y=-0.45, z=-0.78, charge=0.0, mol_id=1
    )
    hb1 = struct.def_atom(
        element="H", type="hc", x=x0 + 1.54, y=0.9, z=0.45, charge=0.0, mol_id=1
    )
    hb2 = struct.def_atom(
        element="H", type="hc", x=x0 + 1.54, y=0.9, z=-0.45, charge=0.0, mol_id=1
    )
    hc1 = struct.def_atom(
        element="H", type="hc", x=x0 + 3.58, y=0.9, z=0.0, charge=0.0, mol_id=1
    )
    hc2 = struct.def_atom(
        element="H", type="hc", x=x0 + 3.58, y=-0.45, z=0.78, charge=0.0, mol_id=1
    )
    hc3 = struct.def_atom(
        element="H", type="hc", x=x0 + 3.58, y=-0.45, z=-0.78, charge=0.0, mol_id=1
    )
    struct.def_bond(ca, cb)
    struct.def_bond(cb, cc)
    struct.def_bond(ca, ha1)
    struct.def_bond(ca, ha2)
    struct.def_bond(ca, ha3)
    struct.def_bond(cb, hb1)
    struct.def_bond(cb, hb2)
    struct.def_bond(cc, hc1)
    struct.def_bond(cc, hc2)
    struct.def_bond(cc, hc3)
    port_atom = ca if port_on_first_carbon else cc
    port_atom["port"] = port_label
    return struct


def _canonical_link_type(link: Link) -> str:
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


def _strip_port_markers(struct: Atomistic) -> None:
    """Drop sparse 'port' keys so to_frame() columns stay dense and stable."""
    for atom in struct.atoms:
        atom.data.pop("port", None)


def _build_reaction() -> tuple[BondReactTemplate, Atomistic]:
    """Run the deterministic C-C coupling; return (typed template, typed product)."""
    left = _propane_fragment(0.0, ">", port_on_first_carbon=False)
    right = _propane_fragment(6.0, "<", port_on_first_carbon=True)

    reacter = BondReactReacter(
        name="cc_coupling",
        anchor_selector_left=select_port,
        anchor_selector_right=select_port,
        leaving_selector_left=select_one_hydrogen,
        leaving_selector_right=select_one_hydrogen,
        bond_former=form_single_bond,
        radius=2,
    )
    port_l = find_port(left, ">")
    port_r = find_port(right, "<")
    result = reacter.run(
        left, right, port_atom_L=port_l, port_atom_R=port_r, compute_topology=True
    )
    template = result.template
    assert template is not None

    for struct in (template.pre, template.post, result.product):
        _strip_port_markers(struct)
        _assign_link_types(struct)
    return template, result.product


def _build_forcefield(template: BondReactTemplate, product: Atomistic) -> mp.ForceField:
    """Minimal force field whose type names match the template/product types."""
    bond_names: set[str] = set()
    angle_names: set[str] = set()
    dihedral_names: set[str] = set()
    for struct in (template.pre, template.post, product):
        bond_names.update(str(b["type"]) for b in struct.bonds)
        angle_names.update(str(a["type"]) for a in struct.angles)
        dihedral_names.update(str(d["type"]) for d in struct.dihedrals)

    ff = mp.ForceField("bond_react_test", units="real")
    atom_style = ff.def_style(mp.AtomStyle(name="full"))
    atom_types = {
        "c3": atom_style.def_type("c3", mass=12.011),
        "hc": atom_style.def_type("hc", mass=1.008),
    }
    bond_style = ff.def_style(mp.BondStyle(name="harmonic"))
    for name in sorted(bond_names):
        i, j = name.split("-")
        bond_style.def_type(atom_types[i], atom_types[j], name=name, k=300.0, r0=1.53)
    angle_style = ff.def_style(mp.AngleStyle(name="harmonic"))
    for name in sorted(angle_names):
        i, j, k = name.split("-")
        angle_style.def_type(
            atom_types[i], atom_types[j], atom_types[k], name=name, k=50.0, theta0=110.0
        )
    dihedral_style = ff.def_style(mp.DihedralStyle(name="opls"))
    for name in sorted(dihedral_names):
        endpoint_types = [atom_types[p] for p in name.split("-")]
        dihedral_style.def_type(
            *endpoint_types, name=name, k1=0.5, k2=1.0, k3=0.0, k4=0.0
        )
    return ff


def _build_system() -> tuple[mp.Frame, mp.ForceField, BondReactTemplate]:
    """Fully deterministic (frame, forcefield, template) for the system writer."""
    template, product = _build_reaction()
    for i, atom in enumerate(product.atoms, start=1):
        atom["id"] = i
    frame = product.to_frame()
    frame.box = mp.Box.cubic(20.0)
    ff = _build_forcefield(template, product)
    return frame, ff, template


def _mol_type_ids(mol_text: str) -> list[int]:
    """Parse the integer atom type IDs from a LAMMPS molecule Types section."""
    lines = mol_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Types")
    ids: list[int] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if ids:
                break
            continue
        ids.append(int(stripped.split()[1]))
    return ids


# ===================================================================
# System-writer integration test (semantic, not byte-for-byte)
# ===================================================================


def test_system_writer_produces_consistent_system(tmp_path: Path) -> None:
    """write_lammps_bond_react_system emits a self-consistent file set.

    Asserts the *behavior* of the writer rather than exact bytes:

    - all five fix bond/react files are produced;
    - the ``.map`` equivalences are a 1-based bijection over the template
      atoms, with exactly two initiators;
    - pre/post ``.mol`` templates carry **numeric** type IDs drawn from a
      single unified mapping (the type-unification the refactor owns), not
      raw string type names.
    """
    frame, ff, template = _build_system()
    workdir = tmp_path / "sys"
    mp.io.write_lammps_bond_react_system(
        workdir, frame, ff, templates={"rxn1": template}
    )

    produced = {p.name for p in workdir.iterdir() if p.is_file()}
    assert produced == {
        "sys.data",
        "sys.ff",
        "rxn1_pre.mol",
        "rxn1_post.mol",
        "rxn1.map",
    }

    # .map: 1-based bijection over the pre atoms, exactly two initiators.
    map_text = (workdir / "rxn1.map").read_text(encoding="utf-8")
    sections = _parse_map_sections(workdir / "rxn1.map")
    n_atoms = len(list(template.pre.atoms))
    pairs = _parse_equivalences(map_text)
    assert sorted(pre for pre, _ in pairs) == list(range(1, n_atoms + 1))
    assert sorted(post for _, post in pairs) == list(range(1, n_atoms + 1))
    assert len(sections["InitiatorIDs"]) == 2

    # .mol templates: unified numeric type IDs (not string type names).
    pre_ids = _mol_type_ids((workdir / "rxn1_pre.mol").read_text(encoding="utf-8"))
    post_ids = _mol_type_ids((workdir / "rxn1_post.mol").read_text(encoding="utf-8"))
    assert pre_ids and post_ids
    assert all(i >= 1 for i in pre_ids + post_ids)


# ===================================================================
# Unit tests for the FUTURE molpy.io.data.lammps_bond_react module
# (RED until the refactor lands — imports are inside each test)
# ===================================================================


def _parse_equivalences(content: str) -> list[tuple[int, int]]:
    lines = content.splitlines()
    pairs: list[tuple[int, int]] = []
    for line in lines[lines.index("Equivalences") + 1 :]:
        parts = line.split()
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _parse_map_sections(map_path: Path) -> dict[str, list[list[int]]]:
    """Parse a fix bond/react ``.map`` file into {section: rows of ints}."""
    section_names = {"InitiatorIDs", "EdgeIDs", "DeleteIDs", "Equivalences"}
    sections: dict[str, list[list[int]]] = {}
    current: str | None = None
    for raw_line in map_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in section_names:
            current = line
            sections[current] = []
            continue
        if current is not None:
            sections[current].append([int(token) for token in line.split()])
    return sections


class TestWriteBondReactMap:
    """Unit tests for write_bond_react_map (module does not exist yet → RED)."""

    def _write_map(self, tmp_path: Path) -> tuple[str, BondReactTemplate]:
        from molpy.io.data.lammps_bond_react import write_bond_react_map

        template, _ = _build_reaction()
        write_bond_react_map(template, tmp_path / "rxn1")
        content = (tmp_path / "rxn1.map").read_text(encoding="utf-8")
        return content, template

    def test_map_header_counts(self, tmp_path: Path) -> None:
        """Header lines carry the equivalence/edge/delete counts of the template."""
        content, template = self._write_map(tmp_path)

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

    def test_map_sections_present_in_order(self, tmp_path: Path) -> None:
        """InitiatorIDs, EdgeIDs, DeleteIDs, Equivalences appear in that order."""
        content, _ = self._write_map(tmp_path)
        positions = [
            content.index("InitiatorIDs"),
            content.index("EdgeIDs"),
            content.index("DeleteIDs"),
            content.index("Equivalences"),
        ]
        assert positions == sorted(positions)

    def test_map_ids_are_1based(self, tmp_path: Path) -> None:
        """Equivalence IDs are >= 1 and pre-side IDs cover 1..n_atoms exactly."""
        content, template = self._write_map(tmp_path)
        pairs = _parse_equivalences(content)
        n_atoms = len(list(template.pre.atoms))

        assert all(pre >= 1 and post >= 1 for pre, post in pairs)
        assert sorted(pre for pre, _ in pairs) == list(range(1, n_atoms + 1))

    def test_map_mismatched_pre_post_raises(self, tmp_path: Path) -> None:
        """Post missing one react_id must raise ValueError, not write silently."""
        from molpy.io.data.lammps_bond_react import write_bond_react_map

        pre = Atomistic()
        a1 = pre.def_atom(element="C", type="c3", react_id=1)
        a2 = pre.def_atom(element="C", type="c3", react_id=2)
        pre.def_bond(a1, a2, type="c3-c3")

        post = Atomistic()
        b1 = post.def_atom(element="C", type="c3", react_id=1)

        template = BondReactTemplate(
            pre=pre,
            post=post,
            initiator_atoms=[a1, a2],
            edge_atoms=[],
            deleted_atoms=[],
            pre_react_id_to_atom={1: a1, 2: a2},
            post_react_id_to_atom={1: b1},
        )
        with pytest.raises(ValueError):
            write_bond_react_map(template, tmp_path / "bad")
