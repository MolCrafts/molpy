"""Golden-file and unit tests for LAMMPS fix bond/react serialization.

Covers two refactor-protection layers for spec ``builder-reacter-02-template-io``:

1. ``test_system_writer_matches_golden`` (integration) — byte-for-byte lock on
   every file produced by :func:`molpy.io.write_lammps_bond_react_system` so the
   upcoming move of map/mol serialization into ``molpy.io.data.lammps_bond_react``
   cannot change output.
2. ``TestWriteBondReactMap`` (basics / edge cases / domain layout) — unit tests
   for the FUTURE ``molpy.io.data.lammps_bond_react.write_bond_react_map``
   function. These are RED until the refactor lands; imports are inside each
   test so the golden test still collects and runs today.

The deterministic system builder constructs two propane fragments (ethane-like
alkanes extended to three carbons so the radius-2 template has genuine edge
atoms), couples them with :class:`BondReactReacter` (radius 2, one H lost per
side), and derives all topology type names from endpoint atom types — no
randomness, no dict/set-order dependence beyond CPython's deterministic
small-int hashing.
"""

from __future__ import annotations

import os
from datetime import datetime
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

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "bond_react"
_FROZEN_NOW = datetime(2026, 1, 1, 0, 0, 0)


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

    ff = mp.ForceField("bond_react_golden", units="real")
    atom_style = ff.def_style(mp.AtomStyle("full"))
    atom_types = {
        "c3": atom_style.def_type("c3", mass=12.011),
        "hc": atom_style.def_type("hc", mass=1.008),
    }
    bond_style = ff.def_style(mp.BondStyle("harmonic"))
    for name in sorted(bond_names):
        i, j = name.split("-")
        bond_style.def_type(atom_types[i], atom_types[j], name=name, k=300.0, r0=1.53)
    angle_style = ff.def_style(mp.AngleStyle("harmonic"))
    for name in sorted(angle_names):
        i, j, k = name.split("-")
        angle_style.def_type(
            atom_types[i], atom_types[j], atom_types[k], name=name, k=50.0, theta0=110.0
        )
    dihedral_style = ff.def_style(mp.DihedralStyle("opls"))
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


class _FrozenDatetime:
    """Stand-in for datetime whose now() returns a fixed instant.

    The LAMMPS data and molecule writers embed ``datetime.now()`` in their
    header comments; freezing it keeps the golden comparison byte-stable.
    """

    @staticmethod
    def now() -> datetime:
        return _FROZEN_NOW


@pytest.fixture
def frozen_writer_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    import molpy.io.data.lammps as lammps_data
    import molpy.io.data.lammps_molecule as lammps_molecule

    monkeypatch.setattr(lammps_data, "datetime", _FrozenDatetime)
    monkeypatch.setattr(lammps_molecule, "datetime", _FrozenDatetime)


# ===================================================================
# Golden-file integration test (passes on current HEAD)
# ===================================================================


def test_system_writer_matches_golden(
    tmp_path: Path, frozen_writer_clock: None
) -> None:
    """write_lammps_bond_react_system output matches the golden fixtures.

    Golden fixtures live in ``tests/test_io/test_data/golden/bond_react/`` and
    were generated with the pre-refactor HEAD (commit 4849ada, the bb18749-era
    serialization path where BondReactTemplate.write_map lives in
    molpy/reacter/bond_react.py). The .map/.mol section layout was eyeballed
    against the LAMMPS fix bond/react docs format
    (https://docs.lammps.org/fix_bond_react.html).

    Comparison policy:

    - ``sys.data``, ``rxn1_pre.mol``, ``rxn1_post.mol``, ``rxn1.map`` — strict
      byte-for-byte (writer header timestamps are frozen via fixture). These
      are the files the lammps_bond_react refactor touches.
    - ``sys.ff`` — non-blank line **multiset** (order- and blank-insensitive).
      The ``*_coeff`` line order comes from ``TypeBucket``'s ``set`` storage,
      whose iteration order depends on ``hash((cls, name))`` — class hashes
      are id-based, so neither coeff order nor the presence of a trailing
      blank separator line can be pinned across interpreter runs without
      modifying production code (verified: pre-refactor src also flaps on the
      trailing blank). Every substantive line must still match exactly.
      The refactor does not touch the .ff writer.

    Regeneration: run with ``MOLPY_REGEN_GOLDEN=1`` to overwrite the golden
    directory with freshly produced files (the test then passes trivially);
    re-run without the variable to confirm a stable pass.
    """
    frame, ff, template = _build_system()
    workdir = tmp_path / "sys"
    mp.io.write_lammps_bond_react_system(
        workdir, frame, ff, templates={"rxn1": template}
    )
    produced = {p.name: p.read_bytes() for p in workdir.iterdir() if p.is_file()}
    assert produced, "system writer produced no files"

    if os.environ.get("MOLPY_REGEN_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for stale in GOLDEN_DIR.iterdir():
            if stale.is_file():
                stale.unlink()
        for name, payload in sorted(produced.items()):
            (GOLDEN_DIR / name).write_bytes(payload)
        return

    assert GOLDEN_DIR.is_dir(), (
        f"golden dir missing: {GOLDEN_DIR} — regenerate with MOLPY_REGEN_GOLDEN=1"
    )
    golden = {p.name: p.read_bytes() for p in GOLDEN_DIR.iterdir() if p.is_file()}
    assert set(produced) == set(golden), (
        f"file set mismatch: produced={sorted(produced)} golden={sorted(golden)}"
    )
    for name in sorted(produced):
        if name == "sys.ff":
            # Compare substantive lines order-insensitively (TypeBucket set
            # iteration order is not pinnable) and drop the incidental
            # version-stamp header (`# ... generated by molpy version X`).
            def _ff_lines(payload: bytes) -> list[bytes]:
                return [
                    li
                    for li in payload.splitlines()
                    if li.strip() and b"generated by molpy version" not in li
                ]

            assert sorted(_ff_lines(produced[name])) == sorted(
                _ff_lines(golden[name])
            ), "sys.ff line content differs from golden fixture"
        else:
            assert produced[name] == golden[name], f"{name} differs from golden fixture"


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
