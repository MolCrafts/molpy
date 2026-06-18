#!/usr/bin/env python3
"""Generate OPLS-AA typifier parity fixtures (molrs spec ``opls-typifier-03-parity``).

Runs molpy's :class:`molpy.typifier.OplsTypifier` over a fixed real-molecule set
and dumps, per molecule, a self-contained JSON ground-truth file:

* the molecule definition (per-atom element + xyz, and bonds as ``[i, j, order]``
  index pairs) so the molrs Rust test rebuilds the *exact same* input topology;
* the per-atom assigned ``opls_NNN`` type;
* every bond / angle / dihedral with its assigned force-field *type name* plus the
  numeric params, **expressed in molrs canonical units** so the Rust parity test
  can compare directly without re-deriving unit conventions.

Unit / convention reconciliation (molpy reader -> molrs reader), verified
empirically against the two ``OplsXmlReader`` implementations:

* bond   : ``r0`` identical (Å); ``k0_molrs = 2 * k_molpy`` (molpy folds the
  ``0.5`` prefactor into ``k``, molrs does not).
* angle  : ``k0_molrs = 2 * k_molpy``; ``theta0_molrs = theta0_molpy * pi/180``
  (molpy stores degrees, molrs stores radians).
* dihedral: ``f{n}_molrs == c{n}_molpy`` (identical units kcal/mol; molpy keys
  them ``c1..c4``, molrs keys ``f1..f4``).

Both the reconciled (molrs-canonical) and the raw molpy values are recorded.

The force field read is the **molrs** ``tests-data/xml/oplsaa.xml`` (NOT molpy's
bundled copy — the two differ in content), so the parity test isolates *engine*
semantics rather than force-field-version drift. molpy special-cases any path
ending in ``oplsaa.xml`` through its OPLS reader, so passing the molrs path
applies the identical kJ->kcal / nm->Å conversions.

Usage::

    python scripts/gen_opls_fixtures.py [--xml PATH] [--out DIR] [--mol2-dir DIR]

Defaults resolve the molrs checkout relative to this molpy repo
(``../molrs``). Output goes to ``tests-data/opls/`` under the molrs checkout
(the binding-neutral data dir). The molrs parity test reads from there.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from molpy.core.atomistic import Angle, Atomistic, Dihedral
from molpy.io import read_xml_forcefield
from molpy.typifier import OplsTypifier

# --- molrs <- molpy reconciliation factors ---------------------------------
_DEG_TO_RAD = math.pi / 180.0
# molrs bond/angle force constants are 2x molpy's (the 0.5 prefactor convention).
_FK_MOLRS_OVER_MOLPY = 2.0


# ===========================================================================
# Molecule set
# ===========================================================================
#
# Each builder returns an (Atomistic, label) describing a real molecule with
# EXPLICIT hydrogens. The set is the PEO-relevant agreeable chemistry
# (aliphatic / alcohol / ether) plus aromatics for divergence characterization.
# `gap=True` marks the documented C/c known-gap aromatics.


@dataclass
class MoleculeCase:
    name: str
    builder: Any
    category: str
    gap: bool = False
    notes: str = ""
    coverage: list[str] = field(default_factory=list)


def _sybyl_element(sybyl_type: str, name: str) -> str:
    """Element symbol from a SYBYL atom type / atom name.

    Mirrors the minimal mol2 element logic in the molrs ``opls.rs`` test:
    drop the ``.`` suffix and trailing digits, prefer a valid two-letter symbol
    (``Cl``), else the leading letter.
    """
    base = "".join(ch for ch in sybyl_type.split(".")[0] if ch.isalpha())
    two = base[:2].capitalize()
    one = base[:1].capitalize()
    if two in _ELEMENTS:
        return two
    if one in _ELEMENTS:
        return one
    nm = "".join(ch for ch in name if ch.isalpha())
    return (
        nm[:2].capitalize() if nm[:2].capitalize() in _ELEMENTS else nm[:1].capitalize()
    )


_ELEMENTS = {
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Na",
    "Li",
    "K",
}


def _sybyl_order(tok: str) -> float:
    return {"1": 1.0, "2": 2.0, "3": 3.0, "ar": 1.5, "am": 1.0}.get(tok, 1.0)


def from_mol2(path: Path) -> Atomistic:
    """Minimal single-molecule Tripos mol2 loader (matches molrs ``opls.rs``)."""
    g = Atomistic()
    atoms: list = []
    section = ""
    seen = False
    for raw in path.read_text().splitlines():
        t = raw.strip()
        if t.startswith("@<TRIPOS>"):
            name = t[len("@<TRIPOS>") :]
            if name == "MOLECULE":
                if seen:
                    break
                seen = True
            section = name if name in ("ATOM", "BOND") else ""
            continue
        if not t:
            continue
        if section == "ATOM":
            f = t.split()
            if len(f) < 6:
                continue
            x, y, z = float(f[2]), float(f[3]), float(f[4])
            elem = _sybyl_element(f[5], f[1])
            atoms.append(g.def_atom(element=elem, x=x, y=y, z=z))
        elif section == "BOND":
            f = t.split()
            if len(f) < 4:
                continue
            i, j = int(f[1]) - 1, int(f[2]) - 1
            if 0 <= i < len(atoms) and 0 <= j < len(atoms):
                g.def_bond(atoms[i], atoms[j], order=_sybyl_order(f[3]))
    return g


def _chain(
    g: Atomistic,
    specs: list[tuple[str, float, float, float]],
    bonds: list[tuple[int, int]],
) -> Atomistic:
    atoms = [g.def_atom(element=e, x=x, y=y, z=z) for (e, x, y, z) in specs]
    for i, j in bonds:
        g.def_bond(atoms[i], atoms[j], order=1.0)
    return g


def build_propane() -> Atomistic:
    """Propane CH3-CH2-CH3 — aliphatic; central CT exercises CT-CT-CT angle and
    X-CT-CT-X wildcard-end dihedrals."""
    g = Atomistic()
    specs = [
        ("C", 0.0, 0.0, 0.0),
        ("C", 1.53, 0.0, 0.0),
        ("C", 2.30, 1.33, 0.0),
        ("H", -0.4, 1.0, 0.0),
        ("H", -0.4, -0.5, 0.87),
        ("H", -0.4, -0.5, -0.87),
        ("H", 1.93, -0.55, 0.87),
        ("H", 1.93, -0.55, -0.87),
        ("H", 1.93, 1.9, 0.0),
        ("H", 3.4, 1.27, 0.0),
        ("H", 2.0, 1.9, 0.87),
    ]
    bonds = [
        (0, 1),
        (1, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 6),
        (1, 7),
        (2, 8),
        (2, 9),
        (2, 10),
    ]
    return _chain(g, specs, bonds)


def build_methanol() -> Atomistic:
    """Methanol CH3-OH — alcohol; O=opls_154, hydroxyl H=opls_155 and methyl
    H=opls_156 are %opls_154-LAYERED + overrides-decided types."""
    g = Atomistic()
    specs = [
        ("C", 0.0, 0.0, 0.0),
        ("O", 1.4, 0.0, 0.0),
        ("H", 2.0, 0.0, 0.0),
        ("H", -0.5, 0.9, 0.0),
        ("H", -0.5, -0.9, 0.0),
        ("H", -0.5, 0.0, 0.9),
    ]
    bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)]
    return _chain(g, specs, bonds)


def build_ethanol() -> Atomistic:
    """Ethanol CH3-CH2-OH — alcohol on a 2-carbon chain."""
    g = Atomistic()
    specs = [
        ("C", 0.0, 0.0, 0.0),
        ("C", 1.5, 0.0, 0.0),
        ("O", 2.1, 1.2, 0.0),
        ("H", 1.7, 1.9, 0.0),
        ("H", -0.4, 1.0, 0.0),
        ("H", -0.4, -0.5, 0.87),
        ("H", -0.4, -0.5, -0.87),
        ("H", 1.9, -0.55, 0.87),
        ("H", 1.9, -0.55, -0.87),
    ]
    bonds = [(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (0, 6), (1, 7), (1, 8)]
    return _chain(g, specs, bonds)


def build_dimethyl_ether() -> Atomistic:
    """Dimethyl ether CH3-O-CH3 — the simplest ether; ether O = opls_180 (OS)."""
    g = Atomistic()
    specs = [
        ("C", 0.0, 0.0, 0.0),
        ("O", 1.4, 0.0, 0.0),
        ("C", 2.1, 1.2, 0.0),
        ("H", -0.4, 1.0, 0.0),
        ("H", -0.4, -0.5, 0.87),
        ("H", -0.4, -0.5, -0.87),
        ("H", 1.7, 1.9, 0.87),
        ("H", 1.7, 1.9, -0.87),
        ("H", 3.2, 1.1, 0.0),
    ]
    bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (2, 6), (2, 7), (2, 8)]
    return _chain(g, specs, bonds)


def build_peo_fragment() -> Atomistic:
    """PEO fragment CH3-O-CH2-CH2-O-CH3 (1,2-dimethoxyethane) — the canonical
    PEO-electrolyte repeat unit; exercises ether O (OS), the C-C-O-C and
    X-CT-CT-X wildcard dihedrals along the glycol backbone."""
    g = Atomistic()
    specs = [
        ("C", 0.0, 0.0, 0.0),  # 0 methyl C
        ("O", 1.4, 0.0, 0.0),  # 1 ether O
        ("C", 2.1, 1.2, 0.0),  # 2 CH2
        ("C", 3.6, 1.2, 0.0),  # 3 CH2
        ("O", 4.3, 0.0, 0.0),  # 4 ether O
        ("C", 5.7, 0.0, 0.0),  # 5 methyl C
        ("H", -0.4, 1.0, 0.0),
        ("H", -0.4, -0.5, 0.87),
        ("H", -0.4, -0.5, -0.87),  # 6,7,8 on C0
        ("H", 1.7, 1.9, 0.87),
        ("H", 1.7, 1.9, -0.87),  # 9,10 on C2
        ("H", 4.0, 1.9, 0.87),
        ("H", 4.0, 1.9, -0.87),  # 11,12 on C3
        ("H", 6.1, 1.0, 0.0),
        ("H", 6.1, -0.5, 0.87),
        ("H", 6.1, -0.5, -0.87),  # 13,14,15 on C5
    ]
    bonds = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (2, 9),
        (2, 10),
        (3, 11),
        (3, 12),
        (5, 13),
        (5, 14),
        (5, 15),
    ]
    return _chain(g, specs, bonds)


def build_benzene() -> Atomistic:
    """Benzene C6H6 — aromatic; molpy types C=opls_145, H=opls_146 via the
    uppercase-C atomic-number SMARTS. KNOWN-GAP vs molrs (aromatic c semantics)."""
    g = Atomistic()
    r = 1.39
    specs = []
    for k in range(6):
        ang = math.pi / 3 * k
        specs.append(("C", r * math.cos(ang), r * math.sin(ang), 0.0))
    rh = 2.48
    for k in range(6):
        ang = math.pi / 3 * k
        specs.append(("H", rh * math.cos(ang), rh * math.sin(ang), 0.0))
    bonds = [(k, (k + 1) % 6) for k in range(6)] + [(k, 6 + k) for k in range(6)]
    # aromatic ring bonds order 1.5; C-H order 1.0
    g_atoms = [g.def_atom(element=e, x=x, y=y, z=z) for (e, x, y, z) in specs]
    for i, j in bonds[:6]:
        g.def_bond(g_atoms[i], g_atoms[j], order=1.5)
    for i, j in bonds[6:]:
        g.def_bond(g_atoms[i], g_atoms[j], order=1.0)
    return g


def build_toluene() -> Atomistic:
    """Toluene C6H5-CH3 — aromatic + methyl; ring C=opls_145/148, methyl
    C=opls_135/148-region. KNOWN-GAP vs molrs on the aromatic ring atoms."""
    g = Atomistic()
    r = 1.39
    ring = []
    for k in range(6):
        ang = math.pi / 3 * k
        ring.append((r * math.cos(ang), r * math.sin(ang), 0.0))
    atoms = []
    for x, y, z in ring:
        atoms.append(("C", x, y, z))
    # H on ring carbons 1..5 (carbon 0 carries the methyl)
    rh = 2.48
    h_ring = []
    for k in range(1, 6):
        ang = math.pi / 3 * k
        h_ring.append(("H", rh * math.cos(ang), rh * math.sin(ang), 0.0))
    # methyl carbon off ring carbon 0
    methyl_c = ("C", r + 1.5, 0.0, 0.0)
    methyl_h = [
        ("H", r + 1.9, 1.0, 0.0),
        ("H", r + 1.9, -0.5, 0.87),
        ("H", r + 1.9, -0.5, -0.87),
    ]
    specs = atoms + h_ring + [methyl_c] + methyl_h
    g_atoms = [g.def_atom(element=e, x=x, y=y, z=z) for (e, x, y, z) in specs]
    # ring bonds (aromatic)
    for k in range(6):
        g.def_bond(g_atoms[k], g_atoms[(k + 1) % 6], order=1.5)
    # ring H bonds (carbons 1..5 -> H atoms at index 6..10)
    for idx, k in enumerate(range(1, 6)):
        g.def_bond(g_atoms[k], g_atoms[6 + idx], order=1.0)
    # methyl C at index 11, its H at 12,13,14
    g.def_bond(g_atoms[0], g_atoms[11], order=1.0)
    for h in (12, 13, 14):
        g.def_bond(g_atoms[11], g_atoms[h], order=1.0)
    return g


def molecule_cases(mol2_dir: Path) -> list[MoleculeCase]:
    return [
        MoleculeCase(
            "ethane",
            lambda: from_mol2(mol2_dir / "ethane.mol2"),
            "aliphatic",
            notes="real mol2; CT carbons + HC hydrogens",
        ),
        MoleculeCase(
            "propane",
            build_propane,
            "aliphatic",
            coverage=["wildcard_dihedral"],
            notes="X-CT-CT-X wildcard-end dihedral + CT-CT-CT angle",
        ),
        MoleculeCase(
            "methanol",
            build_methanol,
            "alcohol",
            coverage=["layered_type", "overrides"],
            notes="O=opls_154 (layer0); hydroxyl-H opls_155 / methyl-H opls_156 "
            "are %opls_154-LAYERED + overrides-decided",
        ),
        MoleculeCase(
            "ethanol",
            build_ethanol,
            "alcohol",
            coverage=["layered_type"],
            notes="alcohol on a 2-carbon chain",
        ),
        MoleculeCase(
            "dimethyl_ether",
            build_dimethyl_ether,
            "ether",
            notes="simplest ether; ether O",
        ),
        MoleculeCase(
            "peo_fragment",
            build_peo_fragment,
            "ether",
            coverage=["wildcard_dihedral"],
            notes="1,2-dimethoxyethane: PEO repeat unit; X-CT-CT-X + C-C-O-C dihedrals",
        ),
        MoleculeCase(
            "benzene",
            build_benzene,
            "aromatic",
            gap=True,
            notes="KNOWN-GAP: molpy uppercase-C atomic-number SMARTS types aromatic "
            "ring (opls_145/146); molrs distinguishes aromatic c",
        ),
        MoleculeCase(
            "toluene",
            build_toluene,
            "aromatic",
            gap=True,
            notes="KNOWN-GAP: aromatic ring carbons; methyl region",
        ),
    ]


# ===========================================================================
# Ground-truth extraction
# ===========================================================================


def _round(v: Any, nd: int = 8) -> Any:
    if isinstance(v, float):
        return round(v, nd)
    return v


def _atom_index_map(typed: Atomistic) -> dict[int, int]:
    """Map each atom's stable molrs ``handle`` to its 0-based iteration index.

    ``id()`` is unstable across atom proxies (each ``bond.itom`` is a fresh
    Python wrapper over the same underlying handle), so the handle is the only
    reliable key tying a bonded-term endpoint back to its atom position.
    """
    return {a.handle: i for i, a in enumerate(typed.atoms)}


def _term_indices(idx: dict[int, int], endpoints) -> list[int]:
    return [idx[ep.handle] for ep in endpoints]


def bond_record(idx: dict[int, int], bond) -> dict[str, Any]:
    k = bond.data.get("k")
    r0 = bond.data.get("r0")
    return {
        "atoms": [idx[bond.itom.handle], idx[bond.jtom.handle]],
        "type": bond.data.get("type"),
        # molrs-canonical
        "r0": _round(r0) if r0 is not None else None,
        "k0": _round(k * _FK_MOLRS_OVER_MOLPY) if k is not None else None,
        # raw molpy
        "molpy_k": _round(k) if k is not None else None,
    }


def angle_record(idx: dict[int, int], angle) -> dict[str, Any]:
    k = angle.data.get("k")
    theta0_deg = angle.data.get("theta0")
    return {
        "atoms": _term_indices(idx, angle.endpoints),
        "type": angle.data.get("type"),
        # molrs-canonical: radians, 2x force constant
        "theta0": _round(theta0_deg * _DEG_TO_RAD) if theta0_deg is not None else None,
        "k0": _round(k * _FK_MOLRS_OVER_MOLPY) if k is not None else None,
        # raw molpy
        "molpy_theta0_deg": _round(theta0_deg) if theta0_deg is not None else None,
        "molpy_k": _round(k) if k is not None else None,
    }


def dihedral_record(idx: dict[int, int], dih) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "atoms": _term_indices(idx, dih.endpoints),
        "type": dih.data.get("type"),
    }
    # molpy c1..c4 == molrs f1..f4 (same units)
    for n in (1, 2, 3, 4):
        c = dih.data.get(f"c{n}")
        rec[f"f{n}"] = _round(c) if c is not None else None
    return rec


def molecule_definition(typed: Atomistic, source: Atomistic) -> dict[str, Any]:
    """Element + xyz per atom and bonds as ``[i, j, order]`` from the (untyped)
    source structure, so the molrs test rebuilds an identical topology."""
    src_idx = {a.handle: i for i, a in enumerate(source.atoms)}
    atoms = [
        {
            "element": a.get("element"),
            "x": _round(a.get("x", 0.0)),
            "y": _round(a.get("y", 0.0)),
            "z": _round(a.get("z", 0.0)),
        }
        for a in source.atoms
    ]
    bonds = []
    for b in source.bonds:
        bonds.append(
            [
                src_idx[b.itom.handle],
                src_idx[b.jtom.handle],
                _round(b.data.get("order", 1.0)),
            ]
        )
    return {"atoms": atoms, "bonds": bonds}


def extract(case: MoleculeCase, ff) -> dict[str, Any]:
    source = case.builder()
    topo = source.get_topo(gen_angle=True, gen_dihe=True)
    # Non-strict bonded typing: a term molpy cannot match is left unparameterized
    # rather than raising. This mirrors the molrs lenient (NoMatch::Skip) policy
    # the parity test uses, and is exactly the ac-004 contract — "any term molpy
    # leaves unparameterized is also unparameterized in molrs (no silent
    # estimation)". Atom typing stays strict-free too (untyped atoms surface as
    # null types and are reported, never fabricated).
    typed = OplsTypifier(ff, strict_typing=False).typify(topo)
    idx = _atom_index_map(typed)

    atom_types = [a.get("type") for a in typed.atoms]

    bonds = [bond_record(idx, b) for b in typed.bonds]
    angles = [angle_record(idx, a) for a in topo_links(typed, Angle)]
    dihedrals = [dihedral_record(idx, d) for d in topo_links(typed, Dihedral)]

    return {
        "name": case.name,
        "category": case.category,
        "known_gap": case.gap,
        "coverage": case.coverage,
        "notes": case.notes,
        "source": "molpy.typifier.OplsTypifier",
        "ff_xml": "oplsaa.xml (molrs tests-data/xml/oplsaa.xml)",
        "units": {
            "r0": "angstrom",
            "theta0": "radian",
            "k0": "kcal/mol per (A^2 | rad^2); = 2 * molpy k",
            "f1..f4": "kcal/mol (== molpy c1..c4)",
        },
        "molecule": molecule_definition(typed, source),
        "atom_types": atom_types,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
    }


def topo_links(typed: Atomistic, link_type):
    return list(typed.links.bucket(link_type))


# ===========================================================================
# CLI
# ===========================================================================


def main() -> int:
    here = Path(__file__).resolve()
    molpy_root = here.parent.parent
    molrs_root = (molpy_root.parent / "molrs").resolve()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--xml",
        type=Path,
        default=molrs_root / "tests-data" / "xml" / "oplsaa.xml",
        help="OPLS-AA XML force field (default: molrs tests-data copy)",
    )
    ap.add_argument(
        "--mol2-dir",
        type=Path,
        default=molrs_root / "tests-data" / "mol2",
        help="directory of source mol2 files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=molrs_root / "tests-data" / "opls",
        help="output fixture directory (default: tests-data/opls/)",
    )
    args = ap.parse_args()

    if not args.xml.exists():
        raise SystemExit(f"oplsaa.xml not found: {args.xml}")

    ff = read_xml_forcefield(str(args.xml.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)

    cases = molecule_cases(args.mol2_dir)
    written = []
    for case in cases:
        data = extract(case, ff)
        path = args.out / f"{case.name}.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        n_typed = sum(1 for t in data["atom_types"] if t)
        written.append((case.name, len(data["atom_types"]), n_typed, case.gap))
        print(
            f"  wrote {path.name}: {n_typed}/{len(data['atom_types'])} atoms typed"
            f"{'  [KNOWN-GAP aromatic]' if case.gap else ''}"
        )

    # A small manifest aids the Rust test in iterating + reporting.
    manifest = {
        "spec": "opls-typifier-03-parity",
        "ff_xml": "oplsaa.xml",
        "molecules": [
            {"name": n, "n_atoms": na, "n_typed": nt, "known_gap": gap}
            for (n, na, nt, gap) in written
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote manifest.json ({len(written)} molecules)")
    print(f"fixtures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
