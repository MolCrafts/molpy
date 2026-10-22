#!/usr/bin/env python3
"""OpenMM's own energies for molrs's OpenMM-XML check (crates/molrs/src/ff/openmm_check.rs).

Builds three molecules with OpenMM's ``app.ForceField`` from XML files cut
out of real force fields, and prices them term by term on the Reference
platform (double precision), ``NoCutoff``, no dispersion correction:

- ``charmm``: ACE-ALA-NME with CHARMM36 (OpenMM's ``charmm36.xml``):
  Urey-Bradley, the harmonic ``CustomTorsionForce`` improper, CMAP,
  ``LennardJonesForce`` with ``sigma14``/``epsilon14``, plus one NBFixPair
  (NH1-O) added so NBFIX is exercised;
- ``amber``: ACE-ALA-NME with AMBER ff14SB (``amber14/protein.ff14SB.xml``):
  multi-term periodic propers, ``ordering="amber"`` wildcard impropers,
  NonbondedForce 1/2 and 5/6;
- ``opls``: 1-propanol with OPLS-AA (the foyer-style ``oplsaa.xml`` molpy
  shipped up to its 2026 typing hand-off, sha256 0151994…fe118): RB torsions
  and geometric mixing, applied the way foyer applies a ``combining_rule``
  (a geometric ``CustomNonbondedForce`` and geometric 1-4 exceptions).

For each case it writes ``<case>.xml`` (the force field: the rows of the
source file whose types or classes the molecule has, and one residue
template) and ``<case>.json`` (atoms, topology as OpenMM priced it, positions
in Å, per-term energies in kcal/mol) into the output directory, which the
Rust test reads. Re-run it after a change to the molecules; the energies it
prints are the ones the test pins.

    python scripts/openmm_xml_check.py --oplsaa <molpy>/data/forcefield/oplsaa.xml \
        [--out crates/molrs/src/ff/testdata/openmm] [--written <MOLRS_OPENMM_CHECK_DIR>]

``--written`` also prices the XML molrs writes back from each fixture (the
``report`` test writes ``<case>.written.xml`` there) with OpenMM.

Needs ``openmm`` (8.x), ``rdkit`` (for the conformers only) and molrs (the
unit factors, molrs's unit registry).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as u
from engine_check_tables import is_chain, split_system, unit_factor
from openmm import app
from rdkit import Chem
from rdkit.Chem import AllChem

FORCE_TAGS = (
    "HarmonicBondForce",
    "HarmonicAngleForce",
    "AmoebaUreyBradleyForce",
    "PeriodicTorsionForce",
    "RBTorsionForce",
    "CustomTorsionForce",
    "CMAPTorsionForce",
    "NonbondedForce",
    "LennardJonesForce",
)

# ACE-ALA-NME, SMILES heavy-atom order then RDKit's AddHs order.
DIPEPTIDE = "CC(=O)N[C@@H](C)C(=O)NC"
# (name, CHARMM36 type, CHARMM36 charge, ff14SB residue, ff14SB name) per atom.
DIPEPTIDE_ATOMS = [
    ("CH3", "CT3", -0.27, "ACE", "CH3"),
    ("C1", "C", 0.51, "ACE", "C"),
    ("O1", "O", -0.51, "ACE", "O"),
    ("N", "NH1", -0.47, "ALA", "N"),
    ("CA", "CT1", 0.07, "ALA", "CA"),
    ("CB", "CT3", -0.27, "ALA", "CB"),
    ("C", "C", 0.51, "ALA", "C"),
    ("O", "O", -0.51, "ALA", "O"),
    ("N2", "NH1", -0.47, "NME", "N"),
    ("C2", "CT3", -0.11, "NME", "CH3"),
    ("H31", "HA3", 0.09, "ACE", "HH31"),
    ("H32", "HA3", 0.09, "ACE", "HH32"),
    ("H33", "HA3", 0.09, "ACE", "HH33"),
    ("HN", "H", 0.31, "ALA", "H"),
    ("HA", "HB1", 0.09, "ALA", "HA"),
    ("HB1", "HA3", 0.09, "ALA", "HB1"),
    ("HB2", "HA3", 0.09, "ALA", "HB2"),
    ("HB3", "HA3", 0.09, "ALA", "HB3"),
    ("HN2", "H", 0.31, "NME", "H"),
    ("H21", "HA3", 0.09, "NME", "HH31"),
    ("H22", "HA3", 0.09, "NME", "HH32"),
    ("H23", "HA3", 0.09, "NME", "HH33"),
]
PROPANOL = "CCCO"
# (name, OPLS-AA type) per atom.
PROPANOL_ATOMS = [
    ("C1", "opls_135"),
    ("C2", "opls_136"),
    ("C3", "opls_157"),
    ("O", "opls_154"),
    ("H11", "opls_140"),
    ("H12", "opls_140"),
    ("H13", "opls_140"),
    ("H21", "opls_140"),
    ("H22", "opls_140"),
    ("H31", "opls_140"),
    ("H32", "opls_140"),
    ("HO", "opls_155"),
]


def conformer(smiles: str, seed: int) -> tuple[Chem.Mol, np.ndarray]:
    """An ETKDG + MMFF conformer, nudged off its minimum (Å)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    x = mol.GetConformer().GetPositions()
    rng = np.random.default_rng(seed)
    x = x + rng.uniform(-0.08, 0.08, x.shape) + 15.0
    return mol, x


def labels(row: ET.Element, n: int) -> list[str] | None:
    out = []
    for i in range(1, n + 1):
        v = row.get(f"type{i}", row.get(f"class{i}"))
        if v is None:
            return None
        out.append(v)
    return out


def arity(tag: str) -> int:
    return {"Bond": 2, "Angle": 3, "UreyBradley": 3, "Proper": 4, "Improper": 4, "Torsion": 5}.get(tag, 0)


def subset(sources: list[Path], types: set[str], residue: ET.Element, extra: list[str] = ()) -> str:
    """The rows of ``sources`` the atom ``types`` can use, and ``residue``."""
    trees = [ET.parse(p).getroot() for p in sources]
    root = ET.Element("ForceField")
    for k, v in trees[0].attrib.items():
        root.set(k, v)
    classes: set[str] = set()
    atom_types = ET.SubElement(root, "AtomTypes")
    for tree in trees:
        for t in tree.iter("Type"):
            if t.get("name") in types and tree.find("AtomTypes") is not None:
                atom_types.append(copy.deepcopy(t))
                classes.add(t.get("class"))
    known = types | classes | {""}
    for tag in FORCE_TAGS:
        for tree in trees:
            for sec in tree.findall(tag):
                out = ET.SubElement(root, tag, dict(sec.attrib))
                maps = []
                for row in sec:
                    if row.tag in ("Atom",):
                        if row.get("type", row.get("class")) in known:
                            out.append(copy.deepcopy(row))
                    elif row.tag == "NBFixPair":
                        if set(labels(row, 2) or []) <= known:
                            out.append(copy.deepcopy(row))
                    elif row.tag in ("UseAttributeFromResidue", "PerTorsionParameter", "GlobalParameter"):
                        out.append(copy.deepcopy(row))
                    elif row.tag == "Map":
                        maps.append(row)
                    else:
                        ends = labels(row, arity(row.tag))
                        if ends is not None and set(ends) <= known:
                            out.append(copy.deepcopy(row))
                if maps:
                    used = sorted({int(r.get("map")) for r in out.findall("Torsion")})
                    for new, old in enumerate(used):
                        out.insert(new, copy.deepcopy(maps[old]))
                    for r in out.findall("Torsion"):
                        r.set("map", str(used.index(int(r.get("map")))))
                rows = [r for r in out if r.tag not in ("UseAttributeFromResidue", "PerTorsionParameter")]
                if not rows:
                    root.remove(out)
    for line in extra:
        tag, _, row = line.partition(":")
        root.find(tag).append(ET.fromstring(row))
    residues = ET.SubElement(root, "Residues")
    residues.append(residue)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def residue_template(mol: Chem.Mol, atoms: list[tuple]) -> ET.Element:
    res = ET.Element("Residue", name="MOL")
    for name, ty, *rest in atoms:
        attrs = {"name": name, "type": ty}
        if rest and rest[0] is not None:
            attrs["charge"] = repr(rest[0])
        ET.SubElement(res, "Atom", attrs)
    for b in mol.GetBonds():
        ET.SubElement(
            res,
            "Bond",
            atomName1=atoms[b.GetBeginAtomIdx()][0],
            atomName2=atoms[b.GetEndAtomIdx()][0],
        )
    return res


def topology(mol: Chem.Mol, atoms: list[tuple]) -> app.Topology:
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("MOL", chain)
    made = [
        top.addAtom(name, app.Element.getBySymbol(a.GetSymbol()), res)
        for (name, *_), a in zip(atoms, mol.GetAtoms())
    ]
    for b in mol.GetBonds():
        top.addBond(made[b.GetBeginAtomIdx()], made[b.GetEndAtomIdx()])
    return top


def graph_terms(mol: Chem.Mol):
    n = mol.GetNumAtoms()
    nbr = [sorted(x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors()) for i in range(n)]
    bonds = sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds())
    angles = sorted({min((i, j, k), (k, j, i)) for j in range(n) for i, k in itertools.combinations(nbr[j], 2)})
    propers = set()
    for j, k in bonds:
        for i in nbr[j]:
            for l in nbr[k]:
                if i not in (j, k) and l not in (j, k) and i != l:
                    propers.add(min((i, j, k, l), (l, k, j, i)))
    return bonds, angles, sorted(propers)


def energies(system: mm.System, groups: dict, positions_nm) -> dict:
    ctx = mm.Context(system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions_nm)
    kj_per_kcal = unit_factor("kcal", "kJ")
    out = {}
    for g, name in groups.items():
        e = ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()
        out[name] = out.get(name, 0.0) + e.value_in_unit(u.kilojoule_per_mole) / kj_per_kcal
    total = ctx.getState(getEnergy=True).getPotentialEnergy()
    out["total"] = total.value_in_unit(u.kilojoule_per_mole) / kj_per_kcal
    return out


def price(xml_path: Path, mol, x, atoms, foyer_geometric: bool) -> dict:
    """OpenMM's per-term energies of the molecule under the XML at ``xml_path``."""
    ff = app.ForceField(str(xml_path))
    system = ff.createSystem(topology(mol, atoms), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    bonds, _, _ = graph_terms(mol)
    groups = split_system(system, bonds, foyer_geometric)
    return energies(system, groups, x / unit_factor("nm", "angstrom"))


def written(name, out_dir, written_dir, mol, x, atoms):
    """Price molrs's own XML of a case (``<written_dir>/<case>.written.xml``,
    from ``MOLRS_OPENMM_CHECK_DIR``) with OpenMM, the residue template of the
    fixture added, against the fixture's energies. No foyer patch: molrs's XML
    carries its geometric mixing as the ``<Script>`` ``createSystem`` runs."""
    root = ET.parse(written_dir / f"{name}.written.xml").getroot()
    root.append(ET.parse(out_dir / f"{name}.xml").getroot().find("Residues"))
    path = written_dir / f"{name}.written+residues.xml"
    ET.ElementTree(root).write(path)
    got = price(path, mol, x, atoms, False)
    want = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))["openmm_kcal"]
    print(f"== {name}: OpenMM on molrs's XML vs OpenMM on the source XML")
    for k, v in want.items():
        g = got.get(k, 0.0)
        print(f"  {k:9s} {g!r} rel {abs(g - v) / max(abs(v), 1e-300):.2e}")


def case(name, xml_text, mol, x, atoms, out_dir, foyer_geometric=False):
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml_text, encoding="utf-8")
    ff = app.ForceField(str(xml_path))
    top = topology(mol, atoms)
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    bonds, angles, propers = graph_terms(mol)
    bond_set = {frozenset(b) for b in bonds}
    impropers, cmaps = [], []
    for f in system.getForces():
        if isinstance(f, mm.PeriodicTorsionForce):
            for i in range(f.getNumTorsions()):
                t = tuple(f.getTorsionParameters(i)[:4])
                if not is_chain(bond_set, t) and t not in impropers:
                    impropers.append(t)
        elif isinstance(f, mm.CustomTorsionForce):
            for i in range(f.getNumTorsions()):
                t = tuple(f.getTorsionParameters(i)[:4])
                if t not in impropers:
                    impropers.append(t)
        elif isinstance(f, mm.CMAPTorsionForce):
            for i in range(f.getNumTorsions()):
                _, *ab = f.getTorsionParameters(i)
                cmaps.append((ab[0], ab[1], ab[2], ab[3], ab[7]))
    charges = []
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            charges = [f.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(f.getNumParticles())]
    groups = split_system(system, bonds, foyer_geometric)
    e = energies(system, groups, x / unit_factor("nm", "angstrom"))
    data = {
        "source": "scripts/openmm_xml_check.py, OpenMM " + mm.__version__ + ", Reference platform, NoCutoff",
        "types": [a[1] for a in atoms],
        "elements": [a.GetSymbol() for a in mol.GetAtoms()],
        "masses": [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())],
        "charges": charges,
        "positions": [[float(v) for v in p] for p in x],
        "bonds": bonds,
        "angles": angles,
        "dihedrals": propers,
        "impropers": impropers,
        "cmaps": cmaps,
        "openmm_kcal": e,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"== {name}")
    for k, v in e.items():
        print(f"  {k:9s} {v!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oplsaa", required=True, type=Path, help="molpy's data/forcefield/oplsaa.xml")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "crates/molrs/src/ff/testdata/openmm")
    ap.add_argument("--written", type=Path, help="a MOLRS_OPENMM_CHECK_DIR holding <case>.written.xml: price molrs's XML too")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = Path(app.__file__).parent / "data"

    mol, x = conformer(DIPEPTIDE, 7)
    assert mol.GetNumAtoms() == len(DIPEPTIDE_ATOMS)
    charmm_atoms = [(n, t, q) for n, t, q, *_ in DIPEPTIDE_ATOMS]
    xml = subset(
        [data / "charmm36.xml"],
        {t for _, t, _ in charmm_atoms},
        residue_template(mol, charmm_atoms),
        # An NBFIX row (not in CHARMM36's protein set) so NBFIX is priced.
        extra=['LennardJonesForce:<NBFixPair type1="NH1" type2="O" sigma="0.29" epsilon="0.6"/>'],
    )
    case("charmm", xml, mol, x, charmm_atoms, args.out)
    if args.written:
        written("charmm", args.out, args.written, mol, x, charmm_atoms)

    amber = ET.parse(data / "amber14" / "protein.ff14SB.xml").getroot()
    templates = {r.get("name"): {a.get("name"): (a.get("type"), float(a.get("charge"))) for a in r.findall("Atom")} for r in amber.iter("Residue")}
    amber_atoms = [(n, *templates[res][an]) for n, _, _, res, an in DIPEPTIDE_ATOMS]
    xml = subset([data / "amber14" / "protein.ff14SB.xml"], {t for _, t, _ in amber_atoms}, residue_template(mol, amber_atoms))
    case("amber", xml, mol, x, amber_atoms, args.out)
    if args.written:
        written("amber", args.out, args.written, mol, x, amber_atoms)

    mol, x = conformer(PROPANOL, 11)
    assert mol.GetNumAtoms() == len(PROPANOL_ATOMS)
    opls_atoms = [(n, t, None) for n, t in PROPANOL_ATOMS]
    xml = subset([args.oplsaa], {t for _, t, _ in opls_atoms}, residue_template(mol, opls_atoms))
    case("opls", xml, mol, x, opls_atoms, args.out, foyer_geometric=True)
    if args.written:
        written("opls", args.out, args.written, mol, x, opls_atoms)


if __name__ == "__main__":
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    main()
