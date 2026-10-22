"""What every engine check shares: molrs's energy TSVs, LAMMPS's thermo line
and force dump, unit factors, element-from-mass, the molecules of a bond list,
the residue-templated OpenMM topology of a molrs-written XML and the per-term
force groups of an OpenMM System.

Shared by scripts/ff_engine_codec_check.py, scripts/ff_equivalence_check.py,
scripts/openmm_xml_check.py and the LAMMPS / GROMACS steps of
scripts/*_check.sh (which import it with ``PYTHONPATH=scripts``). LAMMPS's log
and dump are parsed by molrs's own readers (``molpy.io.read_lammps_log``,
``molpy.io.read_lammps_dump_trajectory``), engine constants come from
``molpy.core.constants`` and every unit conversion from molrs's unit registry
(:func:`unit_factor`). molrs and OpenMM are imported inside the functions
that need them, so AmberTools' python (no molrs) can still load this module
for the sander step of ff_equivalence_check.

As a command, ``python scripts/engine_check_tables.py thermo LOG`` prints the
thermo line of LOG's last run as ``key=value`` words.
"""

from __future__ import annotations

import copy
import itertools
import math
import sys
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path


def read_energy_tsv(path: Path) -> dict[tuple[str, int, str, str], float]:
    """A ``case  config  engine  term  value`` table (``#`` lines are comments)
    as ``{(case, config, engine, term): value}``."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        case, k, engine, term, value = line.split("\t")
        out[(case, int(k), engine, term)] = float(value)
    return out


def lammps_thermo(path: Path | str) -> dict[str, float]:
    """The first thermo row of the last run of the LAMMPS log at ``path``
    (a ``run 0`` single point), as ``{column: value}``."""
    import molpy
    log = molpy.io.read_lammps_log(path)
    runs = [r.thermo for r in log.runs if r.thermo is not None and r.thermo.n_rows]
    if not runs:
        raise SystemExit(f"{path}: no thermo output")
    thermo = runs[-1]
    return {c: float(thermo[c][0]) for c in thermo.columns}


@cache
def _element_masses() -> list[tuple[str, float]]:
    from molpy.core import Element

    out, z = [], 1
    while True:
        try:
            e = Element(z)
        except (ValueError, KeyError):
            return out
        out.append((e.symbol, e.mass))
        z += 1


def element_of_mass(mass: float) -> str:
    """The element whose standard mass is nearest ``mass``, for an atom type
    that names no element."""
    return min(_element_masses(), key=lambda e: abs(e[1] - mass))[0]


def molecule_ids(n: int, bonds) -> list[int]:
    """Each of ``n`` atoms' molecule index: the connected components of
    ``bonds`` (molrs's ``Topology``), numbered in order of their first atom."""
    import numpy as np

    import molpy
    pairs = np.asarray(list(bonds), dtype=np.uint32).reshape(-1, 2)
    frame = molpy.core.Frame(
        {
            "atoms": {"x": np.zeros(n)},
            "bonds": {"atomi": pairs[:, 0].copy(), "atomj": pairs[:, 1].copy()},
        }
    )
    return molpy.core.Topology.from_frame(frame).connected_components().tolist()


@cache
def unit_factor(from_unit: str, to_unit: str) -> float:
    """The factor taking a value in ``from_unit`` to ``to_unit``
    (``value_to = value_from * factor``), from molrs's unit registry."""
    from molpy.core import UnitRegistry

    return UnitRegistry().factor(from_unit, to_unit)


def lammps_dump_forces(path: Path | str) -> list[list[float]]:
    """The per-atom forces (``fx fy fz``) of the first frame of the LAMMPS
    dump at ``path``, in atom-id order."""
    import molpy
    atoms = molpy.io.read_lammps_dump_trajectory(path).read_frame(0)["atoms"]
    rows = sorted(zip(atoms["id"], atoms["fx"], atoms["fy"], atoms["fz"]))
    return [[float(fx), float(fy), float(fz)] for _, fx, fy, fz in rows]


def openmm_residue_topology(xml_in: Path, xml_out: Path, types, bonds, charges=None):
    """The molrs-written OpenMM XML at ``xml_in`` with one residue template per
    molecule (atoms named ``A<i>``; ``charges`` per atom, or 0) written to
    ``xml_out``, and the OpenMM topology it matches.

    Returns ``(topology, order, root)``: the topology holds the atoms
    molecule by molecule (a residue template is a whole molecule), so its
    atom ``t`` is input atom ``order[t]``; ``root`` is the XML's root element.
    Every atom type names its element (OpenMM orders an AMBER improper's
    outer atoms by element), from its mass where the XML gives none."""
    from openmm import app

    root = ET.parse(xml_in).getroot()
    element = {}
    for t in root.iter("Type"):
        element[t.get("name")] = t.get("element") or element_of_mass(
            float(t.get("mass"))
        )
        t.set("element", element[t.get("name")])
    n = len(types)
    mol = molecule_ids(n, bonds)
    residues = ET.SubElement(root, "Residues")
    top = app.Topology()
    chain = top.addChain()
    made = [None] * n
    order = []
    for r in sorted(set(mol)):
        tmpl = ET.SubElement(residues, "Residue", name=f"R{r}")
        res = top.addResidue(f"R{r}", chain)
        for a in (a for a in range(n) if mol[a] == r):
            charge = "0" if charges is None else repr(charges[a])
            ET.SubElement(tmpl, "Atom", name=f"A{a}", type=types[a], charge=charge)
            made[a] = top.addAtom(
                f"A{a}", app.Element.getBySymbol(element[types[a]]), res
            )
            order.append(a)
        for i, j in bonds:
            if mol[i] == r:
                ET.SubElement(tmpl, "Bond", atomName1=f"A{i}", atomName2=f"A{j}")
    for i, j in bonds:
        top.addBond(made[i], made[j])
    ET.ElementTree(root).write(xml_out)
    return top, order, root


def is_chain(bonds: set[frozenset], t) -> bool:
    """Whether the atoms ``t`` are bonded in sequence (a proper dihedral)."""
    return all(frozenset(p) in bonds for p in itertools.pairwise(t))


def split_system(
    system, bonds, foyer_geometric: bool, impropers=None
) -> dict[int, str]:
    """Put every term family of the OpenMM ``system`` in its own force group;
    return ``{group: term}``.

    ``bonds`` (atom pairs) tells a real bond from a Urey-Bradley one in a
    ``HarmonicBondForce``; a ``PeriodicTorsionForce`` torsion is an improper
    when its atoms are in ``impropers`` (atom quadruples), or, without
    ``impropers``, when its atoms are not a bonded chain.
    ``foyer_geometric`` applies a ``combining_rule="geometric"`` the way foyer
    does (a geometric ``CustomNonbondedForce`` and geometric 1-4 sigmas)."""
    import openmm as mm

    bonds = {frozenset(b) for b in bonds}
    if impropers is not None:
        impropers = {frozenset(t) for t in impropers}

    def improper(t) -> bool:
        if impropers is None:
            return not is_chain(bonds, t)
        return frozenset(t) in impropers

    # The System owns its forces: copy them before removing them.
    forces = [copy.deepcopy(f) for f in system.getForces()]
    while system.getNumForces():
        system.removeForce(0)
    groups, out = {}, []

    def add(name, force):
        force.setForceGroup(len(out))
        groups[len(out)] = name
        out.append(force)

    for f in forces:
        if isinstance(f, mm.HarmonicBondForce):
            real, ub = mm.HarmonicBondForce(), mm.HarmonicBondForce()
            for i in range(f.getNumBonds()):
                a, b, r0, k = f.getBondParameters(i)
                (real if frozenset((a, b)) in bonds else ub).addBond(a, b, r0, k)
            add("bond", real)
            if ub.getNumBonds():
                add("angle", ub)
        elif isinstance(f, mm.HarmonicAngleForce):
            add("angle", f)
        elif isinstance(f, mm.PeriodicTorsionForce):
            prop, imp = mm.PeriodicTorsionForce(), mm.PeriodicTorsionForce()
            for i in range(f.getNumTorsions()):
                *t, n, ph, k = f.getTorsionParameters(i)
                (imp if improper(t) else prop).addTorsion(*t, n, ph, k)
            add("dihedral", prop)
            if imp.getNumTorsions():
                add("improper", imp)
        elif isinstance(f, mm.RBTorsionForce):
            add("dihedral", f)
        elif isinstance(f, mm.CustomTorsionForce):
            add("improper", f)
        elif isinstance(f, mm.CMAPTorsionForce):
            add("cmap", f)
        elif isinstance(f, mm.NonbondedForce):
            lj, coul = copy.deepcopy(f), copy.deepcopy(f)
            for i in range(f.getNumParticles()):
                q, s, e = f.getParticleParameters(i)
                lj.setParticleParameters(i, 0.0, s, 0.0 if foyer_geometric else e)
                coul.setParticleParameters(i, q, 1.0, 0.0)
            for i in range(f.getNumExceptions()):
                a, b, qq, s, e = f.getExceptionParameters(i)
                if foyer_geometric and e._value != 0.0:
                    # foyer: the 1-4 sigma mixes geometrically too.
                    _, si, _ = f.getParticleParameters(a)
                    _, sj, _ = f.getParticleParameters(b)
                    s = math.sqrt(si._value * sj._value)
                lj.setExceptionParameters(i, a, b, 0.0, s, e)
                coul.setExceptionParameters(i, a, b, qq, 1.0, 0.0)
            add("vdw", lj)
            add("coul", coul)
            if foyer_geometric:
                geo = mm.CustomNonbondedForce(
                    "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
                    " sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)"
                )
                geo.addPerParticleParameter("sigma")
                geo.addPerParticleParameter("epsilon")
                for i in range(f.getNumParticles()):
                    _, s, e = f.getParticleParameters(i)
                    geo.addParticle([s, e])
                for i in range(f.getNumExceptions()):
                    a, b, *_ = f.getExceptionParameters(i)
                    geo.addExclusion(a, b)
                geo.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
                add("vdw", geo)
        elif isinstance(f, (mm.CustomNonbondedForce, mm.CustomBondForce)):
            add("vdw", f)
        elif isinstance(f, mm.CMMotionRemover):
            continue
        else:
            raise SystemExit(f"unexpected force {type(f).__name__}")
    for f in out:
        system.addForce(f)
    return groups


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] != "thermo":
        raise SystemExit("usage: engine_check_tables.py thermo LOG")
    print(" ".join(f"{k}={v!r}" for k, v in lammps_thermo(argv[1]).items()))


if __name__ == "__main__":
    main(sys.argv[1:])
