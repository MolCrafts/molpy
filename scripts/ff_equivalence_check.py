#!/usr/bin/env python3
"""The engines' side of molrs's cross-engine equivalence check
(crates/molrs/src/ff/equivalence_check.rs; driven by scripts/ff_equivalence_check.sh).

The Rust test writes, per source, the files each engine reads and a
``system.json`` (atoms, bonds, the IR's impropers, the configurations in Å
and the force probes); LAMMPS and GROMACS are run by the shell script. This
script

- ``sander DIR``: prices each prmtop source at its configurations with
  pysander (AmberTools; ``igb = 0``, ``ntb = 0``, ``cut = 999``) and writes
  ``DIR/sander.tsv``;
- ``collect DIR [--pin FILE]``: prices the OpenMM inputs (the molrs-written
  XML with residue templates built here, and an OpenMM-native source's own
  XML) on the Reference platform, ``NoCutoff``, one force group per term;
  reads LAMMPS's logs and force dumps and GROMACS's .edr and .trr; merges
  sander's; prints every engine beside molrs (``DIR/molrs.tsv``) and writes
  the table (``DIR/engines.tsv``, or FILE) the Rust test pins.

Every energy is kcal/mol, every force kcal/(mol·Å). Each row is
``source config engine term value``; the terms are bond, angle (with
Urey-Bradley), dihedral, improper, cmap, vdw, coul (each with its 1-4
pairs), total, and the force fingerprint fdotv = ΣF·v, fnorm2 = Σ|F|².
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from engine_check_tables import (
    lammps_dump_forces,
    lammps_thermo,
    openmm_residue_topology,
    read_energy_tsv,
    split_system,
    unit_factor,
)

TERMS = ["bond", "angle", "dihedral", "improper", "cmap", "vdw", "coul", "total"]
REPO = Path(__file__).resolve().parents[1]


def sources(d: Path):
    for sub in sorted(d.iterdir()):
        if (sub / "system.json").exists():
            yield sub, json.loads((sub / "system.json").read_text(encoding="utf-8"))


def fingerprint(f, v):
    return sum(a * b for a, b in zip(f, v)), sum(a * a for a in f)


def rows_of(name, k, engine, terms, forces, probe):
    out = [(name, k, engine, t, terms.get(t, 0.0)) for t in TERMS]
    dot, norm2 = fingerprint(forces, probe)
    out += [(name, k, engine, "fdotv", dot), (name, k, engine, "fnorm2", norm2)]
    return out


def write_tsv(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# source\tconfig\tengine\tterm\tvalue (kcal/mol; fdotv, fnorm2: kcal/(mol·Å), "
            "(kcal/(mol·Å))²) — scripts/ff_equivalence_check.sh --pin\n"
        )
        f.writelines(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{float(r[4])!r}\n" for r in rows)


# ── sander ────────────────────────────────────────────────────────────────


def run_sander(d: Path):
    import numpy as np
    import sander

    rows = []
    for sub, sysj in sources(d):
        if sysj["native"] != "sander":
            continue
        parm = str(REPO / sysj["native_file"])
        inp = sander.gas_input(0)
        inp.cut = 999.0
        with sander.setup(parm, np.array(sysj["configs"][0]), None, inp):
            for k, x in enumerate(sysj["configs"]):
                sander.set_positions(np.array(x))
                e, f = sander.energy_forces()
                terms = {
                    "bond": e.bond,
                    "angle": e.angle + e.angle_ub,
                    "dihedral": e.dihedral,
                    "improper": e.imp,
                    "cmap": e.cmap,
                    "vdw": e.vdw + e.vdw_14,
                    "coul": e.elec + e.elec_14,
                    "total": e.tot,
                }
                rows += rows_of(
                    sysj["source"], k, "native", terms, list(f), sysj["probes"][k]
                )
    write_tsv(d / "sander.tsv", rows)


# ── OpenMM ────────────────────────────────────────────────────────────────


def openmm_written(sub: Path, sysj):
    """The molrs-written XML with one residue template per molecule (atoms
    named A<i>, charges per atom), the topology it matches and the topology's
    atom order. No foyer patch: molrs's XML carries its geometric mixing as
    the ``<Script>`` ``createSystem`` runs."""
    path = sub / "openmm" / "ff+residues.xml"
    top, order, _ = openmm_residue_topology(
        sub / "openmm" / "ff.xml", path, sysj["types"], sysj["bonds"], sysj["charges"]
    )
    return path, top, order, False


def openmm_native(sysj):
    """An OpenMM-native source's own XML, the topology of its one residue
    template and its atom order (the source's)."""
    from openmm import app

    path = REPO / sysj["native_file"]
    root = ET.parse(path).getroot()
    element = {t.get("name"): t.get("element") for t in root.iter("Type")}
    tmpl = root.find("Residues").find("Residue")
    top = app.Topology()
    res = top.addResidue(tmpl.get("name"), top.addChain())
    made = [
        top.addAtom(a.get("name"), app.Element.getBySymbol(element[a.get("type")]), res)
        for a in tmpl.findall("Atom")
    ]
    for i, j in sysj["bonds"]:
        top.addBond(made[i], made[j])
    return path, top, list(range(len(made))), root.get("combining_rule") == "geometric"


def openmm_price(path, top, order, foyer, sysj):
    """OpenMM's per-term energies and forces at each configuration; the
    topology's atom ``t`` is the source's atom ``order[t]``."""
    import openmm as mm
    import openmm.unit as u
    from openmm import app

    ff = app.ForceField(str(path))
    system = ff.createSystem(
        top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    groups = split_system(system, sysj["bonds"], foyer, impropers=sysj["impropers"])
    ctx = mm.Context(
        system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName("Reference")
    )
    angstrom_per_nm = unit_factor("nm", "angstrom")
    kj_per_kcal = unit_factor("kcal", "kJ")
    out = []
    for x in sysj["configs"]:
        ctx.setPositions(
            [
                mm.Vec3(x[3 * a], x[3 * a + 1], x[3 * a + 2]) / angstrom_per_nm
                for a in order
            ]
        )
        terms = {}
        for g, name in groups.items():
            e = ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()
            terms[name] = (
                terms.get(name, 0.0)
                + e.value_in_unit(u.kilojoule_per_mole) / kj_per_kcal
            )
        state = ctx.getState(getEnergy=True, getForces=True)
        terms["total"] = (
            state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / kj_per_kcal
        )
        f = state.getForces(asNumpy=False).value_in_unit(
            u.kilojoule_per_mole / u.angstrom
        )
        forces = [0.0] * len(x)
        for t, a in enumerate(order):
            forces[3 * a : 3 * a + 3] = [
                c / kj_per_kcal for c in (f[t].x, f[t].y, f[t].z)
            ]
        out.append((terms, forces))
    return out


# ── LAMMPS ────────────────────────────────────────────────────────────────


def lammps(sub: Path, k: int):
    v = lammps_thermo(sub / "lammps" / f"log.{k}")
    terms = {
        "bond": v["E_bond"],
        "angle": v["E_angle"],
        "dihedral": v["E_dihed"],
        "improper": v["E_impro"],
        "cmap": v.get("f_cmap", 0.0),
        "vdw": v["E_vdwl"],
        "coul": v["E_coul"],
        "total": v["PotEng"],
    }
    forces = [
        c for f in lammps_dump_forces(sub / "lammps" / f"forces.{k}.dump") for c in f
    ]
    return terms, forces


# ── GROMACS ───────────────────────────────────────────────────────────────

GMX_TERMS = {
    "bond": ["Bond"],
    "angle": ["Angle", "U-B"],
    "dihedral": ["Proper Dih.", "Ryckaert-Bell.", "Fourier Dih."],
    "improper": ["Improper Dih.", "Per. Imp. Dih."],
    "cmap": ["CMAP Dih."],
    "vdw": ["LJ-14", "LJ (SR)"],
    "coul": ["Coulomb-14", "Coulomb (SR)"],
    "total": ["Potential"],
}


def trr_forces(path: Path):
    """The forces of a GROMACS .trr's first frame, flattened ``fx fy fz`` per
    atom, in kcal/(mol·Å). molrs's TRR reader gives kJ/(mol·Å)."""
    import molrs

    atoms = molrs.io.read_trr_trajectory(path).read_frame(0)["atoms"]
    kj_per_kcal = unit_factor("kcal", "kJ")
    return [
        float(c) / kj_per_kcal
        for fx, fy, fz in zip(atoms["fx"], atoms["fy"], atoms["fz"])
        for c in (fx, fy, fz)
    ]


def gromacs(run: Path):
    import pyedr

    edr = pyedr.edr_to_dict(str(run / "sp.edr"))
    kj_per_kcal = unit_factor("kcal", "kJ")
    terms = {
        t: sum(float(edr[n][0]) for n in names if n in edr) / kj_per_kcal
        for t, names in GMX_TERMS.items()
    }
    return terms, trr_forces(run / "sp.trr")


# ── collect ───────────────────────────────────────────────────────────────


def collect(d: Path, pin: Path | None):
    rows = []
    for sub, sysj in sources(d):
        name, probes = sysj["source"], sysj["probes"]
        for k in range(len(sysj["configs"])):
            rows += rows_of(name, k, "lammps", *lammps(sub, k), probes[k])
            rows += rows_of(
                name, k, "gromacs", *gromacs(sub / "gromacs" / f"run_{k}"), probes[k]
            )
            if sysj["native"] == "gromacs":
                rows += rows_of(
                    name,
                    k,
                    "native",
                    *gromacs(sub / "gromacs" / f"native_{k}"),
                    probes[k],
                )
        for k, (terms, forces) in enumerate(
            openmm_price(*openmm_written(sub, sysj), sysj)
        ):
            rows += rows_of(name, k, "openmm", terms, forces, probes[k])
        if sysj["native"] == "openmm":
            for k, (terms, forces) in enumerate(
                openmm_price(*openmm_native(sysj), sysj)
            ):
                rows += rows_of(name, k, "native", terms, forces, probes[k])
    if (d / "sander.tsv").exists():
        rows += [
            (s, k, e, t, v)
            for (s, k, e, t), v in read_energy_tsv(d / "sander.tsv").items()
        ]
    rows.sort(
        key=lambda r: (
            r[0],
            r[1],
            ["native", "lammps", "openmm", "gromacs"].index(r[2]),
            (TERMS + ["fdotv", "fnorm2"]).index(r[3]),
        )
    )
    write_tsv(pin or d / "engines.tsv", rows)

    molrs = read_energy_tsv(d / "molrs.tsv")
    worst = {}
    print(
        f"{'source':9s} {'k':>1s} {'engine':8s} {'term':9s} {'engine value':>24s} {'molrs':>24s} {'rel':>8s}"
    )
    for s, k, e, t, v in rows:
        m = molrs[(s, k, e, t)]
        if t == "fdotv":
            scale = math.sqrt(molrs[(s, k, e, "fnorm2")]) * math.sqrt(
                sum(
                    p * p
                    for p in json.loads((d / s / "system.json").read_text(encoding="utf-8"))["probes"][
                        k
                    ]
                )
            )
        else:
            scale = max(abs(m), 1.0)
        rel = abs(v - m) / scale
        key = (s, e, t)
        worst[key] = max(worst.get(key, 0.0), rel)
        print(f"{s:9s} {k:1d} {e:8s} {t:9s} {v:24.16g} {m:24.16g} {rel:8.1e}")
    print(
        "\nworst error over the configurations, per source × engine × term (relative to the"
        " term, or to 1 kcal/mol below it; fdotv to |F||v|):"
    )
    names = sorted({s for s, _, _ in worst})
    for s in names:
        for e in ["native", "lammps", "openmm", "gromacs"]:
            cells = " ".join(
                f"{t}={worst[(s, e, t)]:.0e}"
                for t in TERMS + ["fdotv", "fnorm2"]
                if (s, e, t) in worst
            )
            print(f"  {s:9s} {e:8s} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["sander", "collect"])
    ap.add_argument("dir", type=Path)
    ap.add_argument(
        "--pin",
        type=Path,
        help="write the engines' table here (the Rust test's fixture)",
    )
    args = ap.parse_args()
    if args.what == "sander":
        run_sander(args.dir)
    else:
        collect(args.dir, args.pin)


if __name__ == "__main__":
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    sys.exit(main())
