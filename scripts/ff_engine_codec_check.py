#!/usr/bin/env python3
"""The engines' side of molrs's engine-codec check
(crates/molrs/src/ff/engine_codec_check.rs; driven by scripts/ff_engine_codec_check.sh).

``collect DIR [--pin FILE]``: reads LAMMPS's logs of the LAMMPS cases (the
shell script ran ``lmp``), prices the OpenMM cases' molrs-written XML on the
Reference platform (``NoCutoff``, one force group per term, residue templates
built here from ``system.json``), prints every engine number beside molrs's
(``DIR/molrs.tsv``) with the relative difference, and writes the table
(``DIR/engines.tsv``, or FILE) the Rust test pins. Each row is ``case config
engine term value``; a LAMMPS case's numbers are in its files' units, an
OpenMM case's in kcal/mol.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from engine_check_tables import (
    lammps_thermo,
    openmm_residue_topology,
    read_energy_tsv,
    unit_factor,
)


def lammps(sub: Path, k: int):
    v = lammps_thermo(sub / f"log.{k}")
    names = {"E_bond": "bond", "E_angle": "angle", "E_vdwl": "vdw", "PotEng": "total"}
    return {t: v[k] for k, t in names.items() if k in v}


def openmm(sub: Path, sysj):
    import openmm as mm
    import openmm.unit as u
    from openmm import app

    path = sub / "ff+residues.xml"
    top, order, _ = openmm_residue_topology(
        sub / "ff.xml", path, sysj["types"], sysj["bonds"]
    )
    system = app.ForceField(str(path)).createSystem(
        top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    term = {
        "CustomBondForce": "bond",
        "CustomAngleForce": "angle",
        "CustomTorsionForce": "improper",
        "CustomCompoundBondForce": "urey_bradley",
        "CustomNonbondedForce": "vdw",
    }
    groups = {}
    for i, f in enumerate(system.getForces()):
        name = type(f).__name__
        if name == "CMMotionRemover":
            continue
        f.setForceGroup(i)
        groups[i] = term[name]
    ctx = mm.Context(
        system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName("Reference")
    )
    angstrom_per_nm, kj_per_kcal = (
        unit_factor("nm", "angstrom"),
        unit_factor("kcal", "kJ"),
    )
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
        e = ctx.getState(getEnergy=True).getPotentialEnergy()
        terms["total"] = e.value_in_unit(u.kilojoule_per_mole) / kj_per_kcal
        out.append(terms)
    return out


def collect(d: Path, pin: Path | None):
    molrs = read_energy_tsv(d / "molrs.tsv")
    rows = []
    for case, k, engine, term in sorted({key for key in molrs}):
        rows.append((case, k, engine, term))
    engines = {}
    for case in sorted({r[0] for r in rows}):
        sub = d / case
        engine = next(r[2] for r in rows if r[0] == case)
        configs = sorted({r[1] for r in rows if r[0] == case})
        if engine == "lammps":
            for k in configs:
                for t, v in lammps(sub, k).items():
                    engines[(case, k, engine, t)] = v
        else:
            sysj = json.loads((sub / "system.json").read_text(encoding="utf-8"))
            for k, terms in enumerate(openmm(sub, sysj)):
                for t, v in terms.items():
                    engines[(case, k, engine, t)] = v
    worst = 0.0
    lines = [
        "# case\tconfig\tengine\tterm\tvalue (scripts/ff_engine_codec_check.sh --pin)"
    ]
    # A term the engine prints and molrs has no style of is zero, or the
    # check fails here.
    extra = {k: v for k, v in engines.items() if k not in molrs and v != 0.0}
    if extra:
        raise SystemExit(f"engine terms molrs has no style for: {extra}")
    for key in sorted(molrs):
        m, e = molrs.get(key), engines.get(key)
        rel = (
            abs(e - m) / max(abs(m), 1.0)
            if m is not None and e is not None
            else float("nan")
        )
        worst = max(worst, rel) if not math.isnan(rel) else worst
        print(
            f"{key[0]:8s} {key[1]} {key[2]:7s} {key[3]:13s} engine {e!r:>24} molrs {m!r:>24} rel {rel:.1e}"
        )
        if e is not None:
            lines.append("\t".join([key[0], str(key[1]), key[2], key[3], repr(e)]))
    print(f"worst relative difference: {worst:.1e}")
    out = pin or d / "engines.tsv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("dir", type=Path)
    c.add_argument("--pin", type=Path)
    a = ap.parse_args()
    collect(a.dir, a.pin)


if __name__ == "__main__":
    main()
