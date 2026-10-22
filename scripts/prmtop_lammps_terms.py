#!/usr/bin/env python3
"""LAMMPS's per-term energies of one prmtop_check case, in sander's terms.

Reads the ``runs.txt`` scripts/prmtop_check.sh writes (one thermo line per
LAMMPS run; see that script for what each run changes) and prints
``lammps <case> <term> <value>`` lines.
"""

import sys
from pathlib import Path

from molpy.core.constants import COULOMB_REAL  # LAMMPS's qqr2e, units real


def main():
    path = Path(sys.argv[1])
    runs, weights = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "case":
            case = words[1]
        elif words[0] == "coulomb":
            coulomb = float(words[1])
        elif words[0] == "weights":
            weights[int(words[1])] = (float(words[2]), float(words[3]))
        else:
            runs[words[0]] = {k: float(v) for k, v in (w.split("=") for w in words[1:])}
    f = coulomb / COULOMB_REAL
    a, b = runs["A"], runs["B"]
    ff = (path.parent / "body.ff").read_text(encoding="utf-8")
    charmm_improper = "improper_style harmonic" in ff
    terms = {"bond": a["E_bond"]}
    if "U" in runs:
        terms["angle"] = runs["U"]["E_angle"]
        terms["angle_ub"] = a["E_angle"] - runs["U"]["E_angle"]
    else:
        terms["angle"], terms["angle_ub"] = a["E_angle"], 0.0
    terms["dihedral"] = a["E_dihed"] + (0.0 if charmm_improper else a["E_impro"])
    terms["imp"] = a["E_impro"] if charmm_improper else 0.0
    terms["cmap"] = a.get("f_cmap", 0.0)
    terms["vdw"] = b["E_vdwl"]
    terms["elec"] = b["E_coul"] * f
    if weights:
        bq, fq, full = runs["Bq"], runs["Fq"], runs["F"]
        vdw14 = elec14 = 0.0
        for mol, (coul_w, lj_w) in weights.items():
            key = f"c_s{mol}"
            v = fq[key] - bq[key]
            c = (full[key] - b[key]) - v
            vdw14 += lj_w * v
            elec14 += coul_w * c * f
        terms["vdw_14"], terms["elec_14"] = vdw14, elec14
    else:
        e, e0 = runs.get("E", a), runs.get("E0", b)
        terms["vdw_14"] = e["E_vdwl"] - e0["E_vdwl"]
        terms["elec_14"] = (a["E_coul"] - b["E_coul"]) * f
    for term, value in terms.items():
        print(f"lammps {case} {term} {value!r}")


if __name__ == "__main__":
    main()
