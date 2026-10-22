#!/usr/bin/env bash
# GROMACS-read force fields against GROMACS and LAMMPS, term by term
# (crates/molrs/src/io/gromacs/top_reader/engine_check.rs pins what this
# prints).
#
# The fixtures are an ACE-ALA-ALA-NME dipeptide built by `gmx pdb2gmx` under
# three ports — charmm27 (Urey-Bradley, CMAP, [ pairtypes ], a [ nonbond_params ]
# row added), amber99sb-ildn (funct 9, funct 4, gen-pairs 0.5 / 0.8333) and
# oplsaa (funct 3, funct 1 impropers from #define macros) — preprocessed by
# `grompp -pp`, trimmed to the types the molecule uses, and perturbed off its
# minimum; `amber_pairs` is the AMBER one with [ pairs ] rows carrying their own
# parameters (funct 1 and 2).
#
#   scripts/gromacs_engine_check.sh            # check the committed fixtures
#   scripts/gromacs_engine_check.sh --regen    # rebuild them first (pdb2gmx …)
#
# For each fixture it runs GROMACS (single point: `grompp`, `mdrun -rerun`,
# energies read at full precision from the .edr), molrs (the test, with
# MOLRS_GMX_CHECK_DIR set, prints molrs's terms and writes the LAMMPS inputs
# from molrs's LAMMPS writer) and LAMMPS (`run 0`), and prints the three side
# by side in kcal/mol. Nonbonded: plain cut-off at 2.5 nm (no shift, no
# reaction field), a 6 nm periodic box, so every intramolecular pair is inside
# the cutoff and no image is.
#
# Needs a compute node (cargo builds), $GMX (default gmx_d: GROMACS in double
# precision, e.g. `module load GROMACS/2025.3-gcc-2025b-eb`), $LMP (default lmp,
# with MOLECULE and EXTRA-MOLECULE) and $PYTHON (default python3) with `pyedr` and molrs (LAMMPS's log is read
# by molrs.io.read_lammps_log, kJ→kcal is a factor of molrs's unit registry).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/without_slurm_step.sh
GMX=${GMX:-gmx_d}
LMP=${LMP:-lmp}
PYTHON=${PYTHON:-python3}
data=crates/molrs/src/io/gromacs/testdata
work=$(mktemp -d)
# KEEP=1 leaves the inputs and logs behind for a look.
[[ -n ${KEEP:-} ]] && echo "work directory: $work" || trap 'rm -rf "$work"' EXIT

if [[ ${1:-} == --regen ]]; then
    "$PYTHON" - "$work" <<'PY'
import sys
from pathlib import Path

import molrs

DEG = molrs.core.UnitRegistry().factor("deg", "rad")

def place(a, b, c, bond, ang, tor):
    """The atom bonded to c at `bond` Å, angle b-c-x `ang`°, dihedral a-b-c-x `tor`° (NeRF)."""
    return list(molrs.op.place_from_internal_coords(a, b, c, bond, ang * DEG, tor * DEG))

# ACE-ALA-ALA-NME heavy atoms, phi/psi off the CMAP grid.
X = {}
X["ACE", "CH3"] = [0.0, 0.0, 0.0]
X["ACE", "C"] = [1.52, 0.0, 0.0]
X["A1", "N"] = place([0, 1, 0], X["ACE", "CH3"], X["ACE", "C"], 1.33, 116.0, 180.0)
X["A1", "CA"] = place(X["ACE", "CH3"], X["ACE", "C"], X["A1", "N"], 1.46, 122.0, 180.0)
X["A1", "C"] = place(X["ACE", "C"], X["A1", "N"], X["A1", "CA"], 1.52, 111.0, -75.0)
X["A2", "N"] = place(X["A1", "N"], X["A1", "CA"], X["A1", "C"], 1.33, 116.0, 145.0)
X["A2", "CA"] = place(X["A1", "CA"], X["A1", "C"], X["A2", "N"], 1.46, 122.0, 180.0)
X["A2", "C"] = place(X["A1", "C"], X["A2", "N"], X["A2", "CA"], 1.52, 111.0, -150.0)
X["NME", "N"] = place(X["A2", "N"], X["A2", "CA"], X["A2", "C"], 1.33, 116.0, 160.0)
X["NME", "CH3"] = place(X["A2", "CA"], X["A2", "C"], X["NME", "N"], 1.46, 122.0, 180.0)
X["ACE", "O"] = place(X["A1", "N"], X["ACE", "CH3"], X["ACE", "C"], 1.23, 121.0, 180.0)
for r, nxt in (("A1", ("A2", "N")), ("A2", ("NME", "N"))):
    X[r, "O"] = place(X[nxt], X[r, "CA"], X[r, "C"], 1.23, 121.0, 180.0)
    X[r, "CB"] = place(X[r, "C"], X[r, "N"], X[r, "CA"], 1.53, 110.5, -122.5)
order = [("ACE", ["CH3", "C", "O"]), ("A1", ["N", "CA", "CB", "C", "O"]),
         ("A2", ["N", "CA", "CB", "C", "O"]), ("NME", ["N", "CH3"])]
work = Path(sys.argv[1])
for ff, cap in (("charmm27", "CT3"), ("amber99sb-ildn", "NME"), ("oplsaa", "NAC")):
    names = {"ACE": "ACE", "A1": "ALA", "A2": "ALA", "NME": cap}
    lines, k = [], 0
    for ri, (r, atoms) in enumerate(order):
        for a in atoms:
            k += 1
            x = X[r, a]
            lines.append("ATOM  %5d  %-3s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                         % (k, a, names[r], ri + 1, x[0], x[1], x[2], a[0]))
    (work / f"{ff}.pdb").write_text("\n".join(lines + ["TER", "END"]) + "\n")
PY
    for ff in charmm27 amber99sb-ildn oplsaa; do
        mkdir -p "$work/$ff"
        # Termini "None": the caps are residues of their own.
        case $ff in
            charmm27) ter=(-ter); choice=$'3\n5\n' ;;
            oplsaa) ter=(-ter); choice=$'3\n3\n' ;;
            *) ter=(); choice= ;;
        esac
        (cd "$work/$ff" && printf '%s' "$choice" |
            "$GMX" pdb2gmx -f "../$ff.pdb" -o conf.gro -p topol.top -ff "$ff" \
                -water none -ignh "${ter[@]}" >pdb2gmx.log 2>&1)
        cat >"$work/$ff/pp.mdp" <<'MDP'
cutoff-scheme = Verlet
MDP
        (cd "$work/$ff" &&
            "$GMX" editconf -f conf.gro -o box.gro -box 6 >editconf.log 2>&1 &&
            "$GMX" grompp -f pp.mdp -c box.gro -p topol.top -pp processed.top \
                -o pp.tpr -maxwarn 5 >grompp-pp.log 2>&1)
    done
    "$PYTHON" - "$work" "$data" <<'PY'
import math, sys
from pathlib import Path

work, data = Path(sys.argv[1]), Path(sys.argv[2])
DIRECTIVES = ["defaults", "atomtypes", "nonbond_params", "pairtypes", "bondtypes",
              "constrainttypes", "angletypes", "dihedraltypes", "cmaptypes"]

def logical_lines(text):
    """Lines joined at a trailing backslash, comments and blanks dropped."""
    out, pending = [], ""
    for raw in text.splitlines():
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        line = (pending + raw).split(";")[0].strip()
        pending = ""
        if line:
            out.append(line)
    return out

def trim(top, extra_nonbond=""):
    lines = logical_lines(top)
    # Used atom types and their bond types (classes).
    sec, used = None, set()
    for l in lines:
        if l.startswith("["):
            sec = l.strip("[] ").lower()
        elif sec == "atoms":
            used.add(l.split()[1])
    classes = {}
    sec = None
    for l in lines:
        if l.startswith("["):
            sec = l.strip("[] ").lower()
        elif sec == "atomtypes":
            c = l.split()
            lead = c[:-5]
            cls = lead[1] if len(lead) >= 2 and not lead[1].lstrip("-").isdigit() else lead[0]
            classes[lead[0]] = cls
    labels = {classes[t] for t in used} | {"X"}
    sec, out = None, []
    for l in lines:
        if l.startswith("#") or (sec is None and not l.startswith("[")):
            continue
        if l.startswith("["):
            sec = l.strip("[] ").lower()
            out.append(f"\n[ {sec} ]")
            continue
        c = l.split()
        if sec == "atomtypes":
            ok = c[0] in used
        elif sec in ("nonbond_params", "pairtypes"):
            ok = c[0] in used and c[1] in used
        elif sec in ("bondtypes", "constrainttypes"):
            ok = all(x in labels for x in c[:2])
        elif sec == "angletypes":
            ok = all(x in labels for x in c[:3])
        elif sec == "dihedraltypes":
            n = 2 if c[2].isdigit() and not c[4].isdigit() else 4
            ok = all(x in labels for x in c[:n])
        elif sec == "cmaptypes":
            ok = all(x in labels for x in c[:5])
            if ok:
                head, vals = c[:8], c[8:]
                rows = [" ".join(vals[i:i + 10]) for i in range(0, len(vals), 10)]
                l = " ".join(head) + "\\\n" + "\\\n".join(rows)
        else:
            ok = True
        if ok:
            out.append(l)
    text = "\n".join(out).strip() + "\n"
    # Drop sections left empty.
    blocks = text.split("\n\n")
    blocks = [b.strip("\n") for b in blocks if len(b.strip().splitlines()) > 1]
    text = "\n\n".join(blocks) + "\n"
    if extra_nonbond:
        text = text.replace("\n\n[ pairtypes ]", f"\n\n[ nonbond_params ]\n{extra_nonbond}\n\n[ pairtypes ]", 1)
    return text

def perturb(gro):
    """Every coordinate moved by up to 0.012 nm, into a 6 nm box."""
    lines = gro.splitlines()
    n = int(lines[1])
    out = ["ACE-ALA-ALA-NME off its minimum", lines[1]]
    for k in range(n):
        l = lines[2 + k]
        xyz = [float(l[20 + 8 * d:28 + 8 * d]) for d in range(3)]
        xyz = [x + 0.012 * math.sin(1.7 * k + 2.3 * d + 0.4) + 3.0 for d, x in enumerate(xyz)]
        out.append(l[:20] + "".join("%8.3f" % v for v in xyz))
    out.append("   6.00000   6.00000   6.00000")
    return "\n".join(out) + "\n"

head = "; {what}: ACE-ALA-ALA-NME from `gmx pdb2gmx -ff {ff}` (GROMACS 2025.3),\n" \
       "; `grompp -pp`, trimmed to the types it uses (scripts/gromacs_engine_check.sh).\n"
for ff, name, nbfix in (("charmm27", "charmm", "NH1  O  1  0.29  0.6"),
                        ("amber99sb-ildn", "amber", ""), ("oplsaa", "opls", "")):
    top = trim((work / ff / "processed.top").read_text(), nbfix)
    what = name + (" (plus a [ nonbond_params ] row)" if nbfix else "")
    (data / f"{name}.top").write_text(head.format(what=what, ff=ff) + top)
    (data / f"{name}.gro").write_text(perturb((work / ff / "conf.gro").read_text()))
# AMBER with two [ pairs ] rows that carry their own parameters.
top = (data / "amber.top").read_text()
sec = top.index("[ pairs ]")
rows = top[sec:].splitlines()
rows[1] = rows[1] + "  0.3  0.5"
rows[2] = "  ".join(rows[2].split()[:2] + ["2", "0.8", "0.2", "-0.3", "0.31", "0.4"])
pairs = top[:sec] + "\n".join(rows) + "\n"
(data / "amber_pairs.top").write_text(pairs.replace("; amber:", "; amber_pairs: amber with [ pairs ] rows of their own;", 1))
PY
    cp "$data/amber.gro" "$data/amber_pairs.gro"
fi

cat >"$work/sp.mdp" <<'MDP'
integrator              = md
nsteps                  = 0
cutoff-scheme           = Verlet
pbc                     = xyz
nstlist                 = 1
verlet-buffer-tolerance = -1
rlist                   = 2.5
coulombtype             = Cut-off
coulomb-modifier        = None
rcoulomb                = 2.5
vdwtype                 = Cut-off
vdw-modifier            = None
rvdw                    = 2.5
DispCorr                = no
epsilon-r               = 1
nstcalcenergy           = 1
nstenergy               = 1
constraints             = none
MDP

systems=(charmm amber opls amber_pairs)
for sys in "${systems[@]}"; do
    mkdir -p "$work/gmx-$sys"
    (cd "$work/gmx-$sys" &&
        "$GMX" grompp -f ../sp.mdp -c "$OLDPWD/$data/$sys.gro" -p "$OLDPWD/$data/$sys.top" \
            -o sp.tpr -maxwarn 5 >grompp.log 2>&1 &&
        "$GMX" mdrun -s sp.tpr -rerun "$OLDPWD/$data/$sys.gro" -deffnm sp -nt 1 \
            >mdrun.log 2>&1)
done

MOLRS_GMX_CHECK_DIR="$work" cargo mrs-test -- \
    io::gromacs::top_reader::engine_check --nocapture 2>&1 |
    grep '^molrs ' >"$work/molrs.txt"

for sys in charmm amber opls; do
    dir="$work/lmp-$sys"
    for run in full sr; do
        cat >"$dir/in.$run" <<IN
units           real
atom_style      full
boundary        p p p
$(cat "$dir/pre.lmp")
read_data       data.lmp $(cat "$dir/read_data_extra")
include         system.ff
$( [[ $run == sr ]] && cat "$dir/no14.lmp" )
thermo_style    custom step pe ebond eangle edihed eimp evdwl ecoul $(cat "$dir/thermo_extra")
thermo_modify   format float %.17g
run             0
IN
        (cd "$dir" && without_slurm_step "$LMP" -in "in.$run" -log "log.$run" -screen none </dev/null)
    done
done

PYTHONPATH=scripts "$PYTHON" - "$work" <<'PY'
import sys
from pathlib import Path
import pyedr
from engine_check_tables import lammps_thermo, unit_factor

work = Path(sys.argv[1])
GMX_TERMS = {
    "bond": ["Bond"], "angle": ["Angle", "U-B"], "proper": ["Proper Dih."],
    "rb": ["Ryckaert-Bell."], "improper": ["Improper Dih.", "Per. Imp. Dih."],
    "cmap": ["CMAP Dih."], "lj14": ["LJ-14"], "coul14": ["Coulomb-14"],
    "ljsr": ["LJ (SR)"], "coulsr": ["Coulomb (SR)"], "total": ["Potential"],
}

molrs = {}
for line in (work / "molrs.txt").read_text().splitlines():
    _, sys_, term, value = line.split()
    molrs.setdefault(sys_, {})[term] = float(value)

for sys_ in ("charmm", "amber", "opls", "amber_pairs"):
    edr = pyedr.edr_to_dict(str(work / f"gmx-{sys_}" / "sp.edr"))
    gmx = {}
    for term, names in GMX_TERMS.items():
        vals = [float(edr[n][0]) for n in names if n in edr]
        if vals:
            gmx[term] = sum(vals) / unit_factor("kcal", "kJ")
    lmp = {}
    if sys_ != "amber_pairs":
        d = work / f"lmp-{sys_}"
        full, sr = lammps_thermo(d / "log.full"), lammps_thermo(d / "log.sr")
        lmp = {"bond": full["E_bond"], "angle": full["E_angle"],
               "dihedral": full["E_dihed"], "improper": full["E_impro"],
               "ljsr": sr["E_vdwl"], "coulsr": sr["E_coul"],
               "lj14": full["E_vdwl"] - sr["E_vdwl"], "coul14": full["E_coul"] - sr["E_coul"],
               "total": full["PotEng"]}
        if "f_cmap" in full:
            lmp["cmap"] = full["f_cmap"]
    print(f"== {sys_} (kcal/mol)")
    print(f"{'term':10s} {'GROMACS':>24s} {'LAMMPS':>24s} {'molrs':>24s}  {'molrs/gmx-1':>10s}  {'molrs/lmp-1':>10s}")
    m = molrs[sys_]
    for term in m:
        g = gmx.get(term)
        if term == "dihedral":
            g = gmx.get("proper", 0.0) + gmx.get("rb", 0.0)
        l = lmp.get(term)
        rel = lambda a, b: f"{(a - b) / b:10.1e}" if b not in (None, 0.0) else f"{'—':>10s}"
        fmt = lambda v: f"{v:24.17g}" if v is not None else f"{'—':>24s}"
        print(f"{term:10s} {fmt(g)} {fmt(l)} {fmt(m[term])}  {rel(m[term], g) if g is not None else '':>10s}  {rel(m[term], l) if l is not None else '':>10s}")
PY
