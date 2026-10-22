#!/usr/bin/env bash
# The force-field IR extension proof against LAMMPS: what a third party adds
# to the IR — a pair style written as a Rust ScalarForm, a new category, a
# style given by its expression alone — written to a LAMMPS deck by molrs's
# own LAMMPS writer (the positional codecs the specs derive) and priced by
# `lmp` `run 0`, beside molrs's numbers. molrs-python/tests/test_ff_ir_extension.py
# pins what this prints.
#
#   scripts/ff_ir_extension_lammps_check.sh          # run, print LAMMPS beside molrs
#   scripts/ff_ir_extension_lammps_check.sh --pin    # also rewrite the pinned tables
#
# Each case directory holds a deck written by a test with MOLRS_FFEXT_DIR set
# (pre.lmp, system.ff, data_<k>.lmp) and molrs's numbers (molrs.tsv); LAMMPS
# prints its energy with `thermo_modify format float %.17g` and its forces in
# a `%.17g` dump. Cases: `python/*` from the Python proof, run by the python
# MOLRS_PYTHON names (pinned in molrs-python/tests/ff_ir_extension_lammps.tsv).
#
# `lmp` needs `bond_style fene`, `pair_style lj/smooth/linear` and
# `angle_style charmm` (MOLECULE, EXTRA-MOLECULE, EXTRA-PAIR). The
# `bond_angle` cases (class2's bond-angle cross term) need CLASS2 and run
# only when MOLRS_LMP_CLASS2 names such a LAMMPS; scripts/build_lammps_class2.sh
# builds one. They are compared, never pinned (the tests hold that term to
# its hand value, central differences and its expression instead).
#
# Run it where cargo may build (a compute node). Engines (override by env):
#   LMP               LAMMPS with MOLECULE (lmp)
#   MOLRS_LMP_CLASS2  LAMMPS with CLASS2 (scripts/build_lammps_class2.sh;
#                     unset: the bond_angle cases are skipped)
#   MOLRS_PYTHON      python with molrs installed (required)
#   PYTHON            python with molrs installed, which reads LAMMPS's logs
#                     and dumps (molrs.io.read_lammps_log,
#                     molrs.io.read_lammps_dump_trajectory; python3)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/without_slurm_step.sh
LMP=${LMP:-lmp}
PYTHON=${PYTHON:-python3}
pin=
[[ ${1:-} == --pin ]] && pin=--pin
dir=${MOLRS_FFEXT_DIR:-$(mktemp -d)}
[[ -n ${MOLRS_FFEXT_DIR:-} || -n ${KEEP:-} ]] && echo "work directory: $dir" || trap 'rm -rf "$dir"' EXIT
mkdir -p "$dir"
export MOLRS_FFEXT_DIR=$dir

: "${MOLRS_PYTHON:?name a python with molrs installed}"
"$MOLRS_PYTHON" -m pytest -q -p no:cacheprovider \
    molrs-python/tests/test_ff_ir_extension.py -k write_lammps_inputs >/dev/null

for case in "$dir"/*/*/; do
    case=${case%/}
    name=${case##*/}
    lmp=$LMP
    if [[ $name == bond_angle ]]; then
        if [[ -z ${MOLRS_LMP_CLASS2:-} ]]; then
            echo "${case#"$dir"/}: skipped (MOLRS_LMP_CLASS2 names no LAMMPS with CLASS2)"
            continue
        fi
        lmp=$MOLRS_LMP_CLASS2
    fi
    for data in "$case"/data_*.lmp; do
        k=${data##*/data_}
        k=${k%.lmp}
        cat >"$case/in.$k" <<IN
include         pre.lmp
atom_style      full
boundary        f f f
read_data       data_$k.lmp
include         system.ff
neighbor        2.0 nsq
thermo_style    custom step pe
thermo_modify   norm no format float %.17g
dump            d all custom 1 forces_$k.dump id fx fy fz
dump_modify     d format float %.17g sort id
run             0
IN
        (cd "$case" && without_slurm_step "$lmp" -in "in.$k" -log "log.$k" -screen none </dev/null) ||
            { echo "lmp failed on $case/in.$k" >&2; tail -20 "$case/log.$k" >&2; exit 1; }
    done
done

PYTHONPATH=scripts "$PYTHON" - "$dir" $pin <<'PY'
import sys
from pathlib import Path

from engine_check_tables import lammps_dump_forces, lammps_thermo

root = Path(sys.argv[1])
pin = "--pin" in sys.argv
PINS = {
    "python": Path("molrs-python/tests/ff_ir_extension_lammps.tsv"),
}
HEADER = "# case\tconfig\tpe | f atom\tLAMMPS's value(s) (scripts/ff_ir_extension_lammps_check.sh --pin)\n"


def lammps(case: Path, k: str) -> tuple[float, list[list[float]]]:
    pe = lammps_thermo(case / f"log.{k}")["PotEng"]
    return pe, lammps_dump_forces(case / f"forces_{k}.dump")


def molrs(case: Path) -> dict[str, tuple[float, list[list[float]]]]:
    out: dict[str, tuple[float, list[list[float]]]] = {}
    for line in (case / "molrs.tsv").read_text().splitlines():
        c = line.split("\t")
        pe, forces = out.get(c[1], (0.0, []))
        if c[2] == "pe":
            pe = float(c[3])
        else:
            forces = forces + [[float(v) for v in c[4:7]]]
        out[c[1]] = (pe, forces)
    return out


bad = []
for side, target in PINS.items():
    rows = []
    for case in sorted((root / side).glob("*/")) if (root / side).is_dir() else []:
        ours = molrs(case)
        for k in sorted(ours):
            if not (case / f"log.{k}").exists():
                continue
            pe, forces = lammps(case, k)
            mpe, mforces = ours[k]
            scale = max((abs(v) for f in forces for v in f), default=0.0)
            rel_e = abs(mpe - pe) / max(abs(pe), sys.float_info.min)
            rel_f = max(
                (abs(a - b) / max(abs(b), scale) for fm, fl in zip(mforces, forces) for a, b in zip(fm, fl)),
                default=0.0,
            )
            print(f"{side}/{case.name} config {k}: LAMMPS pe {pe:.17e}, molrs {mpe:.17e}; "
                  f"rel energy {rel_e:.1e}, worst rel force {rel_f:.1e}")
            if max(rel_e, rel_f) > 1e-10:
                bad.append(f"{side}/{case.name} config {k}")
            if case.name == "bond_angle":
                continue
            rows.append(f"{case.name}\t{k}\tpe\t{pe!r}\n")
            for atom, f in enumerate(forces):
                rows.append(f"{case.name}\t{k}\tf\t{atom}\t{f[0]!r}\t{f[1]!r}\t{f[2]!r}\n")
    if pin and rows and not bad:
        target.write_text(HEADER + "".join(rows))
        print(f"pinned {target}")
if bad:
    sys.exit("beyond relative 1e-10: " + ", ".join(bad))
PY
