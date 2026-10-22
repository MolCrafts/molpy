#!/usr/bin/env bash
# molrs's AMBER prmtop readers against sander and LAMMPS, term by term
# (crates/molrs/src/io/amber/prmtop_check.rs).
#
#   scripts/prmtop_check.sh            # LAMMPS on the committed fixtures
#   scripts/prmtop_check.sh --rebuild  # also rebuild the fixtures + sander
#
# --rebuild needs AmberTools (tleap, antechamber, parmchk2, ParmEd, pysander:
# `module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2
# AmberTools/26.1`); it rewrites the fixtures under
# crates/molrs/src/io/amber/testdata/prmtop and prints sander's terms,
# the `SANDER_*` numbers the test pins. Then the test runs with
# MOLRS_PRMTOP_LAMMPS_DIR set, so molrs prints its own terms and writes each
# case's LAMMPS data file and include; `lmp` (or $LMP, built with MOLECULE
# and the CMAP fix) runs `run 0` on them and the script prints LAMMPS's
# terms, the `LAMMPS_*` numbers the test pins. Run it where cargo may build
# (a compute node).
#
# LAMMPS splits what `thermo` lumps by running each case again with one
# thing changed: `special_bonds 0 0 0` (the non-1-4 pairs alone; the 1-4
# terms are the difference), Urey-Bradley K_ub = 0 (angle without UB), and
# for a chamber file epsilon14/sigma14 in the regular slots with
# `special_bonds 0 0 1` (LAMMPS prices epsilon14 only through `dihedral
# charmm`; the 1-4 pairs at their 1-4 parameters are this run less the
# same with `special_bonds 0 0 0`). LAMMPS has no per-pair 1-4 weight: the non-uniform `glycam`
# case gets its 1-4 pairs unweighted per molecule (`compute pe/atom pair`
# summed per molecule, intermolecular pairs cancelling in the difference)
# and the weight of each molecule's torsions applied here. LAMMPS's Coulomb
# constant is its own `qqr2e`; every Coulomb term is rescaled to the force
# field's (AMBER 332.0522173, CHARMM 332.0716), which is exact: the energy
# is linear in it.
#
# $PYTHON (default python3) needs molrs installed: it reads LAMMPS's log
# with molrs.io.read_lammps_log (scripts/engine_check_tables.py) and takes
# `qqr2e` from molrs.core.constants.COULOMB_REAL.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/without_slurm_step.sh
PYTHON=${PYTHON:-python3}
fixtures=crates/molrs/src/io/amber/testdata/prmtop
dir=${PRMTOP_CHECK_DIR:-$(mktemp -d)}
[ -n "${PRMTOP_CHECK_DIR:-}" ] || trap 'rm -rf "$dir"' EXIT

if [ "${1:-}" = "--rebuild" ]; then
    python3 scripts/prmtop_fixtures.py "$dir/fixtures" | grep '^sander\|^gaff2_multi'
    cp "$dir"/fixtures/*.parm7 "$dir"/fixtures/*.rst7 "$fixtures"/
    # As committed: no trailing blanks (the hygiene hook's rule; a prmtop
    # reader trims every line).
    sed -i 's/[ \t]*$//' "$fixtures"/*.parm7 "$fixtures"/*.rst7
fi

MOLRS_PRMTOP_LAMMPS_DIR="$dir" cargo mrs-test -- \
    io::amber::prmtop_check --nocapture >"$dir/molrs.txt"
grep '^molrs' "$dir/molrs.txt"

# run <case dir> <name> <extra commands after the include> [thermo extras]
run() {
    local case=$1 name=$2 extra=$3
    local pre="" post="" cmap=""
    if [ -f "$case/charmm.cmap" ]; then
        pre=$(grep -E '^fix ' "$case/system.ff")
        post=$(grep -E '^fix_modify ' "$case/system.ff")
        cmap="fix cmap crossterm CMAP"
    fi
    grep -vE '^(units|fix |fix_modify )' "$case/system.ff" >"$case/body.ff"
    cat >"$case/in.$name" <<IN
units           real
atom_style      full
boundary        f f f
$pre
read_data       data.lmp $cmap
$post
include         body.ff
$extra
group           m1 molecule 1
group           m2 molecule 2
compute         pa all pe/atom pair
compute         s1 m1 reduce sum c_pa
compute         s2 m2 reduce sum c_pa
thermo_style    custom step pe ebond eangle edihed eimp evdwl ecoul c_s1 c_s2 ${cmap:+f_cmap}
thermo_modify   format float %.17g
run             0
IN
    (cd "$case" && without_slurm_step "${LMP:-lmp}" -in "in.$name" -log "log.$name" \
        -screen none </dev/null) ||
        { echo "lmp failed on $case/in.$name" >&2; exit 1; }
    "$PYTHON" scripts/engine_check_tables.py thermo "$case/log.$name"
}

for cdir in "$dir"/*/; do
    case=${cdir%/}
    [ -f "$case/system.ff" ] || continue
    name=$(basename "$case")
    {
        echo "case $name"
        echo "coulomb $(cat "$case/coulomb.txt")"
        [ -f "$case/weights.txt" ] && cat "$case/weights.txt"
        echo -n "A " && run "$case" A ""
        echo -n "B " && run "$case" B "special_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0"
        echo -n "Bq " && run "$case" Bq "special_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0
set group all charge 0.0"
        echo -n "Fq " && run "$case" Fq "special_bonds lj 0.0 0.0 1.0 coul 0.0 0.0 1.0
set group all charge 0.0"
        echo -n "F " && run "$case" F "special_bonds lj 0.0 0.0 1.0 coul 0.0 0.0 1.0"
        if grep -q '^angle_style charmm' "$case/system.ff"; then
            awk '$1 == "angle_coeff" { $5 = 0 } { print }' "$case/system.ff" >"$case/noub.ff"
            mv "$case/noub.ff" "$case/system.ff"
            echo -n "U " && run "$case" U ""
        fi
        if grep -q '^pair_style lj/charmm' "$case/system.ff"; then
            awk '$1 == "pair_coeff" && NF == 7 { $4 = $6; $5 = $7 } { print }' \
                "$case/system.ff" >"$case/e14.ff"
            mv "$case/e14.ff" "$case/system.ff"
            echo -n "E " && run "$case" E "special_bonds lj 0.0 0.0 1.0 coul 0.0 0.0 1.0"
            echo -n "E0 " && run "$case" E0 "special_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0"
        fi
    } >"$case/runs.txt"
    "$PYTHON" scripts/prmtop_lammps_terms.py "$case/runs.txt"
done
