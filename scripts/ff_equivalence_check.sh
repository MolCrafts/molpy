#!/usr/bin/env bash
# The force-field IR across engines (crates/molrs/src/ff/equivalence_check.rs pins
# what this prints): five sources — AMBER ff14SB, GAFF2 and CHARMM36 (chamber)
# prmtops, CHARMM36 OpenMM XML, OPLS-AA GROMACS .top — read into the IR,
# written by molrs to a LAMMPS data file + include, an OpenMM XML and a
# GROMACS topology, and priced by each engine and by the source's own at three
# configurations each, term by term, with forces.
#
#   scripts/ff_equivalence_check.sh          # run, print engines beside molrs
#   scripts/ff_equivalence_check.sh --pin    # also rewrite the pinned table
#                                            # (crates/molrs/src/ff/testdata/equivalence/engines.tsv)
#
# Run it where cargo may build (a compute node). Engines (override by env):
#   LMP           LAMMPS with MOLECULE, EXTRA-MOLECULE and the CMAP fix (lmp)
#   GMX           GROMACS in double precision (gmx_d; `module load
#                 GROMACS/2025.3-gcc-2025b-eb`)
#   PYTHON        python with openmm 8.x, pyedr and molrs (python3), e.g. a
#                 venv made by `uv venv -p 3.12 omm && uv pip install -p
#                 omm/bin/python openmm==8.6.1 pyedr numpy` plus molrs
#                 (`maturin develop` of molrs-python)
#   AMBER_ENV     a script that puts AmberTools' python (pysander) on PATH, e.g.
#                 one running `module load buildtool-easybuild/5.2.1-hpca3ef7d197
#                 GCC/14.3.0 MPICH/4.3.2 AmberTools/26.1`; sander runs in a
#                 subshell that sources it
# GROMACS: plain cut-off at 4.5 nm (no shift, no reaction field) in a 10 nm
# box, past every pair of each molecule and short of every image; LAMMPS and
# OpenMM: no cutoff; sander: cut = 999.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/without_slurm_step.sh
LMP=${LMP:-lmp}
GMX=${GMX:-gmx_d}
PYTHON=${PYTHON:-python3}
pin=
[[ ${1:-} == --pin ]] && pin=crates/molrs/src/ff/testdata/equivalence/engines.tsv
dir=${FF_EQUIV_DIR:-$(mktemp -d)}
[[ -n ${FF_EQUIV_DIR:-} || -n ${KEEP:-} ]] && echo "work directory: $dir" || trap 'rm -rf "$dir"' EXIT
mkdir -p "$dir"

MOLRS_FF_EQUIV_DIR="$dir" cargo mrs-test -- \
    ff::equivalence_check::write_engine_inputs --exact >/dev/null

cat >"$dir/sp.mdp" <<'MDP'
integrator              = md
nsteps                  = 0
cutoff-scheme           = Verlet
pbc                     = xyz
nstlist                 = 1
verlet-buffer-tolerance = -1
rlist                   = 4.5
coulombtype             = Cut-off
coulomb-modifier        = None
rcoulomb                = 4.5
vdwtype                 = Cut-off
vdw-modifier            = None
rvdw                    = 4.5
DispCorr                = no
epsilon-r               = 1
nstcalcenergy           = 1
nstenergy               = 1
nstfout                 = 1
constraints             = none
MDP

for src in "$dir"/*/; do
    src=${src%/}
    [[ -f $src/system.json ]] || continue
    native=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["native"])' "$src/system.json")
    native_file=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["native_file"])' "$src/system.json")
    for k in 0 1 2; do
        # LAMMPS, `run 0` on the data file and include molrs wrote.
        read_data="read_data       data_$k.lmp"
        cmap=""
        if [[ -f $src/lammps/system.cmap ]]; then
            read_data="$read_data fix cmap crossterm CMAP"
            cmap=" f_cmap"
        fi
        cat >"$src/lammps/in.$k" <<IN
include         pre.lmp
atom_style      full
boundary        f f f
$read_data
include         system.ff
neighbor        2.0 nsq
thermo_style    custom step evdwl ecoul ebond eangle edihed eimp$cmap pe
thermo_modify   format float %.17g
dump            d all custom 1 forces.$k.dump id fx fy fz
dump_modify     d format float %.17g sort id
run             0
IN
        (cd "$src/lammps" && without_slurm_step "$LMP" -in "in.$k" -log "log.$k" -screen none </dev/null) ||
            { echo "lmp failed on $src/lammps/in.$k" >&2; exit 1; }
        # GROMACS, single point on the topology molrs wrote (and, for a
        # GROMACS source, on the source's own).
        runs=("run_$k:$src/gromacs/topol.top")
        [[ $native == gromacs ]] && runs+=("native_$k:$PWD/$native_file")
        for spec in "${runs[@]}"; do
            run=$src/gromacs/${spec%%:*}
            mkdir -p "$run"
            (cd "$run" &&
                "$GMX" grompp -f "$dir/sp.mdp" -c "../conf_$k.gro" -p "${spec#*:}" \
                    -o sp.tpr -maxwarn 10 >grompp.log 2>&1 &&
                "$GMX" mdrun -s sp.tpr -rerun "../conf_$k.gro" -deffnm sp -nt 1 \
                    >mdrun.log 2>&1) ||
                { echo "GROMACS failed in $run" >&2; tail -20 "$run"/grompp.log "$run"/mdrun.log >&2; exit 1; }
        done
    done
done

# sander, in AmberTools' own python.
if [[ -n ${AMBER_ENV:-} ]]; then
    (set +u; source "$AMBER_ENV"; python3 scripts/ff_equivalence_check.py sander "$dir")
else
    echo "AMBER_ENV unset: sander not run (the prmtop sources' native column stays empty)" >&2
fi

"$PYTHON" scripts/ff_equivalence_check.py collect "$dir" ${pin:+--pin "$pin"}
