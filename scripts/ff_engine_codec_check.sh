#!/usr/bin/env bash
# Engine codecs against the engines (crates/molrs/src/ff/engine_codec_check.rs pins
# what this prints): styles written through their LAMMPS codecs (a run-time
# `bond fene` and `pair lj/smooth/linear` registered with a positional LAMMPS
# form; built-in `bond morse` and `pair buck` written in `metal`) priced by
# LAMMPS `run 0`, and expression styles written as OpenMM `Custom*Force`s
# (a run-time `bond fene`, built-in `bond morse` and `angle class2`, a
# run-time compound category `urey_bradley`, a `lj/smooth/linear` pair)
# priced by OpenMM's Reference platform — each beside molrs's energy.
#
#   scripts/ff_engine_codec_check.sh          # run, print engines beside molrs
#   scripts/ff_engine_codec_check.sh --pin    # also rewrite the pinned table
#                                             # (crates/molrs/src/ff/testdata/engine_codecs/engines.tsv)
#
# Run it where cargo may build (a compute node). Engines (override by env):
#   LMP     LAMMPS with MOLECULE (lmp); `bond_style fene`, `pair_style
#           lj/smooth/linear`, `buck`, `morse` are in its standard packages
#   PYTHON  python with openmm 8.x and molrs (python3)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/without_slurm_step.sh
LMP=${LMP:-lmp}
PYTHON=${PYTHON:-python3}
pin=
[[ ${1:-} == --pin ]] && pin=crates/molrs/src/ff/testdata/engine_codecs/engines.tsv
dir=${FF_CODEC_DIR:-$(mktemp -d)}
[[ -n ${FF_CODEC_DIR:-} || -n ${KEEP:-} ]] && echo "work directory: $dir" || trap 'rm -rf "$dir"' EXIT
mkdir -p "$dir"

MOLRS_ENGINE_CODEC_DIR="$dir" cargo mrs-test -- \
    ff::engine_codec_check::write_engine_inputs --exact >/dev/null

for case in "$dir"/*/; do
    case=${case%/}
    [[ -f $case/system.ff ]] || continue
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
thermo_style    custom step ebond eangle evdwl pe
thermo_modify   format float %.17g
run             0
IN
        (cd "$case" && without_slurm_step "$LMP" -in "in.$k" -log "log.$k" -screen none </dev/null) ||
            { echo "lmp failed on $case/in.$k" >&2; tail -20 "$case/log.$k" >&2; exit 1; }
    done
done

"$PYTHON" scripts/ff_engine_codec_check.py collect "$dir" ${pin:+--pin "$pin"}
