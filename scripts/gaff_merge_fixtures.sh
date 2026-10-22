#!/usr/bin/env bash
# Build the fixtures of molrs's GAFF merge check (crates/molrs/src/io/amber/
# gaff_merge_check.rs): for each molecule and GAFF set, antechamber's mol2 and
# tleap's prmtop.
#
# Run under AmberTools (antechamber, parmchk2, tleap; AMBERHOME set and its
# bin on PATH), e.g. with the conda env the fixtures were made in:
#
#     export AMBERHOME=/path/to/envs/molcrafts-tools PATH=$AMBERHOME/bin:$PATH
#     scripts/gaff_merge_fixtures.sh
#
# The fixtures in the repo were made with AmberTools 26.0 (conda-forge
# `ambertools` 26.0). Inputs are the hand-built `<mol>.pdb` files beside
# them: ethanol, and TFSI (the bis(trifluoromethanesulfonyl)imide anion),
# which needs parmchk2's estimates. Per molecule and set:
#
#     antechamber -fi pdb -at <set> -c gas   -> <mol>_<set>.mol2
#     parmchk2 -s <set>                      -> frcmod (not kept)
#     tleap: source leaprc.<set>, loadamberparams, loadmol2, saveamberparm
#                                            -> <mol>_<set>.parm7
#
# Gasteiger charges keep antechamber fast; charges take no part in the check.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../crates/molrs/src/io/amber/testdata/gaff"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

for mol in ethanol tfsi; do
    charge=0
    [ "$mol" = tfsi ] && charge=-1
    for set in gaff gaff2; do
        out="${mol}_${set}"
        (
            cd "$work"
            antechamber -i "$OLDPWD/$mol.pdb" -fi pdb -o "$out.mol2" -fo mol2 \
                -at "$set" -c gas -nc "$charge" -rn MOL -s 0 > antechamber.log 2>&1
            parmchk2 -i "$out.mol2" -f mol2 -o "$out.frcmod" -s "$set"
            cat > leap.in <<EOF
source leaprc.$set
loadamberparams $out.frcmod
m = loadmol2 $out.mol2
saveamberparm m $out.parm7 $out.rst7
quit
EOF
            tleap -f leap.in > leap.log 2>&1
            grep -q "Errors = 0" leap.log || { cat leap.log; exit 1; }
        )
        cp "$work/$out.mol2" "$work/$out.parm7" .
    done
done
