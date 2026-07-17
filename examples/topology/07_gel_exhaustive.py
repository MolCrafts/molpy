"""N1/C2 — Crosslinked gel: build → mark → Replicas → ExhaustiveSelector.

Guide: docs/user-guide/topology/07_gel_exhaustive.md
Run:   python topology/07_gel_exhaustive.py
"""

import molpy as mp
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas

from eo_kit import XLINK, eo_builder, mark_backbone_crosslink_sites


def main() -> None:
    strand = eo_builder().build_linear("EO", 8)
    mark_backbone_crosslink_sites(strand, step=2)
    melt = Replicas(strand).grid(2, spacing=6.0, jitter=0.4, seed=3)
    n0 = melt.n_atoms

    gel = GraphAssembler(mp.Reaction(XLINK)).assemble(
        melt,
        ExhaustiveSelector(cutoff=6.0, exclude_same_molecule=True),
    )
    n_xlink = (n0 - gel.n_atoms) // 2
    print(f"exhaustive gel: {n0} → {gel.n_atoms} atoms  (~{n_xlink} crosslinks)")
    print(f"  chains in melt: {2**3} (grid 2³)")


if __name__ == "__main__":
    main()
