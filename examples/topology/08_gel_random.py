"""N2 — Random network to a target conversion (Flory–Stockmayer style).

Guide: docs/user-guide/topology/08_gel_random.md
Run:   python topology/08_gel_random.py
"""

import molpy as mp
from molpy.builder.assembly import GraphAssembler, RandomSelector, Replicas

from eo_kit import XLINK, eo_builder, mark_backbone_crosslink_sites


def main() -> None:
    strand = eo_builder().build_linear("EO", 8)
    mark_backbone_crosslink_sites(strand, step=2)
    melt = Replicas(strand).grid(2, spacing=6.0, jitter=0.4, seed=3)
    n0 = melt.n_atoms

    gel = GraphAssembler(mp.Reaction(XLINK)).assemble(
        melt,
        RandomSelector(
            conversion=0.5,
            seed=7,
            cutoff=6.0,
            exclude_same_molecule=True,
        ),
    )
    n_xlink = (n0 - gel.n_atoms) // 2
    print(
        f"random gel (conv=0.5, seed=7): {n0} → {gel.n_atoms} atoms  (~{n_xlink} xlinks)"
    )

    gel2 = GraphAssembler(mp.Reaction(XLINK)).assemble(
        melt,
        RandomSelector(
            conversion=0.5,
            seed=7,
            cutoff=6.0,
            exclude_same_molecule=True,
        ),
    )
    print(f"  reproducible: {gel.n_atoms == gel2.n_atoms}")


if __name__ == "__main__":
    main()
