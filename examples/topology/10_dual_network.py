"""N8 — Dual network: two assemble steps, two SITE namespaces.

Guide: docs/user-guide/topology/10_dual_network.md
Run:   python topology/10_dual_network.py
"""

import molpy as mp
from molpy.builder.assembly import (
    ExhaustiveSelector,
    GraphAssembler,
    RandomSelector,
    Replicas,
    SiteMap,
)
from molpy.core import fields

from eo_kit import XLINK, XLINK2, eo_builder, mark_backbone_crosslink_sites


def main() -> None:
    strand = eo_builder().build_linear("EO", 8)
    # First network: every 2nd carbon → site x (partial conversion)
    mark_backbone_crosslink_sites(strand, step=2, site="x", leaving="h")
    melt = Replicas(strand).grid(2, spacing=5.5, jitter=0.5, seed=1)
    n0 = melt.n_atoms

    net1 = GraphAssembler(mp.Reaction(XLINK)).assemble(
        melt,
        RandomSelector(
            conversion=0.4,
            seed=1,
            cutoff=7.0,
            exclude_same_molecule=True,
        ),
    )
    n1 = (n0 - net1.n_atoms) // 2

    # Clear spent first-network labels, then mark a second population as y/k
    for atom in net1.atoms:
        if atom.get(fields.SITE) in ("x", "h"):
            atom[fields.SITE] = ""
    carbons = [
        a
        for a in net1.atoms
        if a.get(fields.ELEMENT) == "C"
        and any(n.get(fields.ELEMENT) == "H" for n in net1.get_neighbors(a))
    ]
    SiteMap(net1).every_nth(carbons, 2, "y", leaving="k", fold_charge=True)

    # After the first network percolates, bond-components are one "molecule",
    # so exclude_same_molecule would forbid every pair. The second pass allows
    # same-component pairs and relies on SITE y/k + cutoff only.
    net2 = GraphAssembler(mp.Reaction(XLINK2)).assemble(
        net1,
        ExhaustiveSelector(cutoff=7.0, exclude_same_molecule=False),
    )
    n2 = (net1.n_atoms - net2.n_atoms) // 2
    print(
        f"dual network: {n0} → {net1.n_atoms} (net1 ~{n1} xlinks) "
        f"→ {net2.n_atoms} (net2 ~{n2} xlinks)"
    )
    print("  site namespaces: x/h then y/k (cleared between steps)")


if __name__ == "__main__":
    main()
