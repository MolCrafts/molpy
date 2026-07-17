"""N9 — End-linked network: telechelic chains, only end residues react.

Guide: docs/user-guide/topology/09_end_linked.md
Run:   python topology/09_end_linked.py
"""

import molpy as mp
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas

from eo_kit import XLINK, eo_builder, full_library, mark_residue_crosslink_sites


def main() -> None:
    lib = full_library()
    builder = eo_builder(extra={"CAPA": lib["CAPA"], "CAPB": lib["CAPB"]})
    strand = builder.build_sequence(["CAPA"] + ["EO"] * 5 + ["CAPB"])
    mark_residue_crosslink_sites(strand, {"CAPA", "CAPB"}, site="x", leaving="h")
    melt = Replicas(strand).times(6, spacing=5.0)
    n0 = melt.n_atoms

    gel = GraphAssembler(mp.Reaction(XLINK)).assemble(
        melt,
        ExhaustiveSelector(cutoff=8.0, exclude_same_molecule=True),
    )
    n_xlink = (n0 - gel.n_atoms) // 2
    print(f"end-linked: {n0} → {gel.n_atoms} atoms  (~{n_xlink} end-links)")
    print("  marked residues: CAPA + CAPB only")


if __name__ == "__main__":
    main()
