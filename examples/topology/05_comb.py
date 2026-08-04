"""Comb polymer: backbone with branch points (topology IR, no CGSmiles parser).

Guide: docs/user-guide/topology/05_comb.md
Run:   python topology/05_comb.py
"""

from eo_kit import branch_unit, eo_builder, report
from molpy.builder.assembly import (
    CGSmilesBondIR,
    CGSmilesGraphIR,
    CGSmilesNodeIR,
)


def comb_topology() -> CGSmilesGraphIR:
    """EO–BR–EO–BR–EO backbone with a one-unit graft on each BR."""
    eo1 = CGSmilesNodeIR(label="EO")
    br1 = CGSmilesNodeIR(label="BR")
    g1 = CGSmilesNodeIR(label="EO")
    eo2 = CGSmilesNodeIR(label="EO")
    br2 = CGSmilesNodeIR(label="BR")
    g2 = CGSmilesNodeIR(label="EO")
    eo3 = CGSmilesNodeIR(label="EO")
    nodes = [eo1, br1, g1, eo2, br2, g2, eo3]
    bonds = [
        CGSmilesBondIR(node_i=eo1, node_j=br1),
        CGSmilesBondIR(node_i=br1, node_j=g1),
        CGSmilesBondIR(node_i=br1, node_j=eo2),
        CGSmilesBondIR(node_i=eo2, node_j=br2),
        CGSmilesBondIR(node_i=br2, node_j=g2),
        CGSmilesBondIR(node_i=br2, node_j=eo3),
    ]
    return CGSmilesGraphIR(nodes=nodes, bonds=bonds)


def main() -> None:
    builder = eo_builder(extra={"BR": branch_unit()})
    comb = builder.build(comb_topology())
    report("comb", comb)
    print("  topology: EO-BR(EO)-EO-BR(EO)-EO")


if __name__ == "__main__":
    main()
