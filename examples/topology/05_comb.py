"""Comb polymer: backbone with branch points (topology IR, no CGSmiles parser).

Guide: docs/user-guide/topology/05_comb.md
Run:   python topology/05_comb.py
"""

from eo_kit import branch_unit, eo_builder, report
from molpy.builder.assembly import (
    ResidueBond,
    ResidueTopology,
    ResidueNode,
)


def comb_topology() -> ResidueTopology:
    """EO–BR–EO–BR–EO backbone with a one-unit graft on each BR."""
    eo1 = ResidueNode(label="EO")
    br1 = ResidueNode(label="BR")
    g1 = ResidueNode(label="EO")
    eo2 = ResidueNode(label="EO")
    br2 = ResidueNode(label="BR")
    g2 = ResidueNode(label="EO")
    eo3 = ResidueNode(label="EO")
    nodes = [eo1, br1, g1, eo2, br2, g2, eo3]
    bonds = [
        ResidueBond(node_i=eo1, node_j=br1),
        ResidueBond(node_i=br1, node_j=g1),
        ResidueBond(node_i=br1, node_j=eo2),
        ResidueBond(node_i=eo2, node_j=br2),
        ResidueBond(node_i=br2, node_j=g2),
        ResidueBond(node_i=br2, node_j=eo3),
    ]
    return ResidueTopology(nodes=nodes, bonds=bonds)


def main() -> None:
    builder = eo_builder(extra={"BR": branch_unit()})
    comb = builder.build(comb_topology())
    report("comb", comb)
    print("  topology: EO-BR(EO)-EO-BR(EO)-EO")


if __name__ == "__main__":
    main()
