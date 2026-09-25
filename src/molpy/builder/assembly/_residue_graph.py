"""Residue topology constructors for polymer assembly.

A residue topology is a graph whose nodes are monomer copies (residues) and
whose edges say which residues are bonded. The constructors here build the
common shapes — a path, a ring, a star — as a :class:`ResidueTopology`; any
other shape is built by instantiating :class:`ResidueNode` /
:class:`ResidueBond` directly. This module never parses text.
"""

from __future__ import annotations

from collections.abc import Sequence

from molpy.builder.assembly._residue_ir import (
    ResidueBond,
    ResidueTopology,
    ResidueNode,
)


def linear_topology(labels: Sequence[str]) -> ResidueTopology:
    """Path topology: one node per label, edges between consecutive residues.

    Args:
        labels: Monomer library keys, in chain order; each becomes one residue.

    Returns:
        A topology with ``len(labels)`` nodes and ``len(labels) - 1`` bonds.

    Raises:
        ValueError: if ``labels`` is empty.
    """
    if not labels:
        raise ValueError("linear topology needs at least one residue label")
    labels = [str(lab) for lab in labels]
    nodes = [ResidueNode(label=lab) for lab in labels]
    bonds = [
        ResidueBond(node_i=nodes[i], node_j=nodes[i + 1]) for i in range(len(nodes) - 1)
    ]
    return ResidueTopology(nodes=nodes, bonds=bonds)


def ring_topology(label: str, n: int) -> ResidueTopology:
    """Cycle of ``n`` identical residues.

    The path ``1 - 2 - ... - n`` plus one closing bond ``n - 1``. A placer
    forms the closing bond but does not place it; its length is the
    caller's concern (a ring-shaped trace, or geometry optimization after
    assembly).

    Args:
        label: Monomer library key used for every residue.
        n: Number of residues in the ring.

    Returns:
        A topology with ``n`` nodes and ``n`` bonds.

    Raises:
        ValueError: if ``n < 3``.
    """
    if n < 3:
        raise ValueError(f"a residue ring needs n >= 3, got {n}")
    path = linear_topology([str(label)] * n)
    closure = ResidueBond(node_i=path.nodes[-1], node_j=path.nodes[0])
    return ResidueTopology(nodes=path.nodes, bonds=[*path.bonds, closure])


def star_topology(
    core: str,
    arm: str,
    *,
    n_arms: int,
    arm_length: int,
    cap: str | None = None,
) -> ResidueTopology:
    """Star: one core node bonded to ``n_arms`` linear arms of ``arm_length``.

    The core monomer must carry at least ``n_arms`` reaction sites for the
    build to succeed; this constructor does not check that (it knows labels,
    not templates).

    Args:
        core: Monomer library key of the central residue.
        arm: Monomer library key repeated along every arm.
        n_arms: Number of arms bonded to the core.
        arm_length: Residues per arm, not counting the cap.
        cap: Optional monomer library key appended to the end of every arm.

    Returns:
        A topology with ``1 + n_arms * (arm_length + (cap is not None))``
        nodes; the core is ``nodes[0]``.

    Raises:
        ValueError: if ``n_arms < 2`` or ``arm_length < 1``.
    """
    if n_arms < 2:
        raise ValueError(f"a star needs n_arms >= 2, got {n_arms}")
    if arm_length < 1:
        raise ValueError(f"arm_length must be >= 1, got {arm_length}")

    core_node = ResidueNode(label=str(core))
    nodes: list[ResidueNode] = [core_node]
    bonds: list[ResidueBond] = []

    for _ in range(n_arms):
        prev = core_node
        for _j in range(arm_length):
            node = ResidueNode(label=str(arm))
            nodes.append(node)
            bonds.append(ResidueBond(node_i=prev, node_j=node))
            prev = node
        if cap is not None:
            cap_node = ResidueNode(label=str(cap))
            nodes.append(cap_node)
            bonds.append(ResidueBond(node_i=prev, node_j=cap_node))

    return ResidueTopology(nodes=nodes, bonds=bonds)
