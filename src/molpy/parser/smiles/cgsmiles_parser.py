"""CGSmiles parser implementation."""

from pathlib import Path
from typing import Any

from lark import Token, Transformer

from molpy.parser.base import GrammarConfig, GrammarParserBase

from .cgsmiles_ir import (
    BondOrder,
    CGSmilesBondIR,
    CGSmilesFragmentIR,
    CGSmilesGraphIR,
    CGSmilesIR,
    CGSmilesNodeIR,
)


class CGSmilesTransformer(Transformer):
    """Transformer for CGSmiles grammar to CGSmilesIR."""

    def __init__(self):
        super().__init__()
        # Ring openings shared across the entire molecule
        self.ring_openings: dict[str, tuple[CGSmilesNodeIR, BondOrder]] = {}

    @staticmethod
    def _token_value(item: Token | str | int | None) -> Any:
        """Extract value from token or return as-is."""
        if isinstance(item, Token):
            return item.value
        return item

    @staticmethod
    def _bond_from_symbol(symbol: str | None) -> BondOrder:
        """Convert bond symbol to bond order."""
        if symbol is None:
            return 1
        mapping = {
            ".": 0,
            "-": 1,
            "=": 2,
            "#": 3,
            "$": 4,
        }
        return mapping.get(symbol, 1)

    def INT(self, n: Token) -> int:
        return int(n.value)

    def NAME(self, s: Token) -> str:
        return str(s.value)

    def VALUE(self, s: Token) -> str:
        return str(s.value)

    def fragment_body(self, children: list) -> str:
        """Parse fragment body from regex match."""
        # children[0] is the Token from the regex
        from lark import Token

        child = children[0]
        if isinstance(child, Token):
            return str(child.value).strip()
        return str(child).strip()

    def node_label(self, children: list) -> str:
        """Extract node label (without the # prefix)."""
        # children = ['#', NAME]
        # Return the NAME (second child), but Token inherits from str!
        from lark import Token

        for child in children:
            if isinstance(child, str) and not isinstance(child, Token):
                return child
        return ""

    def annotation(self, children: list) -> tuple[str, str]:
        """Parse annotation as (key, value) tuple."""
        # children = [NAME, '=', VALUE]
        # Filter out Token objects
        from lark import Token

        values = [
            c for c in children if isinstance(c, str) and not isinstance(c, Token)
        ]
        if len(values) >= 2:
            return (values[0], values[1])
        return ("", "")

    def node_annots(self, children: list) -> dict[str, str]:
        """Convert list of annotations to dict."""
        # children = [';', annotation, ';', annotation, ...]
        # Filter out semicolon tokens and keep only tuples
        annots = [c for c in children if isinstance(c, tuple)]
        return dict(annots)

    def node(self, children: list) -> CGSmilesNodeIR:
        """Parse node: [#LABEL] or [#LABEL;key=value;...]"""
        # children = ['[', node_label, node_annots?, ']']
        # Filter out tokens and get label and annotations
        # Note: Token inherits from str, so we must check explicitly!
        from lark import Token

        label = None
        annotations = {}
        for child in children:
            if isinstance(child, str) and not isinstance(child, Token):
                if label is None:
                    label = child
            elif isinstance(child, dict):
                annotations = child
        return CGSmilesNodeIR(label=label or "", annotations=annotations)

    def bond(self, children: list) -> str:
        """Parse bond symbol."""
        # children contains the bond token
        from lark import Token

        if children:
            child = children[0]
            if isinstance(child, Token):
                return str(child.value)
            return str(child)
        return "-"

    def repeat_op(self, children: list) -> int:
        """Parse repeat operator |INT."""
        # children = ['|', INT]
        # Return the INT value (second child)
        from lark import Token

        for child in children:
            if isinstance(child, int):
                return child
            elif isinstance(child, Token) and child.type == "INT":
                return int(child.value)
        return 1

    def RING_BOND(self, token: Token) -> tuple[str, BondOrder]:
        """Parse RING_BOND terminal."""
        value = str(token.value)
        # RING_BOND format: [bond_symbol]?INT or [bond_symbol]?%INT
        # Extract bond symbol and ring number
        bond_symbol = None
        ring_num = value

        # Check for bond symbol at start
        if value and value[0] in ".=-#$":
            bond_symbol = value[0]
            ring_num = value[1:]

        bond_order = self._bond_from_symbol(bond_symbol) if bond_symbol else 1
        return (ring_num, bond_order)

    def ring_bond(self, children: list) -> tuple[str, BondOrder]:
        """Parse ring_bond rule."""
        # children[0] is the result from RING_BOND terminal
        return children[0]

    def atom(
        self, children: list
    ) -> tuple[CGSmilesNodeIR, list[tuple[str, BondOrder]]]:
        """Parse atom: node ring_bond*."""
        node = children[0]
        ring_bonds = children[1:] if len(children) > 1 else []
        return (node, ring_bonds)

    def branch(self, children: list) -> tuple[BondOrder, CGSmilesGraphIR]:
        """Parse branch: (bond? chain)."""
        # children = ['(', bond?, chain, ')']
        # Filter out Token objects
        from lark import Token

        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            # No explicit bond: (chain)
            bond_order = 1
            graph = filtered[0]
        else:
            # Explicit bond: (bond chain)
            bond_symbol = filtered[0]
            bond_order = self._bond_from_symbol(bond_symbol)
            graph = filtered[1]
        return (bond_order, graph)

    def branched_atom(self, children: list) -> dict[str, Any]:
        """Parse branched_atom: atom branch* repeat_op?."""
        from lark import Token

        result = {
            "atom": None,
            "branches": [],
            "repeat": 1,
        }

        for child in children:
            # Skip Token objects that are just syntax
            if isinstance(child, Token):
                continue
            elif isinstance(child, tuple) and len(child) == 2:
                if isinstance(child[0], CGSmilesNodeIR):
                    # It's an atom (node, ring_bonds)
                    result["atom"] = child
                elif isinstance(child[1], CGSmilesGraphIR):
                    # It's a branch (bond_order, graph)
                    result["branches"].append(child)
            elif isinstance(child, int):
                # Repeat operator result
                result["repeat"] = child

        return result

    def chain(self, children: list) -> CGSmilesGraphIR:
        """Parse chain: branched_atom (bond? branched_atom)*."""
        graph = CGSmilesGraphIR()
        prev_node = None
        prev_bond_order = 1

        for child in children:
            if isinstance(child, str):
                # It's a bond symbol
                prev_bond_order = self._bond_from_symbol(child)
            elif isinstance(child, dict):
                # It's a branched_atom
                branched_atom = child
                repeat = branched_atom["repeat"]
                atom_data = branched_atom["atom"]
                branches = branched_atom["branches"]

                if atom_data is None:
                    continue

                node_template, ring_bonds = atom_data

                # Repeat the branched_atom
                for r in range(repeat):
                    # Create node
                    # Defensive check: ensure annotations is a dict
                    if not isinstance(node_template.annotations, dict):
                        annots = {}
                    else:
                        annots = node_template.annotations.copy()

                    node = CGSmilesNodeIR(
                        label=node_template.label,
                        annotations=annots,
                    )
                    graph.nodes.append(node)

                    # Connect to previous node
                    if prev_node is not None:
                        bond = CGSmilesBondIR(
                            node_i=prev_node,
                            node_j=node,
                            order=prev_bond_order,
                        )
                        graph.bonds.append(bond)
                        prev_bond_order = 1  # Reset to default

                    # Handle ring closures
                    for ring_num, ring_bond_order in ring_bonds:
                        if ring_num in self.ring_openings:
                            # Close the ring
                            ring_node, opening_bond_order = self.ring_openings[ring_num]
                            # Use the bond order from opening or closing (prefer non-default)
                            final_order = (
                                ring_bond_order
                                if ring_bond_order != 1
                                else opening_bond_order
                            )
                            bond = CGSmilesBondIR(
                                node_i=ring_node,
                                node_j=node,
                                order=final_order,
                            )
                            graph.bonds.append(bond)
                            del self.ring_openings[ring_num]
                        else:
                            # Open the ring
                            self.ring_openings[ring_num] = (node, ring_bond_order)

                    # Process branches
                    for branch_bond_order, branch_graph in branches:
                        # Merge branch nodes and bonds into main graph
                        node_offset = len(graph.nodes)

                        # Add all branch nodes
                        for branch_node in branch_graph.nodes:
                            new_node = CGSmilesNodeIR(
                                label=branch_node.label,
                                annotations=branch_node.annotations.copy(),
                            )
                            graph.nodes.append(new_node)

                        # Add all branch bonds (with offset indices)
                        for branch_bond in branch_graph.bonds:
                            # Find the indices in the original branch
                            i_idx = branch_graph.nodes.index(branch_bond.node_i)
                            j_idx = branch_graph.nodes.index(branch_bond.node_j)
                            # Map to new nodes
                            new_i = graph.nodes[node_offset + i_idx]
                            new_j = graph.nodes[node_offset + j_idx]
                            new_bond = CGSmilesBondIR(
                                node_i=new_i,
                                node_j=new_j,
                                order=branch_bond.order,
                            )
                            graph.bonds.append(new_bond)

                        # Connect first node of branch to current node
                        if branch_graph.nodes:
                            first_branch_node = graph.nodes[node_offset]
                            bond = CGSmilesBondIR(
                                node_i=node,
                                node_j=first_branch_node,
                                order=branch_bond_order,
                            )
                            graph.bonds.append(bond)

                    prev_node = node

        return graph

    def graph(self, children: list[CGSmilesGraphIR]) -> CGSmilesGraphIR:
        """Parse graph (just returns the chain)."""
        return children[0] if children else CGSmilesGraphIR()

    def base_graph(self, children: list) -> CGSmilesGraphIR:
        """Parse base_graph: { graph? }."""
        # children = ['{', graph?, '}']
        # Filter out tokens and get the graph
        for child in children:
            if isinstance(child, CGSmilesGraphIR):
                return child
        return CGSmilesGraphIR()

    def fragment_def(self, children: list) -> CGSmilesFragmentIR:
        """Parse fragment definition: #NAME=BODY."""
        # children = ['#', NAME, '=', fragment_body]
        # Filter out Token objects
        from lark import Token

        values = [
            c for c in children if isinstance(c, str) and not isinstance(c, Token)
        ]
        if len(values) >= 2:
            name = values[0]
            body = values[1]
        else:
            name = ""
            body = ""
        return CGSmilesFragmentIR(name=name, body=body)

    def fragment_block(self, children: list) -> list[CGSmilesFragmentIR]:
        """Parse fragment_block: { fragment_def, ... }."""
        # Filter out tokens ('{', ',', '}') and keep only CGSmilesFragmentIR
        return [child for child in children if isinstance(child, CGSmilesFragmentIR)]

    def cgsmiles(self, children: list) -> CGSmilesIR:
        """Parse complete CGSmiles: base_graph (.fragment_block)*."""
        base_graph = children[0]
        fragments = []

        for child in children[1:]:
            if isinstance(child, list):
                fragments.extend(child)

        # Check for unclosed rings
        if self.ring_openings:
            unclosed = ", ".join(self.ring_openings.keys())
            raise ValueError(f"Unclosed ring(s): {unclosed}")

        return CGSmilesIR(base_graph=base_graph, fragments=fragments)


class CGSmilesParserImpl(GrammarParserBase[CGSmilesIR]):
    """CGSmiles parser implementation."""

    def __init__(self):
        config = GrammarConfig(
            grammar_path=Path(__file__).parent / "grammars" / "cgsmiles.lark",
            start="cgsmiles",
            parser="lalr",
            propagate_positions=False,
            maybe_placeholders=False,
            auto_reload=True,
        )
        super().__init__(config)
        self.transformer = CGSmilesTransformer()

    def parse(self, src: str) -> CGSmilesIR:
        """Parse a CGSmiles string.

        Args:
            src: CGSmiles string

        Returns:
            CGSmilesIR

        Raises:
            ValueError: if syntax errors detected or unclosed rings
        """
        # Create a fresh transformer for each parse to avoid state pollution
        transformer = CGSmilesTransformer()

        try:
            tree = self.parse_tree(src)
            result = transformer.transform(tree)
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse CGSmiles: {e}") from e


# Global parser instance
_parser = CGSmilesParserImpl()


def parse_cgsmiles(src: str) -> CGSmilesIR:
    """Parse a CGSmiles string.

    Args:
        src: CGSmiles string (e.g., "{[#PEO][#PMA]}.{#PEO=[$]COC[$]}")

    Returns:
        CGSmilesIR with base graph and fragment definitions

    Raises:
        ValueError: if syntax errors detected

    Examples:
        >>> result = parse_cgsmiles("{[#PEO][#PMA][#PEO]}")
        >>> len(result.base_graph.nodes)
        3
        >>> result = parse_cgsmiles("{[#PEO]|5}")
        >>> len(result.base_graph.nodes)
        5
    """
    return _parser.parse(src)
