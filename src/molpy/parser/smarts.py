from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lark import Token, Transformer

from .base import GrammarConfig, GrammarParserBase

# ===================================================================
#   1. Intermediate Representation (IR) for SMARTS
# ===================================================================


@dataclass(eq=True)
class AtomPrimitiveIR:
    """
    Represents a single primitive atom pattern in SMARTS.

    Examples:
        - symbol='C' (carbon atom)
        - atomic_num=6 (atomic number 6)
        - neighbor_count=3 (X3, exactly 3 neighbors)
        - ring_size=6 (r6, in 6-membered ring)
        - ring_count=2 (R2, in exactly 2 rings)
        - has_label='%atom1' (has label %atom1)
        - matches_smarts=SmartsIR(...) (recursive SMARTS)
    """

    type: Literal[
        "symbol",
        "atomic_num",
        "neighbor_count",
        "ring_size",
        "ring_count",
        "has_label",
        "matches_smarts",
        "wildcard",
        "hydrogen_count",
        "implicit_hydrogen_count",
    ]
    value: str | int | SmartsIR | None = None
    id: int = field(
        default_factory=lambda: id(AtomPrimitiveIR), compare=False, repr=False
    )

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.type == "wildcard":
            return "AtomPrimitiveIR(type='wildcard')"
        return f"AtomPrimitiveIR(type={self.type!r}, value={self.value!r})"


@dataclass(eq=True)
class AtomExpressionIR:
    """
    Represents logical expressions combining atom primitives.

    Operators:
        - 'and' (&): high-priority AND
        - 'or' (,): OR
        - 'weak_and' (;): low-priority AND
        - 'not' (!): negation

    Examples:
        - AtomExpressionIR(op='and', children=[primitive1, primitive2])
        - AtomExpressionIR(op='not', children=[primitive])
    """

    op: Literal["and", "or", "weak_and", "not", "primitive"]
    children: list[AtomPrimitiveIR | AtomExpressionIR] = field(default_factory=list)
    id: int = field(
        default_factory=lambda: id(AtomExpressionIR), compare=False, repr=False
    )

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.op == "primitive" and len(self.children) == 1:
            return f"AtomExpressionIR({self.children[0]!r})"
        return f"AtomExpressionIR(op={self.op!r}, children={self.children!r})"


@dataclass(eq=True)
class SmartsAtomIR:
    """
    Represents a complete SMARTS atom with expression and optional label.

    Attributes:
        expression: The atom pattern expression
        label: Optional numeric label for ring closures or references
    """

    expression: AtomExpressionIR | AtomPrimitiveIR
    label: int | None = None
    id: int = field(default_factory=lambda: id(SmartsAtomIR), compare=False, repr=False)

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.label is not None:
            return f"SmartsAtomIR(expression={self.expression!r}, label={self.label})"
        return f"SmartsAtomIR(expression={self.expression!r})"


@dataclass(eq=True)
class SmartsBondIR:
    """
    Represents a bond between two SMARTS atoms.

    In SMARTS, bonds are implicit (single or aromatic) unless specified.
    Explicit bond types can be specified between atoms.
    """

    start: SmartsAtomIR
    end: SmartsAtomIR
    bond_type: str = "-"  # default single bond, can be =, #, :, ~, @, etc.

    def __repr__(self):
        expr_start = getattr(self.start.expression, "value", str(self.start.expression))
        expr_end = getattr(self.end.expression, "value", str(self.end.expression))
        return f"SmartsBondIR({expr_start!r}, {expr_end!r}, {self.bond_type!r})"


@dataclass(eq=True)
class SmartsIR:
    """
    Complete SMARTS pattern intermediate representation.

    Attributes:
        atoms: List of all atoms in the pattern
        bonds: List of all bonds in the pattern
    """

    atoms: list[SmartsAtomIR] = field(default_factory=list)
    bonds: list[SmartsBondIR] = field(default_factory=list)

    def __repr__(self):
        return f"SmartsIR(atoms={self.atoms!r}, bonds={self.bonds!r})"


# ===================================================================
#   2. SmartsTransformer
# ===================================================================


class SmartsTransformer(Transformer):
    """
    Transforms Lark parse tree into SmartsIR.

    Handles:
        - Atom primitives (symbols, atomic numbers, properties)
        - Logical expressions (AND, OR, NOT, weak AND)
        - Branches
        - Ring closures
        - Recursive SMARTS patterns
    """

    def __init__(self):
        super().__init__()
        # Track ring openings: {ring_id: (atom, bond_type)}
        self.ring_openings: dict[str, tuple[SmartsAtomIR, str | None]] = {}

    # ================== Terminals ==================
    def NUM(self, n: Token) -> int:
        return int(n.value)

    def SYMBOL(self, s: Token) -> str:
        return str(s.value)

    def STAR(self, s: Token) -> str:
        return "*"

    def LABEL(self, s: Token) -> str:
        return str(s.value)

    # ================== Atom Primitives ==================
    def atom_symbol(self, children: list[Token]) -> AtomPrimitiveIR:
        """Process atom symbol (element or wildcard)."""
        symbol = children[0] if isinstance(children[0], str) else children[0].value
        if symbol == "*":
            return AtomPrimitiveIR(type="wildcard")
        return AtomPrimitiveIR(type="symbol", value=symbol)

    def atomic_num(self, children: list[int]) -> int:
        """Extract atomic number."""
        return children[0]

    def neighbor_count(self, children: list[int]) -> int:
        """Extract neighbor count."""
        return children[0]

    def ring_size(self, children: list[int]) -> int:
        """Extract ring size."""
        return children[0]

    def ring_count(self, children: list[int]) -> int:
        """Extract ring count."""
        return children[0]

    def hydrogen_count(self, children: list[int]) -> int:
        """Extract explicit hydrogen count."""
        return children[0]

    def implicit_hydrogen_count(self, children: list[int]) -> int:
        """Extract implicit hydrogen count."""
        return children[0]

    def has_label(self, children: list[str]) -> str:
        """Extract label."""
        return children[0]

    def matches_string(self, children: list) -> SmartsIR:
        """Extract recursive SMARTS pattern."""
        return children[0]

    def atom_id(self, children: list) -> AtomPrimitiveIR:
        """
        Process atom identifier (primitive).

        Can be:
            - atom_symbol
            - # + atomic_num (atomic number)
            - $( + SMARTS + ) (recursive SMARTS)
            - %label (has label)
            - X + N (neighbor count)
            - r + N (ring size)
            - R + N (ring count)
        """
        # Filter out operator tokens like #, X, r, R, H, h, $(, )
        filtered = [
            c
            for c in children
            if not (
                isinstance(c, Token)
                and c.value in {"#", "X", "r", "R", "H", "h", "$(", ")", "$", "("}
            )
        ]

        if not filtered:
            return AtomPrimitiveIR(type="wildcard")

        child = filtered[0]

        # Already processed atom_symbol
        if isinstance(child, AtomPrimitiveIR):
            return child

        # Check the first token in original children to determine type
        first_token = next((c for c in children if isinstance(c, Token)), None)

        if first_token and first_token.value == "#":
            # Atomic number (#N)
            if isinstance(child, int):
                return AtomPrimitiveIR(type="atomic_num", value=child)

        elif first_token and first_token.value in {"$", "$("}:
            # Recursive SMARTS ($(...)
            if isinstance(child, SmartsIR):
                return AtomPrimitiveIR(type="matches_smarts", value=child)

        elif first_token and first_token.value == "X":
            # Neighbor count (XN)
            if isinstance(child, int):
                return AtomPrimitiveIR(type="neighbor_count", value=child)

        elif first_token and first_token.value == "r":
            # Ring size (rN)
            if isinstance(child, int):
                return AtomPrimitiveIR(type="ring_size", value=child)

        elif first_token and first_token.value == "R":
            # Ring count (RN)
            if isinstance(child, int):
                return AtomPrimitiveIR(type="ring_count", value=child)

        elif first_token and first_token.value == "H":
            # Explicit hydrogen count (H2)
            if isinstance(child, int):
                return AtomPrimitiveIR(type="hydrogen_count", value=child)

        elif first_token and first_token.value == "h":
            # Implicit hydrogen count (h2)
            if isinstance(child, int):
                return AtomPrimitiveIR(type="implicit_hydrogen_count", value=child)

        # Label (%label)
        if isinstance(child, str) and child.startswith("%"):
            return AtomPrimitiveIR(type="has_label", value=child)

        # Fallback to symbol
        return AtomPrimitiveIR(type="symbol", value=str(child))

    # ================== Logical Expressions ==================
    def not_expression(self, children: list) -> AtomExpressionIR:
        """Process NOT expression (!)."""
        # Filter out operator tokens like !
        filtered = [c for c in children if not isinstance(c, Token)]
        if not filtered:
            return AtomExpressionIR(op="not", children=[])
        return AtomExpressionIR(op="not", children=[filtered[0]])

    def and_expression(self, children: list) -> AtomExpressionIR:
        """Process high-priority AND expression (&)."""
        # Filter out operator tokens like &
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            # Single primitive, wrap it
            child = filtered[0]
            if isinstance(child, AtomPrimitiveIR):
                return AtomExpressionIR(op="primitive", children=[child])
            return child

        # Flatten nested AND expressions
        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "and":
                # Flatten nested AND
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        # Multiple children connected by &
        return AtomExpressionIR(op="and", children=flat_children)

    def or_expression(self, children: list) -> AtomExpressionIR:
        """Process OR expression (,)."""
        # Filter out operator tokens like ,
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            return filtered[0]

        # Flatten nested OR expressions
        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "or":
                # Flatten nested OR
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        return AtomExpressionIR(op="or", children=flat_children)

    def weak_and_expression(self, children: list) -> AtomExpressionIR:
        """Process low-priority AND expression (;)."""
        # Filter out operator tokens like ;
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            return filtered[0]

        # Flatten nested weak AND expressions
        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "weak_and":
                # Flatten nested weak AND
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        return AtomExpressionIR(op="weak_and", children=flat_children)

    # ================== Atom Struct ==================
    def atom_label(self, children: list[int]) -> int:
        """Extract atom label (numeric)."""
        return children[0]

    def atom(self, children: list) -> SmartsAtomIR:
        """
        Process complete atom: [expression] or symbol, with optional label.

        Returns:
            SmartsAtomIR
        """
        # Filter out bracket tokens
        filtered = [
            c
            for c in children
            if not (isinstance(c, Token) and c.type in {"LSQB", "RSQB"})
        ]

        if not filtered:
            # Empty bracketed atom, use wildcard
            prim = AtomPrimitiveIR(type="wildcard")
            expr = AtomExpressionIR(op="primitive", children=[prim])
            return SmartsAtomIR(expression=expr)

        expression = filtered[0]
        label = filtered[1] if len(filtered) > 1 else None

        # Wrap primitive in expression if needed
        if isinstance(expression, AtomPrimitiveIR):
            expression = AtomExpressionIR(op="primitive", children=[expression])

        return SmartsAtomIR(expression=expression, label=label)

    # ================== Chain Struct ==================
    def branch(self, children: list) -> SmartsIR:
        """
        Process branch: the content inside or after chain.
        This just returns the SmartsIR from _string.
        """
        # Branch contains the result of _string
        return self._build_ir_from_children(children)

    def _chain(self, children: list) -> SmartsIR:
        """
        Process chain: atom+ with implicit bonds.
        """
        return self._build_ir_from_children(children)

    def nonlastbranch(self, children: list) -> tuple:
        """Process non-last branch: (branch_content)."""
        # Filter parentheses and get branch IR
        filtered = [
            c
            for c in children
            if not (isinstance(c, Token) and c.type in {"LPAR", "RPAR"})
        ]
        if filtered and isinstance(filtered[0], SmartsIR):
            return ("branch", filtered[0], None)
        return ("branch", SmartsIR(), None)

    def _string(self, children: list) -> SmartsIR:
        """
        Process complete SMARTS string.

        Returns:
            SmartsIR with all atoms and bonds
        """
        return self._build_ir_from_children(children)

    def start(self, children: list) -> SmartsIR:
        """Entry point: process complete SMARTS pattern.

        The grammar produces a tree like:
        start
          atom ...
          atom ...

        We need to build the IR from this flat or nested structure.
        """
        if not children:
            return SmartsIR()

        # children from start contains result from _string
        # which could be atoms or nested structures
        return self._build_ir_from_children(children)

    def _build_ir_from_children(self, children: list) -> SmartsIR:
        """Build SmartsIR from list of atoms and branches."""
        ir = SmartsIR()
        active_atom: SmartsAtomIR | None = None

        for item in children:
            if isinstance(item, SmartsAtomIR):
                ir.atoms.append(item)

                # Connect to previous atom in chain
                if active_atom is not None:
                    ir.bonds.append(SmartsBondIR(active_atom, item, "-"))

                # Handle ring closures
                if item.label is not None:
                    label_str = str(item.label)
                    if label_str in self.ring_openings:
                        # Close the ring
                        opening_atom, bond_type = self.ring_openings[label_str]
                        ir.bonds.append(
                            SmartsBondIR(opening_atom, item, bond_type or "-")
                        )
                        del self.ring_openings[label_str]
                    else:
                        # Open a new ring
                        self.ring_openings[label_str] = (item, "-")

                active_atom = item

            elif isinstance(item, tuple) and len(item) == 3 and item[0] == "branch":
                _, branch_ir, branch_bond_type = item
                if isinstance(branch_ir, SmartsIR) and branch_ir.atoms:
                    if active_atom is not None:
                        # Connect branch to current active atom
                        head_atom = branch_ir.atoms[0]
                        ir.bonds.append(
                            SmartsBondIR(
                                active_atom, head_atom, branch_bond_type or "-"
                            )
                        )
                    # Add all branch atoms and bonds
                    ir.atoms.extend(branch_ir.atoms)
                    ir.bonds.extend(branch_ir.bonds)
                    # Don't update active_atom - continue from main chain

            elif isinstance(item, SmartsIR):
                # Merge nested IR (could be from branch or _lastbranch)
                if not ir.atoms:
                    ir = item
                    if item.atoms:
                        active_atom = item.atoms[-1]
                else:
                    if active_atom is not None and item.atoms:
                        ir.bonds.append(SmartsBondIR(active_atom, item.atoms[0], "-"))
                    ir.atoms.extend(item.atoms)
                    ir.bonds.extend(item.bonds)
                    if item.atoms:
                        active_atom = item.atoms[-1]

        return ir


# ===================================================================
#   3. SmartsParser
# ===================================================================


class SmartsParser(GrammarParserBase):
    """
    Main parser for SMARTS patterns.

    Usage:
        parser = SmartsParser()
        ir = parser.parse_smarts("[#6]")
        ir = parser.parse_smarts("c1ccccc1")
        ir = parser.parse_smarts("[C,N,O]")
    """

    def __init__(self):
        config = GrammarConfig(
            grammar_path=Path(__file__).parent / "grammar" / "smarts.lark",
            start="start",
            parser="earley",
            propagate_positions=True,
            maybe_placeholders=False,
            auto_reload=True,
        )
        super().__init__(config)

    def parse_smarts(self, smarts: str) -> SmartsIR:
        """
        Parse SMARTS string into SmartsIR.

        Args:
            smarts: SMARTS pattern string

        Returns:
            SmartsIR representing the pattern

        Raises:
            ValueError: if parsing fails or rings are unclosed

        Examples:
            >>> parser = SmartsParser()
            >>> ir = parser.parse_smarts("C")
            >>> len(ir.atoms)
            1
            >>> ir = parser.parse_smarts("[#6]")
            >>> ir.atoms[0].expression.children[0].type
            'atomic_num'
        """
        tree = self.parse_tree(smarts)
        transformer = SmartsTransformer()
        ir: SmartsIR = transformer.transform(tree)

        # Check for unclosed rings
        if transformer.ring_openings:
            unclosed = list(transformer.ring_openings.keys())
            raise ValueError(f"Unclosed rings in SMARTS: {unclosed}")

        return ir


def _ir_to_smarts_string(ir: SmartsIR) -> str:
    """
    Convert SmartsIR back to SMARTS string.

    This is a helper for the converter and for debugging.
    """
    if not ir.atoms:
        return ""

    # Build adjacency list
    adj: dict[int, list[tuple[int, str]]] = {id(atom): [] for atom in ir.atoms}
    for bond in ir.bonds:
        start_id = id(bond.start)
        end_id = id(bond.end)
        adj[start_id].append((end_id, bond.bond_type))
        adj[end_id].append((start_id, bond.bond_type))

    # Depth-first traversal to build SMARTS string
    visited: set[int] = set()
    rings: dict[int, int] = {}  # atom_id -> ring_number
    ring_counter = 1

    def atom_to_str(atom: SmartsAtomIR) -> str:
        """Convert atom IR to SMARTS string."""
        expr = atom.expression

        # Simple primitive
        if (
            isinstance(expr, AtomExpressionIR)
            and expr.op == "primitive"
            and len(expr.children) == 1
        ):
            prim = expr.children[0]
            if isinstance(prim, AtomPrimitiveIR):
                if prim.type == "wildcard":
                    return "*"
                elif prim.type == "symbol":
                    return f"[{prim.value}]"
                elif prim.type == "atomic_num":
                    return f"[#{prim.value}]"

        # Complex expression - need to build properly
        return f"[{_expr_to_str(expr)}]"

    def _expr_to_str(expr: AtomExpressionIR | AtomPrimitiveIR) -> str:
        """Convert expression to SMARTS string."""
        if isinstance(expr, AtomPrimitiveIR):
            if expr.type == "wildcard":
                return "*"
            elif expr.type == "symbol":
                return str(expr.value)
            elif expr.type == "atomic_num":
                return f"#{expr.value}"
            elif expr.type == "neighbor_count":
                return f"X{expr.value}"
            elif expr.type == "ring_size":
                return f"r{expr.value}"
            elif expr.type == "ring_count":
                return f"R{expr.value}"
            return str(expr.value)

        if expr.op == "primitive":
            return _expr_to_str(expr.children[0])
        elif expr.op == "not":
            return f"!{_expr_to_str(expr.children[0])}"
        elif expr.op == "and":
            return "&".join(_expr_to_str(c) for c in expr.children)
        elif expr.op == "or":
            return ",".join(_expr_to_str(c) for c in expr.children)
        elif expr.op == "weak_and":
            return ";".join(_expr_to_str(c) for c in expr.children)
        return ""

    def dfs(atom: SmartsAtomIR, parent_id: int | None = None) -> str:
        """Depth-first traversal to build SMARTS string."""
        nonlocal ring_counter

        atom_id = id(atom)
        result = atom_to_str(atom)
        visited.add(atom_id)

        # Process neighbors
        neighbors = adj.get(atom_id, [])
        branches = []

        for neighbor_id, _bond_type in neighbors:
            if neighbor_id == parent_id:
                continue

            neighbor_atom = next(a for a in ir.atoms if id(a) == neighbor_id)

            if neighbor_id in visited:
                # Ring closure
                if neighbor_id not in rings:
                    rings[neighbor_id] = ring_counter
                    ring_num = ring_counter
                    ring_counter += 1
                else:
                    ring_num = rings[neighbor_id]
                result += str(ring_num)
            else:
                # Regular bond
                neighbor_str = dfs(neighbor_atom, atom_id)
                branches.append(neighbor_str)

        # Add branches
        if branches:
            result += "".join(f"({b})" for b in branches[1:])
            if branches:
                result += branches[0]

        return result

    return dfs(ir.atoms[0])
