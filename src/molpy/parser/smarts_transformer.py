"""SMARTS transformer: converts Lark parse tree into SmartsIR."""

from __future__ import annotations

from lark import Token, Transformer

from .smarts_ir import (
    AtomExpressionIR,
    AtomPrimitiveIR,
    SmartsAtomIR,
    SmartsBondIR,
    SmartsIR,
)


class SmartsTransformer(Transformer):
    """Transforms Lark parse tree into SmartsIR.

    Handles:
        - Atom primitives (symbols, atomic numbers, properties)
        - Logical expressions (AND, OR, NOT, weak AND)
        - Branches
        - Ring closures
        - Recursive SMARTS patterns
    """

    def __init__(self):
        super().__init__()
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
        symbol = children[0] if isinstance(children[0], str) else children[0].value
        if symbol == "*":
            return AtomPrimitiveIR(type="wildcard")
        return AtomPrimitiveIR(type="symbol", value=symbol)

    def atomic_num(self, children: list[int]) -> int:
        return children[0]

    def neighbor_count(self, children: list[int]) -> int:
        return children[0]

    def ring_size(self, children: list[int]) -> int:
        return children[0]

    def ring_count(self, children: list[int]) -> int:
        return children[0]

    def hydrogen_count(self, children: list[int]) -> int:
        return children[0]

    def implicit_hydrogen_count(self, children: list[int]) -> int:
        return children[0]

    def has_label(self, children: list[str]) -> str:
        return children[0]

    def ring_connectivity(self, children: list[int]) -> int:
        return children[0]

    def degree(self, children: list[int]) -> int | None:
        return children[0] if children else None

    def valence(self, children: list[int]) -> int | None:
        return children[0] if children else None

    def charge(self, children: list) -> AtomPrimitiveIR:
        tokens = [
            c for c in children if isinstance(c, Token) and c.type == "CHARGE_SIGN"
        ]
        num = next((c for c in children if isinstance(c, int)), None)

        if tokens:
            sign = tokens[0].value
            if len(tokens) == 2:
                value = 2 if sign == "+" else -2
            elif num is not None:
                value = num if sign == "+" else -num
            else:
                value = 1 if sign == "+" else -1
        else:
            value = 0

        return AtomPrimitiveIR(type="charge", value=value)

    def chirality(self, children: list) -> AtomPrimitiveIR:
        tokens = [c for c in children if isinstance(c, Token) and c.value == "@"]
        if len(tokens) >= 2:
            return AtomPrimitiveIR(type="chirality", value="@@")
        return AtomPrimitiveIR(type="chirality", value="@")

    def isotope(self, children: list[int]) -> int:
        return children[0]

    def atom_class(self, children: list[str]) -> str | None:
        return children[0] if children else None

    def bond(self, children: list) -> str:
        has_negation = False
        bond_char = "-"

        for token in children:
            if isinstance(token, Token):
                if token.value == "!":
                    has_negation = True
                else:
                    bond_char = token.value

        if has_negation:
            return f"!{bond_char}"
        return bond_char

    def matches_string(self, children: list) -> SmartsIR:
        return children[0]

    def atom_id(self, children: list) -> AtomPrimitiveIR:
        """Process atom identifier (primitive)."""
        first_token = next((c for c in children if isinstance(c, Token)), None)

        if first_token and first_token.value == "a":
            return AtomPrimitiveIR(type="aromatic", value=True)

        if first_token and first_token.value == "A":
            return AtomPrimitiveIR(type="aliphatic", value=True)

        if isinstance(children[0], AtomPrimitiveIR) and children[0].type == "chirality":
            return children[0]

        if isinstance(children[0], AtomPrimitiveIR) and children[0].type == "charge":
            return children[0]

        filtered = [
            c
            for c in children
            if not (
                isinstance(c, Token)
                and c.value
                in {
                    "#",
                    "X",
                    "x",
                    "r",
                    "R",
                    "H",
                    "h",
                    "D",
                    "v",
                    "a",
                    "A",
                    "$(",
                    ")",
                    "$",
                    "(",
                }
            )
        ]

        if not filtered:
            if first_token:
                token_map = {
                    "X": ("neighbor_count", None),
                    "x": ("ring_connectivity", None),
                    "r": ("ring_size", None),
                    "R": ("ring_count", None),
                    "H": ("hydrogen_count", None),
                    "h": ("implicit_hydrogen_count", None),
                    "D": ("degree", None),
                    "v": ("valence", None),
                }
                if first_token.value in token_map:
                    ptype, pval = token_map[first_token.value]
                    return AtomPrimitiveIR(type=ptype, value=pval)
            return AtomPrimitiveIR(type="wildcard")

        child = filtered[0]

        if isinstance(child, AtomPrimitiveIR):
            return child

        if isinstance(child, str) and not (first_token and first_token.value == "%"):
            if child.startswith("%"):
                return AtomPrimitiveIR(type="has_label", value=child)
            else:
                if len(child) <= 2 and child[0].isupper():
                    return AtomPrimitiveIR(type="symbol", value=child)
                else:
                    return AtomPrimitiveIR(type="atom_class", value=child)

        if first_token and first_token.value == "#":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="atomic_num", value=child)

        elif first_token and first_token.value in {"$", "$("}:
            if isinstance(child, SmartsIR):
                return AtomPrimitiveIR(type="matches_smarts", value=child)

        elif first_token and first_token.value == "X":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="neighbor_count", value=child)

        elif first_token and first_token.value == "x":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="ring_connectivity", value=child)

        elif first_token and first_token.value == "r":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="ring_size", value=child)

        elif first_token and first_token.value == "R":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="ring_count", value=child)

        elif first_token and first_token.value == "H":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="hydrogen_count", value=child)

        elif first_token and first_token.value == "h":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="implicit_hydrogen_count", value=child)

        elif first_token and first_token.value == "D":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="degree", value=child)

        elif first_token and first_token.value == "v":
            if isinstance(child, int):
                return AtomPrimitiveIR(type="valence", value=child)

        if isinstance(child, str) and child.startswith("%"):
            return AtomPrimitiveIR(type="has_label", value=child)

        return AtomPrimitiveIR(type="symbol", value=str(child))

    # ================== Logical Expressions ==================
    def not_expression(self, children: list) -> AtomExpressionIR:
        filtered = [c for c in children if not isinstance(c, Token)]
        if not filtered:
            return AtomExpressionIR(op="not", children=[])
        return AtomExpressionIR(op="not", children=[filtered[0]])

    def isotope_atom(self, children: list) -> AtomPrimitiveIR:
        isotope_val = None
        symbol_child = None
        for c in children:
            if isinstance(c, int) and isotope_val is None:
                isotope_val = c
            if isinstance(c, AtomPrimitiveIR) and c.type == "symbol":
                symbol_child = c
        if isotope_val is not None and symbol_child is not None:
            return AtomPrimitiveIR(
                type="isotope", value=(isotope_val, symbol_child.value)
            )
        if symbol_child is not None:
            return symbol_child
        return AtomPrimitiveIR(type="wildcard")

    def implicit_and(self, children: list) -> AtomExpressionIR | AtomPrimitiveIR:
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            child = filtered[0]
            if isinstance(child, AtomPrimitiveIR):
                return child
            return child

        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "and":
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        return AtomExpressionIR(op="and", children=flat_children)

    def and_expression(self, children: list) -> AtomExpressionIR:
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            child = filtered[0]
            if isinstance(child, AtomPrimitiveIR):
                return AtomExpressionIR(op="primitive", children=[child])
            return child

        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "and":
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        return AtomExpressionIR(op="and", children=flat_children)

    def or_expression(self, children: list) -> AtomExpressionIR:
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            return filtered[0]

        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "or":
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        return AtomExpressionIR(op="or", children=flat_children)

    def weak_and_expression(self, children: list) -> AtomExpressionIR:
        filtered = [c for c in children if not isinstance(c, Token)]

        if len(filtered) == 1:
            return filtered[0]

        flat_children = []
        for child in filtered:
            if isinstance(child, AtomExpressionIR) and child.op == "weak_and":
                flat_children.extend(child.children)
            else:
                flat_children.append(child)

        return AtomExpressionIR(op="weak_and", children=flat_children)

    # ================== Atom Struct ==================
    def atom_label(self, children: list[int]) -> int:
        return children[0]

    def bare_atom(self, children: list) -> AtomPrimitiveIR:
        if children and isinstance(children[0], AtomPrimitiveIR):
            return children[0]
        if children and isinstance(children[0], str):
            name = children[0]
            if len(name) <= 2 and name[0].isupper():
                return AtomPrimitiveIR(type="symbol", value=name)
            return AtomPrimitiveIR(type="atom_class", value=name)
        return AtomPrimitiveIR(type="wildcard")

    # ------------------------------------------------------------------
    # Bare-H disambiguation
    # ------------------------------------------------------------------
    #
    # In SMARTS the token ``H`` is overloaded:
    #
    #   [H]    → the hydrogen **element** (atomic number 1)
    #   [CH]   → carbon with exactly 1 hydrogen neighbour  (= H1)
    #   [C;H]  → same (weak-AND variant)
    #   [!H]   → not a hydrogen atom
    #   [CH2]  → carbon with exactly 2 hydrogen neighbours
    #
    # The Lark grammar always emits ``hydrogen_count`` for the ``H``
    # token.  ``value=None`` means "bare H, no digit".  We fix the
    # semantics here by inspecting the surrounding expression:
    #
    #   * ``hydrogen_count(None)`` with **no** element/symbol sibling
    #     → rewrite to ``symbol('H')``  (hydrogen element)
    #   * ``hydrogen_count(None)`` **with** an element/symbol sibling
    #     → rewrite to ``hydrogen_count(1)``

    @staticmethod
    def _has_element(expr) -> bool:
        """Return True if *expr* (or any descendant) contains a symbol
        or atomic_num primitive."""
        if isinstance(expr, AtomPrimitiveIR):
            return expr.type in ("symbol", "atomic_num")
        if isinstance(expr, AtomExpressionIR):
            return any(SmartsTransformer._has_element(c) for c in expr.children)
        return False

    @staticmethod
    def _rewrite_bare_h(expr, has_elem: bool):
        """Recursively rewrite ``hydrogen_count(None)``."""
        if isinstance(expr, AtomPrimitiveIR):
            if expr.type == "hydrogen_count" and expr.value is None:
                if has_elem:
                    return AtomPrimitiveIR(type="hydrogen_count", value=1)
                else:
                    return AtomPrimitiveIR(type="symbol", value="H")
            return expr
        if isinstance(expr, AtomExpressionIR):
            # For compound expressions, check element presence among
            # siblings so that ``[CH]`` → symbol + H1, ``[!H]`` → !symbol.
            elem_here = SmartsTransformer._has_element(expr)
            new_children = [
                SmartsTransformer._rewrite_bare_h(c, has_elem or elem_here)
                for c in expr.children
            ]
            return AtomExpressionIR(op=expr.op, children=new_children)
        return expr

    def atom(self, children: list) -> SmartsAtomIR:
        filtered = [
            c
            for c in children
            if not (isinstance(c, Token) and c.type in {"LSQB", "RSQB"})
        ]

        if not filtered:
            prim = AtomPrimitiveIR(type="wildcard")
            expr = AtomExpressionIR(op="primitive", children=[prim])
            return SmartsAtomIR(expression=expr)

        expression = filtered[0]
        label = filtered[1] if len(filtered) > 1 else None

        if isinstance(expression, AtomPrimitiveIR):
            expression = AtomExpressionIR(op="primitive", children=[expression])

        # Disambiguate bare H (hydrogen_count=None)
        expression = self._rewrite_bare_h(expression, self._has_element(expression))

        return SmartsAtomIR(expression=expression, label=label)

    # ================== Chain Struct ==================
    def branch(self, children: list) -> SmartsIR:
        return self._build_ir_from_children(children)

    def _chain(self, children: list) -> SmartsIR:
        return self._build_ir_from_children(children)

    def nonlastbranch(self, children: list) -> tuple:
        filtered = [
            c
            for c in children
            if not (isinstance(c, Token) and c.type in {"LPAR", "RPAR"})
        ]

        bond_type = None
        branch_ir = None

        if filtered:
            if isinstance(filtered[0], str):
                bond_type = filtered[0]
                if len(filtered) > 1 and isinstance(filtered[1], SmartsIR):
                    branch_ir = filtered[1]
            elif isinstance(filtered[0], SmartsIR):
                branch_ir = filtered[0]

        if branch_ir is None:
            branch_ir = SmartsIR()

        return ("branch", branch_ir, bond_type)

    def _string(self, children: list) -> SmartsIR:
        return self._build_ir_from_children(children)

    def start(self, children: list) -> SmartsIR:
        if not children:
            return SmartsIR()
        return self._build_ir_from_children(children)

    def _build_ir_from_children(self, children: list) -> SmartsIR:
        """Build SmartsIR from list of atoms, bonds, and branches."""
        ir = SmartsIR()
        active_atom: SmartsAtomIR | None = None
        pending_bond_type: str | None = None

        for item in children:
            if isinstance(item, str):
                pending_bond_type = item
                continue

            if isinstance(item, SmartsAtomIR):
                ir.atoms.append(item)

                if active_atom is not None:
                    bond_type = pending_bond_type if pending_bond_type else "implicit"
                    ir.bonds.append(SmartsBondIR(active_atom, item, bond_type))
                    pending_bond_type = None

                if item.label is not None:
                    label_str = str(item.label)
                    if label_str in self.ring_openings:
                        opening_atom, bond_type = self.ring_openings[label_str]
                        ir.bonds.append(
                            SmartsBondIR(opening_atom, item, bond_type or "implicit")
                        )
                        del self.ring_openings[label_str]
                    else:
                        self.ring_openings[label_str] = (item, "implicit")

                active_atom = item

            elif isinstance(item, tuple) and len(item) == 3 and item[0] == "branch":
                _, branch_ir, branch_bond_type = item
                if isinstance(branch_ir, SmartsIR) and branch_ir.atoms:
                    if active_atom is not None:
                        head_atom = branch_ir.atoms[0]
                        bond_type = (
                            pending_bond_type
                            if pending_bond_type
                            else (branch_bond_type or "implicit")
                        )
                        ir.bonds.append(SmartsBondIR(active_atom, head_atom, bond_type))
                        pending_bond_type = None
                    ir.atoms.extend(branch_ir.atoms)
                    ir.bonds.extend(branch_ir.bonds)

            elif isinstance(item, SmartsIR):
                if not ir.atoms:
                    ir = item
                    if item.atoms:
                        active_atom = item.atoms[-1]
                else:
                    if active_atom is not None and item.atoms:
                        bond_type = (
                            pending_bond_type if pending_bond_type else "implicit"
                        )
                        ir.bonds.append(
                            SmartsBondIR(active_atom, item.atoms[0], bond_type)
                        )
                        pending_bond_type = None
                    ir.atoms.extend(item.atoms)
                    ir.bonds.extend(item.bonds)
                    if item.atoms:
                        active_atom = item.atoms[-1]

        return ir
