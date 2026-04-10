"""Shared base transformer for SMILES and BigSMILES grammars.

Extracts the ~20 methods common to SmilesTransformer and BigSmilesTransformer
so each subclass only carries its unique logic.
"""

from __future__ import annotations

from typing import Any, Literal

from lark import Token, Transformer

from .smiles_ir import BondOrder, SmilesAtomIR


class BaseSmilesTransformer(Transformer):
    """Base transformer with methods shared across SMILES and BigSMILES grammars.

    Subclasses must implement:
        - smiles(): assemble the top-level IR from branched atoms
        - branch(): handle branch parentheses
    """

    def __init__(self):
        super().__init__()
        self.ring_openings: dict[str, tuple[SmilesAtomIR, str | None]] = {}

    # ------------------------------------------------------------------
    # Token / utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _token_value(item: Token | str | int | None) -> str:
        if isinstance(item, Token):
            return str(item.value)
        if item is None:
            return ""
        return str(item)

    @staticmethod
    def _bond_from_symbol(
        symbol: str | None,
    ) -> tuple[BondOrder, Literal["/", "\\"] | None]:
        if symbol == "/":
            return 1, "/"
        if symbol == "\\":
            return 1, "\\"
        if symbol == "=":
            return 2, None
        if symbol == "#":
            return 3, None
        if symbol == ":":
            return "ar", None
        return 1, None

    # ------------------------------------------------------------------
    # Terminal transformations
    # ------------------------------------------------------------------
    def INT(self, n: Token) -> int:
        return int(n)

    def ATOM_SYM(self, s: Token) -> str:
        return str(s.value)

    def ALIPHATIC_ORGANIC(self, s: Token) -> SmilesAtomIR:
        return SmilesAtomIR(element=s.value, aromatic=False)

    def AROMATIC_ORGANIC(self, s: Token) -> SmilesAtomIR:
        return SmilesAtomIR(element=s.value.upper(), aromatic=True)

    def ELEMENT_SYM(self, s: Token) -> SmilesAtomIR:
        return SmilesAtomIR(element=s.value, aromatic=False)

    def BOND_SYM(self, s: Token) -> str:
        return s.value

    # ------------------------------------------------------------------
    # Atom / element helpers
    # ------------------------------------------------------------------
    def atom_symbol(self, children: list) -> str:
        return self._token_value(children[0] if children else None)

    def element_symbol(self, children: list) -> str:
        return self._token_value(children[0] if children else None)

    def aliphatic_organic(self, children: list) -> SmilesAtomIR:
        return self.ALIPHATIC_ORGANIC(children[0])

    def aromatic_organic(self, children: list) -> SmilesAtomIR:
        return self.AROMATIC_ORGANIC(children[0])

    def aromatic_symbol(self, children: list) -> str:
        return self._token_value(children[0] if children else None)

    def bond_symbol(self, children: list) -> str:
        return self.BOND_SYM(children[0]) if children else "-"

    # ------------------------------------------------------------------
    # Atom construction helpers
    # ------------------------------------------------------------------
    def atom(self, children: list[SmilesAtomIR]) -> SmilesAtomIR:
        return children[0]

    def isotope(self, n: list[int]) -> tuple[str, int]:
        return "isotope", n[0]

    def chiral(self, c: list[Token]) -> tuple[str, str]:
        return "chiral", "".join(t.value for t in c)

    def h_count(self, h: list) -> tuple[str, int]:
        return "h_count", h[1] if len(h) > 1 else 1

    def atom_charge(self, items: list) -> tuple[str, int]:
        sign = -1 if items[0].value == "-" else 1
        if len(items) == 1:
            return "charge", sign
        second = items[1]
        if isinstance(second, Token):
            return "charge", 2 * sign
        val = int(second) if not isinstance(second, int) else second
        return "charge", sign * val

    def atom_class(self, c: list) -> tuple[str, int]:
        return "class_", c[1]

    def bracket_atom(self, items: list) -> SmilesAtomIR:
        """Parse bracket atom.

        Default SMILES implementation.  BigSmilesTransformer overrides this
        to handle BondingDescriptorIR children.
        """
        filtered = [
            it
            for it in items
            if not (isinstance(it, Token) and it.type in {"LSQB", "RSQB"})
        ]
        symbol = None
        props_pairs: list[tuple[str, int | str]] = []
        for it in filtered:
            if isinstance(it, tuple):
                if len(it) == 2:
                    props_pairs.append(it)
            elif symbol is None:
                symbol = it
        if symbol is None:
            raise ValueError("Bracket atom missing symbol")

        element_str = str(symbol)
        aromatic = element_str.islower()
        element = element_str.upper() if aromatic else element_str

        charge = None
        hydrogens = None
        extras: dict[str, Any] = {}

        for k, v in props_pairs:
            if k == "charge":
                charge = int(v)
            elif k == "h_count":
                hydrogens = int(v)
            elif k == "isotope":
                extras["isotope"] = int(v)
            elif k == "chiral":
                extras["chiral"] = str(v)
            elif k == "class_":
                extras["class_"] = int(v)

        return SmilesAtomIR(
            element=element,
            aromatic=aromatic,
            charge=charge,
            hydrogens=hydrogens,
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Ring / assembly helpers
    # ------------------------------------------------------------------
    def ring_bond(self, children: list) -> tuple:
        bond_kind = None
        ring_digits: list[str] = []
        for c in children:
            if isinstance(c, str) and c in {"-", "=", "#", "$", ":", "/", "\\"}:
                bond_kind = c
            elif isinstance(c, Token) and c.type == "DIGIT":
                ring_digits.append(c.value)
        ring_id = (
            "".join(ring_digits)
            if ring_digits
            else (
                children[-1].value
                if isinstance(children[-1], Token)
                else str(children[-1])
            )
        )
        return "ring", ring_id, bond_kind

    def atom_assembly(self, children: list):
        if len(children) == 1:
            return ("asm", None, children[0])
        return ("asm", children[0], children[1])

    def branched_atom(self, children: list) -> tuple:
        node = children[0]
        rings: list = []
        branches: list = []
        for item in children[1:]:
            if isinstance(item, tuple):
                tag = item[0]
                if tag == "ring":
                    rings.append((item[1], item[2]))
                elif tag == "branch":
                    branches.append(item)
            elif isinstance(item, Token) and item.type == "DIGIT":
                rings.append((item.value, None))
        return node, rings or [], branches or []
