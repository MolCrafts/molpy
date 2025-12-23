"""SMILES parser implementation."""

import os
from pathlib import Path
from typing import Literal

from lark import Token, Transformer

from molpy.parser.base import GrammarConfig, GrammarParserBase

from .smiles_ir import BondOrder, SmilesAtomIR, SmilesBondIR, SmilesGraphIR


class SmilesTransformer(Transformer):
    """Transformer for SMILES grammar to SmilesGraphIR."""

    def __init__(self):
        super().__init__()
        # Ring openings shared across the entire molecule (allows cross-branch closure)
        self.ring_openings: dict[str, tuple[SmilesAtomIR, str | None]] = {}

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
        if symbol in ("/", "\\"):
            return 1, symbol  # stereo single bond
        if symbol == "=":
            return 2, None
        if symbol == "#":
            return 3, None
        if symbol == ":":
            return "ar", None
        return 1, None

    # ================== Terminal transformation ==================
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

    # ---- unified grammar aliases ----
    def atom_symbol(self, children: list) -> str:
        return self.ATOM_SYM(children[0]) if children else ""

    def element_symbol(self, children: list) -> str:
        token = children[0] if children else ""
        return self._token_value(token)

    def aliphatic_organic(self, children: list) -> SmilesAtomIR:
        return self.ALIPHATIC_ORGANIC(children[0])

    def aromatic_organic(self, children: list) -> SmilesAtomIR:
        return self.AROMATIC_ORGANIC(children[0])

    def aromatic_symbol(self, children: list) -> str:
        token = children[0] if children else ""
        value = self._token_value(token)
        return value

    def bond_symbol(self, children: list) -> str:
        return self.BOND_SYM(children[0]) if children else "-"

    def ring_id(self, d: list[Token]) -> str:
        return "".join(t.value for t in d)

    # ================== Atom and its attributes construction ==================
    def atom(self, children: list[SmilesAtomIR]) -> SmilesAtomIR:
        return children[0]

    def isotope(self, n: list[int]) -> tuple[str, int]:
        return "isotope", n[0]

    def chiral(self, c: list[Token]) -> tuple[str, str]:
        return "chiral", "".join(t.value for t in c)

    def h_count(self, h: list) -> tuple[str, int]:
        return "h_count", h[1] if len(h) > 1 else 1

    def class_(self, c: list) -> tuple[str, int]:
        return "class_", c[1]

    def atom_class(self, c: list) -> tuple[str, int]:
        return "class_", c[1]

    def atom_charge(self, items: list) -> tuple[str, int]:
        sign = -1 if items[0].value == "-" else 1
        # forms: '+' '-' '++' '--' '+2' '-2'
        if len(items) == 1:
            return "charge", sign
        second = items[1]
        if isinstance(second, Token):  # ++ or -- already collapsed by grammar
            return "charge", 2 * sign
        # numeric token already converted? ensure int
        val = int(second) if not isinstance(second, int) else second
        return "charge", sign * val

    def bracket_atom(self, items: list) -> SmilesAtomIR:
        # items include '[' and ']' tokens plus optional property tuples
        filtered = [
            it
            for it in items
            if not (isinstance(it, Token) and it.type in {"LSQB", "RSQB"})
        ]
        # first non-tuple is the element symbol
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

        # Determine element and aromaticity
        element_str = str(symbol)
        aromatic = element_str.islower()
        element = element_str.upper() if aromatic else element_str

        # Extract properties
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

    # ================== Core SMILES assembly ==================
    def smiles(self, children: list) -> SmilesGraphIR:
        """Assemble linear smiles: children = [first_branched_atom, (bond, branched_atom)*]"""
        ir = SmilesGraphIR()
        active_atom: SmilesAtomIR | None = None
        pending_bond_kind: str | None = None
        debug = os.getenv("SMILES_DEBUG")
        if debug:
            print(f"[SMILES] processing {len(children)} children")

        # Normalize children into sequence: branched_atom, (bond, branched_atom)...
        seq: list = []
        if not children:
            return ir
        first = children[0]
        rest = children[1:]
        seq.append(first)
        i = 0
        while i < len(rest):
            item = rest[i]
            if isinstance(item, str):
                # unlikely direct token here, but keep for robustness
                if i + 1 >= len(rest):
                    break
                seq.append(item)
                seq.append(rest[i + 1])
                i += 2
            elif isinstance(item, tuple) and item and item[0] == "asm":
                _, bond, ba = item
                if bond is not None:
                    seq.append(bond)
                seq.append(ba)
                i += 1
            else:
                seq.append(item)
                i += 1

        for item in seq:
            if isinstance(item, str):
                pending_bond_kind = item
                continue
            # Ensure item is a tuple with 3 elements
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            atom, rings, branches = item
            # Ensure rings and branches are lists
            if rings is None:
                rings = []
            if branches is None:
                branches = []

            ir.atoms.append(atom)
            if active_atom is not None:
                order, stereo = self._bond_from_symbol(pending_bond_kind)
                ir.bonds.append(
                    SmilesBondIR(
                        atom_i=active_atom, atom_j=atom, order=order, stereo=stereo
                    )
                )
            active_atom = atom
            pending_bond_kind = None
            for ring_id, bond_kind in rings:
                if active_atom is None:
                    continue
                # Check if ring was opened before, if so close it
                if ring_id in self.ring_openings:
                    start_atom, start_bond = self.ring_openings.pop(ring_id)
                    final_bond = bond_kind or start_bond
                    order, stereo = self._bond_from_symbol(final_bond)
                    ir.bonds.append(
                        SmilesBondIR(
                            atom_i=start_atom,
                            atom_j=active_atom,
                            order=order,
                            stereo=stereo,
                        )
                    )
                    if debug:
                        print(
                            f"[SMILES] close ring {ring_id}: {start_atom.element}->{active_atom.element} bond={final_bond}"
                        )
                else:
                    self.ring_openings[ring_id] = (active_atom, bond_kind)
                    if debug:
                        print(
                            f"[SMILES] open ring {ring_id} at atom {active_atom.element} bond={bond_kind}"
                        )
            if branches:
                for branch_item in branches:
                    if not isinstance(branch_item, tuple) or len(branch_item) != 2:
                        continue
                    branch_ir, branch_bond_kind = branch_item
                    if active_atom is None:
                        continue
                    # Skip empty branches
                    if not hasattr(branch_ir, "atoms") or not branch_ir.atoms:
                        continue
                    head_atom = branch_ir.atoms[0]
                    order, stereo = self._bond_from_symbol(branch_bond_kind)
                    ir.bonds.append(
                        SmilesBondIR(
                            atom_i=active_atom,
                            atom_j=head_atom,
                            order=order,
                            stereo=stereo,
                        )
                    )
                    ir.atoms.extend(branch_ir.atoms)
                    ir.bonds.extend(branch_ir.bonds)

        return ir

    def branched_atom(self, children: list) -> tuple:
        atom = children[0]
        rings: list = []
        branches: list = []
        for item in children[1:]:
            if isinstance(item, tuple):
                tag = item[0]
                if tag == "ring":
                    rings.append((item[1], item[2]))
                elif tag == "branch":
                    branches.append((item[1], item[2]))
            elif isinstance(item, Token) and item.type == "DIGIT":
                rings.append((item.value, None))
        # Ensure rings and branches are always lists, never None
        return atom, rings or [], branches or []

    def ring_bond(self, children: list) -> tuple:
        # children could include optional bond sym and digits, possibly with '%'
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

    def branch(self, children: list) -> tuple:
        # children may include LPAR/RPAR tokens; filter them out
        filtered = [
            c
            for c in children
            if not (isinstance(c, Token) and c.type in {"LPAR", "RPAR"})
        ]
        bond_kind = filtered[0] if filtered and isinstance(filtered[0], str) else None
        smiles_ir = filtered[-1]
        return "branch", smiles_ir, bond_kind

    def atom_assembly(self, children: list):
        # children: [branched_atom] or [bond, branched_atom]
        if len(children) == 1:
            return ("asm", None, children[0])
        return ("asm", children[0], children[1])

    def smiles_mixture(self, children: list) -> SmilesGraphIR | list[SmilesGraphIR]:
        """Transform smiles_mixture: smiles ("." smiles)*

        Returns:
            - SmilesGraphIR if single molecule
            - list[SmilesGraphIR] if multiple molecules (dot-separated)
        """
        # Filter out dot tokens
        graphs = [child for child in children if isinstance(child, SmilesGraphIR)]

        if len(graphs) == 1:
            return graphs[0]
        return graphs


class SmilesParserImpl(GrammarParserBase):
    """SMILES parser implementation."""

    def __init__(self):
        config = GrammarConfig(
            grammar_path=Path(__file__).parent / "grammars" / "smiles.lark",
            start="smiles_mixture",
            parser="earley",
            propagate_positions=True,
            maybe_placeholders=False,
            auto_reload=True,
        )
        super().__init__(config)

    def parse(self, src: str) -> SmilesGraphIR | list[SmilesGraphIR]:
        """
        Parse SMILES string into SmilesGraphIR.

        Args:
            src: SMILES string

        Returns:
            SmilesGraphIR

        Raises:
            ValueError: if syntax errors detected or unclosed rings
        """
        tree = self.parse_tree(src)
        transformer = SmilesTransformer()
        result = transformer.transform(tree)
        # Check for unclosed rings after transformation
        if transformer.ring_openings:
            unclosed = list(transformer.ring_openings.keys())
            raise ValueError(f"Unclosed rings: {unclosed}")

        # Handle both single IR and list of IRs
        if isinstance(result, list):
            # Check for unclosed rings in each graph
            for ir in result:
                # Note: ring_openings is shared, so we check once above
                pass
            return result
        return result
