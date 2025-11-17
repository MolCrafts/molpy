import os
from dataclasses import dataclass, field
from pathlib import Path

from lark import Token, Transformer

from molpy.core.atomistic import Atom, Atomistic
from molpy.core.wrappers.monomer import Monomer

from .base import GrammarConfig, GrammarParserBase


class MolPyAPIError(Exception):
    """Custom exception for API errors."""

    pass


# ===================================================================
#   1. Intermediate Representation (IR) data classes (SMILES part only)
# ===================================================================
@dataclass(eq=True)
class AtomIR:
    symbol: str
    isotope: int | None = None
    chiral: str | None = None
    h_count: int | None = None
    charge: int | None = None
    class_: int | None = None
    id: int = field(default_factory=lambda: id(AtomIR), compare=False, repr=False)

    def __hash__(self):
        return self.id

    def __repr__(self):
        attrs = [f"symbol={self.symbol!r}"]
        if self.isotope is not None:
            attrs.append(f"isotope={self.isotope}")
        if self.chiral is not None:
            attrs.append(f"chiral={self.chiral!r}")
        if self.h_count is not None:
            attrs.append(f"h_count={self.h_count}")
        if self.charge is not None:
            attrs.append(f"charge={self.charge}")
        return f"AtomIR({', '.join(attrs)})"


@dataclass(eq=True)
class BondIR:
    start: AtomIR
    end: AtomIR
    kind: str

    def __repr__(self):
        return f"BondIR({self.start.symbol!r}, {self.end.symbol!r}, {self.kind!r})"


@dataclass(eq=True)
class SmilesIR:
    atoms: list[AtomIR] = field(default_factory=list)
    bonds: list[BondIR] = field(default_factory=list)

    def __repr__(self):
        return f"SmilesIR(atoms={self.atoms!r}, bonds={self.bonds!r})"


# ===================================================================
#   2. SmilesTransformer (base class)
# ===================================================================
class SmilesTransformer(Transformer):
    def __init__(self):
        super().__init__()
        # Ring openings shared across the entire molecule (allows cross-branch closure)
        self.ring_openings: dict[str, tuple[AtomIR, str | None]] = {}

    # ================== Terminal transformation ==================
    def INT(self, n: Token) -> int:
        return int(n)

    def ATOM_SYM(self, s: Token) -> str:
        return str(s.value)

    def ALIPHATIC_ORGANIC(self, s: Token) -> AtomIR:
        return AtomIR(symbol=s.value)

    def AROMATIC_ORGANIC(self, s: Token) -> AtomIR:
        return AtomIR(symbol=s.value)

    def ELEMENT_SYM(self, s: Token) -> AtomIR:
        return AtomIR(symbol=s.value)

    def BOND_SYM(self, s: Token) -> str:
        return s.value

    def bond_symbol(self, children: list[Token]) -> str:
        return children[0].value

    def ring_id(self, d: list[Token]) -> str:
        return "".join(t.value for t in d)

    # ================== Atom and its attributes construction ==================
    def atom(self, children: list[AtomIR]) -> AtomIR:
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

    def bracket_atom(self, items: list) -> AtomIR:
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
        # coerce to correct types
        kwargs: dict[str, object] = {}
        for k, v in props_pairs:
            if k in {"isotope", "h_count", "charge", "class_"}:
                kwargs[k] = int(v)
            elif k == "chiral":
                kwargs[k] = str(v)
        return AtomIR(symbol=str(symbol), **kwargs)  # type: ignore[arg-type]

    # ================== Core SMILES assembly ==================
    def smiles(self, children: list) -> SmilesIR:
        """Assemble linear smiles: children = [first_branched_atom, (bond, branched_atom)*]"""
        ir = SmilesIR()
        active_atom: AtomIR | None = None
        pending_bond_kind = "-"
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
            atom, rings, branches = item

            # In BigSMILES, atom might be BondDescriptorIR (endpoint marker)
            # These should not be added to the atoms list or create bonds
            if isinstance(atom, BondDescriptorIR):
                # Skip bond descriptors in SMILES chains (they're markers, not atoms)
                # Reset active atom and bond kind
                active_atom = None
                pending_bond_kind = "-"
                continue

            ir.atoms.append(atom)
            if active_atom is not None:
                ir.bonds.append(BondIR(active_atom, atom, pending_bond_kind))
            active_atom = atom
            pending_bond_kind = "-"
            for ring_id, bond_kind in rings:
                if active_atom is None:
                    continue
                # Check if ring was opened before, if so close it
                if ring_id in self.ring_openings:
                    start_atom, start_bond = self.ring_openings.pop(ring_id)
                    final_bond = bond_kind or start_bond or "-"
                    ir.bonds.append(BondIR(start_atom, active_atom, final_bond))
                    if debug:
                        print(
                            f"[SMILES] close ring {ring_id}: {start_atom.symbol}->{active_atom.symbol} bond={final_bond}"
                        )
                else:
                    self.ring_openings[ring_id] = (active_atom, bond_kind)
                    if debug:
                        print(
                            f"[SMILES] open ring {ring_id} at atom {active_atom.symbol} bond={bond_kind}"
                        )
            for branch_ir, branch_bond_kind in branches:
                if active_atom is None:
                    continue
                head_atom = branch_ir.atoms[0]
                ir.bonds.append(BondIR(active_atom, head_atom, branch_bond_kind or "-"))
                ir.atoms.extend(branch_ir.atoms)
                ir.bonds.extend(branch_ir.bonds)

        # Note: We don't check for unclosed rings here because rings can span across branches
        # e.g., C1CCC2C(C1)CCC2 has ring 1 opened in main chain and closed in a branch
        # Ring validation should be done at the parser level, not transformer level
        if debug:
            print("[SMILES] exit")
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
        return atom, rings, branches

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


# ===================================================================
#   3. BigSMILES IR (extension)
# ===================================================================
@dataclass
class BondDescriptorIR:
    symbol: str | None = None
    index: int | None = None
    generation: list[int] | None = None


@dataclass
class StochasticDistributionIR:
    name: str
    params: list[float]


@dataclass
class RepeatSegmentIR:
    stochastic_objects: list["StochasticObjectIR"]
    implicit_smiles: SmilesIR | None = None


@dataclass
class BigSmilesChainIR:
    start_smiles: SmilesIR
    repeat_segments: list[RepeatSegmentIR]


@dataclass
class StochasticObjectIR:
    left_descriptor: BondDescriptorIR
    right_descriptor: BondDescriptorIR
    repeat_units: list[SmilesIR | BigSmilesChainIR]
    end_groups: list[SmilesIR | BigSmilesChainIR] | None = None
    distribution: StochasticDistributionIR | None = None


@dataclass
class BigSmilesMoleculeIR:
    chain: BigSmilesChainIR


@dataclass(eq=True)
class BigSmilesIR(SmilesIR):
    """BigSMILES IR extends SmilesIR with polymer-specific chain structure."""

    chain: BigSmilesChainIR | None = None

    def __post_init__(self):
        # Ensure chain is initialized if not provided
        if self.chain is None:
            empty_smiles = SmilesIR(atoms=[], bonds=[])
            object.__setattr__(
                self,
                "chain",
                BigSmilesChainIR(start_smiles=empty_smiles, repeat_segments=[]),
            )

    def degenerate(self) -> SmilesIR:
        """
        Remove bond descriptors, return clean chemical structure.

        This strips away BigSMILES-specific markers ([<], [>], [$])
        leaving only the chemical structure.

        Returns:
            SmilesIR with only AtomIR and BondIR (no BondDescriptorIR)

        Examples:
            >>> ir = parse_bigsmiles("{[<]CC[>]}")
            >>> clean_ir = ir.degenerate()
            >>> len(clean_ir.atoms)  # Only 2 carbons, no descriptors
            2
        """
        # Filter out BondDescriptorIR from atoms
        real_atoms = [a for a in self.atoms if isinstance(a, AtomIR)]

        # Filter bonds that connect to descriptors
        real_bonds = [
            b
            for b in self.bonds
            if isinstance(b.start, AtomIR) and isinstance(b.end, AtomIR)
        ]

        return SmilesIR(atoms=real_atoms, bonds=real_bonds)


# ===================================================================
#   4. BigSmilesTransformer (subclass)
# ===================================================================
class BigSmilesTransformer(SmilesTransformer):
    def NUMBER(self, n: Token) -> float:
        return float(n)

    def bond_descriptor_symbol(self, t: list[Token]) -> str:
        return t[0].value

    # ================== Bond descriptor assembly ==================
    def bond_descriptor_symbol_idx(self, items: list) -> tuple:
        return items[0], items[1] if len(items) > 1 else None

    def bond_descriptor_generation(self, items: list) -> list[int]:
        # Filter out pipe tokens and whitespace, extract numbers
        numbers = [
            it
            for it in items
            if isinstance(it, (int, float))
            or (isinstance(it, Token) and it.type == "NUMBER")
        ]
        return [int(float(n.value) if isinstance(n, Token) else n) for n in numbers]

    def inner_bond_descriptor(self, items: list) -> BondDescriptorIR:
        # items: [bond_descriptor_symbol_idx, [bond_descriptor_generation]]
        symbol, index = items[0] if isinstance(items[0], tuple) else (None, None)
        gen = items[1] if len(items) > 1 else None
        return BondDescriptorIR(symbol=symbol, index=index, generation=gen)  # type: ignore[arg-type]

    def simple_bond_descriptor(self, items: list) -> BondDescriptorIR:
        # Filter out brackets
        filtered = [
            it
            for it in items
            if not (isinstance(it, Token) and it.type in {"LSQB", "RSQB"})
        ]
        return filtered[0] if filtered else BondDescriptorIR()

    def bond_descriptor(self, items: list) -> BondDescriptorIR:
        # bond_descriptor: simple_bond_descriptor | ladder_bond_descriptor | non_covalent_bond_descriptor
        return items[0]

    def terminal_bond_descriptor(self, items: list) -> BondDescriptorIR:
        # Filter out bracket tokens
        filtered = [
            it
            for it in items
            if not (isinstance(it, Token) and it.type in {"LSQB", "RSQB"})
        ]

        symbol, index, gen = None, None, None
        if filtered:
            if isinstance(filtered[0], tuple):  # symbol_idx
                symbol, index = filtered[0]
                if len(filtered) > 1:
                    gen = filtered[1]
            elif isinstance(filtered[0], list):  # generation only
                gen = filtered[0]
        return BondDescriptorIR(symbol=symbol, index=index, generation=gen)  # type: ignore[arg-type]

    # ================== Stochastic object assembly ==================
    def stochastic_distribution(self, items: list) -> StochasticDistributionIR:
        return items[0]

    def flory_schulz(self, p: list) -> StochasticDistributionIR:
        return StochasticDistributionIR("flory_schulz", [p[0]])

    def schulz_zimm(self, p: list) -> StochasticDistributionIR:
        return StochasticDistributionIR("schulz_zimm", p)

    def poisson(self, p: list) -> StochasticDistributionIR:
        return StochasticDistributionIR("poisson", [p[0]])

    def gauss(self, p: list) -> StochasticDistributionIR:
        return StochasticDistributionIR("gauss", p)

    def uniform(self, p: list) -> StochasticDistributionIR:
        return StochasticDistributionIR("uniform", p)

    def log_normal(self, p: list) -> StochasticDistributionIR:
        return StochasticDistributionIR("log_normal", p)

    def _repeat_units(self, items: list) -> list:
        return items

    def _end_group(self, items: list) -> list:
        return items

    def stochastic_generation(self, items: list):
        # Contains stochastic_distribution
        return items[0] if items else None

    def stochastic_object(self, items: list) -> StochasticObjectIR:
        # Filter out tokens (braces, WS, pipes, brackets, commas, semicolons)
        filtered = [
            it
            for it in items
            if not isinstance(it, Token)
            or it.type
            not in {
                "LBRACE",
                "RBRACE",
                "WS_INLINE",
                "VBAR",
                "LSQB",
                "RSQB",
                "COMMA",
                "SEMICOLON",
            }
        ]

        # Grammar: "{" terminal_bond_descriptor _repeat_units [_end_group] terminal_bond_descriptor "}" [stochastic_generation]
        # Since _repeat_units and _end_group are inlined:
        # - _repeat_units becomes: repeat_unit_item, repeat_unit_item, ...
        # - _end_group (if present) is preceded by a ";" token
        # So we expect: left_descriptor, [repeat units...], [end groups...], right_descriptor, [distribution]

        if len(filtered) < 2:
            raise ValueError(
                f"Stochastic object needs at least left and right descriptors. Got {len(filtered)}"
            )

        # First element is left descriptor
        left_desc = filtered[0]
        if not isinstance(left_desc, BondDescriptorIR):
            raise ValueError(
                f"Expected BondDescriptorIR as first element, got {type(left_desc)}"
            )

        # Last element could be distribution or right descriptor
        # Work backwards to find right descriptor
        right_desc = None
        dist = None
        right_idx = len(filtered) - 1

        # Check if last element is a distribution
        if isinstance(filtered[-1], StochasticDistributionIR):
            dist = filtered[-1]
            right_idx -= 1

        # Next from end should be right descriptor
        if right_idx >= 1 and isinstance(filtered[right_idx], BondDescriptorIR):
            right_desc = filtered[right_idx]
        else:
            raise ValueError("Missing right bond descriptor in stochastic object")

        # Everything between left and right descriptors are repeat units (and possibly end groups)
        middle_items = filtered[1:right_idx]

        # Separate repeat units from end groups
        # End groups are after a semicolon token, but since we filtered tokens,
        # we need to check the original items list for semicolon position
        semicolon_idx = None
        for i, it in enumerate(items):
            if isinstance(it, Token) and it.type == "SEMICOLON":
                semicolon_idx = i
                break

        if semicolon_idx is not None:
            # Split middle_items into repeat units and end groups
            # This is tricky - need to count which filtered items came before/after semicolon
            # For now, assume no end groups (will implement later)
            repeat_units = middle_items
            end_groups = None
        else:
            repeat_units = middle_items
            end_groups = None

        # Wrap single items in list if needed
        if not isinstance(repeat_units, list):
            repeat_units = [repeat_units] if repeat_units else []

        return StochasticObjectIR(
            left_descriptor=left_desc,
            right_descriptor=right_desc,
            repeat_units=repeat_units,  # type: ignore[arg-type]
            end_groups=end_groups,
            distribution=dist,
        )

    # ================== Top-level assembly ==================
    def repeat_segment(self, items: list) -> RepeatSegmentIR:
        smiles = items.pop() if items and isinstance(items[-1], SmilesIR) else None
        return RepeatSegmentIR(stochastic_objects=items, implicit_smiles=smiles)

    def big_smiles_chain(self, items: list) -> BigSmilesChainIR:
        # Grammar: smiles? repeat_segment+ | smiles
        # Cases:
        # 1. [smiles, repeat_seg, ...] - smiles followed by segments
        # 2. [repeat_seg, repeat_seg, ...] - only segments (no leading smiles)
        # 3. [smiles] - only smiles (no segments)

        if not items:
            # Empty chain
            return BigSmilesChainIR(
                start_smiles=SmilesIR(atoms=[], bonds=[]), repeat_segments=[]
            )

        # Check if first item is SmilesIR
        if isinstance(items[0], SmilesIR):
            start_smiles_candidate = items[0]
            repeat_segments = items[1:] if len(items) > 1 else []

            # Filter out Stochastic Objects and Bond Descriptors from start_smiles
            # These should not be in the start_smiles atoms list
            real_atoms = [
                a for a in start_smiles_candidate.atoms if isinstance(a, AtomIR)
            ]
            real_bonds = [
                b
                for b in start_smiles_candidate.bonds
                if isinstance(b.start, AtomIR) and isinstance(b.end, AtomIR)
            ]

            # If start_smiles has StochasticObjectIR, create repeat segments from them
            stochastic_from_smiles = [
                a
                for a in start_smiles_candidate.atoms
                if isinstance(a, StochasticObjectIR)
            ]
            if stochastic_from_smiles:
                # Create repeat segments from these stochastic objects
                new_segments = [
                    RepeatSegmentIR(stochastic_objects=[obj], implicit_smiles=None)
                    for obj in stochastic_from_smiles
                ]
                repeat_segments = new_segments + repeat_segments

            start_smiles = SmilesIR(atoms=real_atoms, bonds=real_bonds)
        elif isinstance(items[0], RepeatSegmentIR):
            # No leading smiles
            start_smiles = SmilesIR(atoms=[], bonds=[])
            repeat_segments = items
        else:
            # Shouldn't happen, but handle gracefully
            start_smiles = SmilesIR(atoms=[], bonds=[])
            repeat_segments = []

        return BigSmilesChainIR(
            start_smiles=start_smiles, repeat_segments=repeat_segments
        )

    def big_smiles_molecule(self, items: list) -> BigSmilesMoleculeIR:
        return BigSmilesMoleculeIR(chain=items[0])

    def big_smiles(self, items: list) -> list:
        return items


class SmilesParser(GrammarParserBase):
    def __init__(self):
        config = GrammarConfig(
            grammar_path=Path(__file__).parent / "grammar" / "smiles.lark",
            start="smiles",
            parser="earley",
            propagate_positions=True,
            maybe_placeholders=False,
            auto_reload=True,
        )
        super().__init__(config)

    def parse_smiles(self, smiles: str) -> SmilesIR:
        # Support disconnected components separated by '.' by parsing each separately
        if "." in smiles:
            raise MolPyAPIError(
                "Disconnected components ('.') not supported in this parser method, use parse_dot_smiles instead."
            )
        tree = self.parse_tree(smiles)
        transformer = SmilesTransformer()
        ir: SmilesIR = transformer.transform(tree)
        # Check for unclosed rings after transformation
        if transformer.ring_openings:
            unclosed = list(transformer.ring_openings.keys())
            raise ValueError(f"Unclosed rings: {unclosed}")
        return ir

    def parse_dot_smiles(self, smiles: str) -> list[SmilesIR]:
        """
        Parse SMILES string with disconnected components separated by '.'.

        Args:
            smiles: SMILES string with possible disconnected components

        Returns:
            List of SmilesIR for each component

        Raises:
            ValueError: if syntax errors detected
        """
        parts = [p.strip() for p in smiles.split(".") if p.strip()]
        results: list[SmilesIR] = []
        for part in parts:
            tree = self.parse_tree(part)
            transformer = SmilesTransformer()
            ir: SmilesIR = transformer.transform(tree)
            # Check for unclosed rings after transformation
            if transformer.ring_openings:
                unclosed = list(transformer.ring_openings.keys())
                raise ValueError(f"Unclosed rings in component '{part}': {unclosed}")
            results.append(ir)
        return results

    def parse_bigsmiles(self, text: str) -> BigSmilesIR:
        """
        Parse BigSMILES string into BigSmilesIR.

        Args:
            text: BigSMILES string (can be plain SMILES or BigSMILES with {})

        Returns:
            BigSmilesIR with chain structure

        Raises:
            ValueError: if syntax errors detected
        """
        if not text:
            # Empty string -> empty IR
            empty_smiles = SmilesIR(atoms=[], bonds=[])
            chain = BigSmilesChainIR(start_smiles=empty_smiles, repeat_segments=[])
            return BigSmilesIR(atoms=[], bonds=[], chain=chain)

        # Check if this is plain SMILES (no BigSMILES syntax)
        has_bigsmiles_syntax = any(c in text for c in "{}")

        if not has_bigsmiles_syntax:
            # Plain SMILES - parse as SMILES and wrap in BigSmilesIR
            smiles_ir = self.parse_smiles(text)
            chain = BigSmilesChainIR(start_smiles=smiles_ir, repeat_segments=[])
            return BigSmilesIR(
                atoms=smiles_ir.atoms, bonds=smiles_ir.bonds, chain=chain
            )

        # Full BigSMILES parsing with stochastic objects
        # Parse with big_smiles_molecule start rule
        config = GrammarConfig(
            grammar_path=Path(__file__).parent / "grammar" / "smiles.lark",
            start="big_smiles_molecule",
            parser="earley",
            propagate_positions=True,
            maybe_placeholders=False,
            auto_reload=True,
        )
        temp_parser = GrammarParserBase(config)
        tree = temp_parser.parse_tree(text)

        # Transform using BigSmilesTransformer
        transformer = BigSmilesTransformer()
        result = transformer.transform(tree)

        # result is BigSmilesMoleculeIR, extract chain
        if isinstance(result, BigSmilesMoleculeIR):
            chain = result.chain
            # Collect all atoms and bonds from chain
            atoms = list(chain.start_smiles.atoms)
            bonds = list(chain.start_smiles.bonds)

            for seg in chain.repeat_segments:
                for obj in seg.stochastic_objects:
                    for unit in obj.repeat_units:
                        if isinstance(unit, SmilesIR):
                            atoms.extend(unit.atoms)
                            bonds.extend(unit.bonds)
                    if obj.end_groups:
                        for eg in obj.end_groups:
                            if isinstance(eg, SmilesIR):
                                atoms.extend(eg.atoms)
                                bonds.extend(eg.bonds)
                if seg.implicit_smiles:
                    atoms.extend(seg.implicit_smiles.atoms)
                    bonds.extend(seg.implicit_smiles.bonds)

            return BigSmilesIR(atoms=atoms, bonds=bonds, chain=chain)
        else:
            # Fallback - shouldn't happen but handle gracefully
            empty_smiles = SmilesIR(atoms=[], bonds=[])
            chain = BigSmilesChainIR(start_smiles=empty_smiles, repeat_segments=[])
            return BigSmilesIR(atoms=[], bonds=[], chain=chain)


# ===================================================================
#   Converter: SmilesIR -> RDKit Mol
# ===================================================================


def smilesir_to_mol(ir: SmilesIR) -> "Chem.Mol":
    """
    Convert SmilesIR to RDKit Mol by directly constructing the molecule graph.

    This approach preserves IR-specific information and supports extended syntax
    (BigSMILES, G-BigSMILES) where explicit topology is essential.

    Args:
        ir: SmilesIR instance with atoms and bonds

    Returns:
        RDKit Mol object

    Raises:
        ImportError: if RDKit is not available
        ValueError: if IR contains invalid molecular data

    Example:
        >>> parser = SmilesParser()
        >>> ir = parser.parser_smiles("CCO")
        >>> mol = smilesir_to_mol(ir)
        >>> mol.GetNumAtoms()
        3
    """
    assert isinstance(ir, SmilesIR), "Input must be a SmilesIR instance"

    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError("RDKit is required for smilesir_to_mol conversion") from e

    if not ir.atoms:
        # Empty molecule
        return Chem.Mol()

    # Bond type mapping
    bond_type_map = {
        "-": Chem.BondType.SINGLE,
        "=": Chem.BondType.DOUBLE,
        "#": Chem.BondType.TRIPLE,
        ":": Chem.BondType.AROMATIC,
        "/": Chem.BondType.SINGLE,  # Stereochemistry, treat as single for now
        "\\": Chem.BondType.SINGLE,  # Stereochemistry, treat as single for now
    }

    # Create editable molecule
    mol = Chem.RWMol()

    # Map AtomIR -> RDKit atom index (using object identity)
    atom_to_idx: dict[int, int] = {}

    # Add atoms
    for atom_ir in ir.atoms:
        # Handle aromatic symbols (lowercase in SMILES → uppercase + aromatic flag)
        symbol = atom_ir.symbol.upper() if atom_ir.symbol.islower() else atom_ir.symbol
        is_aromatic = atom_ir.symbol.islower()

        # Create RDKit atom
        rdkit_atom = Chem.Atom(symbol)

        # Set properties
        if atom_ir.charge is not None:
            rdkit_atom.SetFormalCharge(atom_ir.charge)

        if atom_ir.isotope is not None:
            rdkit_atom.SetIsotope(atom_ir.isotope)

        if atom_ir.h_count is not None:
            rdkit_atom.SetNumExplicitHs(atom_ir.h_count)

        # Handle chirality
        if atom_ir.chiral is not None:
            if atom_ir.chiral == "@":
                rdkit_atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif atom_ir.chiral == "@@":
                rdkit_atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            # Other chiral tags can be added as needed

        # Set aromaticity
        if is_aromatic:
            rdkit_atom.SetIsAromatic(True)

        # Add atom and store mapping (use id() for object identity)
        atom_idx = mol.AddAtom(rdkit_atom)
        atom_to_idx[id(atom_ir)] = atom_idx

    # Add bonds
    for bond_ir in ir.bonds:
        start_idx = atom_to_idx.get(id(bond_ir.start))
        end_idx = atom_to_idx.get(id(bond_ir.end))

        if start_idx is None or end_idx is None:
            raise ValueError(f"Bond references unknown atom: {bond_ir}")

        # Determine bond kind (upgrade single bonds between aromatic atoms to aromatic)
        bond_kind_str = bond_ir.kind
        if (
            bond_kind_str == "-"
            and bond_ir.start.symbol.islower()
            and bond_ir.end.symbol.islower()
        ):
            # Single bond between aromatic atoms → aromatic bond
            rdkit_bond_type = Chem.BondType.AROMATIC
        else:
            rdkit_bond_type = bond_type_map.get(bond_kind_str)
            if rdkit_bond_type is None:
                raise ValueError(f"Unknown bond kind: {bond_kind_str}")

        mol.AddBond(start_idx, end_idx, rdkit_bond_type)

    # Convert to immutable Mol
    final_mol = mol.GetMol()

    # Sanitize molecule (compute aromaticity, implicit Hs, etc.)
    try:
        Chem.SanitizeMol(final_mol)
    except Exception as e:
        # If sanitization fails, return unsanitized molecule with warning
        import warnings

        warnings.warn(
            f"Molecule sanitization failed: {e}. Returning unsanitized molecule.",
            stacklevel=2,
        )

    return final_mol


# Note: Converter registration is handled by molpy.adapter module
# Parser module should NOT import adapter to avoid circular dependencies


@dataclass
class PolymerSegment:
    """
    Polymer segment corresponding to one stochastic object {...}.

    Represents a block in block copolymers or the entire polymer for homopolymers.

    Attributes:
        monomers: List of monomers in this segment
        composition_type: How monomers are arranged ("random", "alternating", "statistical", None)
        distribution_params: Distribution parameters if specified (e.g., ratio, probability)
        end_groups: Optional end-group structures
    """

    monomers: list[Monomer[Atomistic]]
    composition_type: str | None = None
    distribution_params: dict | None = None
    end_groups: list[Monomer[Atomistic]] = field(default_factory=list)


@dataclass
class PolymerSpec:
    """
    Complete polymer specification from BigSMILES.

    Describes the full polymer structure including topology and composition.

    Attributes:
        segments: List of polymer segments (blocks)
        topology: Overall polymer topology
            - "homopolymer": Single repeat unit
            - "random_copolymer": Multiple repeat units, random arrangement
            - "block_copolymer": Multiple segments/blocks
            - "alternating_copolymer": Alternating repeat units
        all_monomers: Flattened list of all unique monomers

    Examples:
        >>> # Homopolymer: {[<]CC[>]}
        >>> spec.topology == "homopolymer"
        >>> len(spec.segments) == 1
        >>> len(spec.segments[0].monomers) == 1

        >>> # Random copolymer: {[<]CC[>],[<]OCC[>]}
        >>> spec.topology == "random_copolymer"
        >>> len(spec.segments) == 1
        >>> len(spec.segments[0].monomers) == 2

        >>> # Block copolymer: {[<]CC[>]}{[<]OCC[>]}
        >>> spec.topology == "block_copolymer"
        >>> len(spec.segments) == 2
    """

    segments: list[PolymerSegment]
    topology: str
    all_monomers: list[Monomer[Atomistic]] = field(default_factory=list)

    def __post_init__(self):
        """Compute all_monomers from segments."""
        if not self.all_monomers:
            self.all_monomers = [
                monomer for segment in self.segments for monomer in segment.monomers
            ]


def bigsmilesir_to_monomer(ir: BigSmilesIR) -> Monomer[Atomistic]:
    """
    Convert BigSmilesIR to Monomer (topology only).

    Single responsibility: IR → Monomer conversion only.
    Parsing should be done separately.

    Supports two input formats:
    1. BigSMILES with stochastic object: {[<]CC[>]} (ONE repeat unit only)
    2. Plain SMILES with atom class ports: CCCCO[*:1], CC(C[*:2])O[*:3]

    Args:
        ir: BigSmilesIR from parser

    Returns:
        Monomer[Atomistic] with ports set, NO positions

    Raises:
        ValueError: If IR contains multiple repeat units (use bigsmilesir_to_polymerspec instead)

    Examples:
        >>> parser = SmilesParser()

        >>> # BigSMILES format
        >>> ir = parser.parse_bigsmiles("{[<]CC[>]}")
        >>> monomer = bigsmilesir_to_monomer(ir)
        >>> monomer.port_names()
        ['in', 'out']

        >>> # Plain SMILES with atom class ports
        >>> ir = parser.parse_bigsmiles("CCCCO[*:1]")
        >>> monomer = bigsmilesir_to_monomer(ir)
        >>> monomer.port_names()
        ['port_1']
        >>> monomer.ports['port_1']  # Points to the [*:1] atom
        <Atom object>

    Note:
        To add coordinates:
        1. Get ChemMol: mol = smilesir_to_mol(ir.degenerate())
        2. Generate coords: mol_3d = add_3d_coords(mol)
        3. Extract coords: coords = extract_coords_from_mol(mol_3d)
        4. Bind to atoms: monomer.unwrap().atoms[i]['xyz'] = coords[i]
    """
    # Try BigSMILES format first (with stochastic objects)
    monomers = extract_monomers_from_ir(ir)

    if len(monomers) == 1:
        return monomers[0]
    elif len(monomers) > 1:
        raise ValueError(
            f"BigSmilesIR contains {len(monomers)} repeat units. "
            "Use bigsmilesir_to_polymerspec() for multiple repeat units."
        )

    # If no stochastic objects, try plain SMILES with atom class ports
    monomer = create_monomer_from_atom_class_ports(ir)
    if monomer is not None:
        return monomer

    raise ValueError(
        "BigSmilesIR contains no repeat units or atom class ports. "
        "Use {[<]...[>]} format or add [*:n] port markers."
    )


def bigsmilesir_to_polymerspec(ir: BigSmilesIR) -> PolymerSpec:
    """
    Convert BigSmilesIR to complete polymer specification.

    Single responsibility: IR → PolymerSpec conversion only.
    Parsing should be done separately.

    Extracts monomers and analyzes polymer topology and composition.

    Args:
        ir: BigSmilesIR from parser

    Returns:
        PolymerSpec with segments, topology, and all monomers

    Examples:
        >>> # Homopolymer
        >>> parser = SmilesParser()
        >>> ir = parser.parse_bigsmiles("{[<]CC[>]}")
        >>> spec = bigsmilesir_to_polymerspec(ir)
        >>> spec.topology
        'homopolymer'
        >>> len(spec.segments)
        1

        >>> # Random copolymer
        >>> ir = parser.parse_bigsmiles("{[<]CC[>],[<]OCC[>]}")
        >>> spec = bigsmilesir_to_polymerspec(ir)
        >>> spec.topology
        'random_copolymer'
        >>> spec.segments[0].composition_type
        'random'

        >>> # Block copolymer
        >>> ir = parser.parse_bigsmiles("{[<]CC[>]}{[<]OCC[>]}")
        >>> spec = bigsmilesir_to_polymerspec(ir)
        >>> spec.topology
        'block_copolymer'
        >>> len(spec.segments)
        2
    """
    return extract_polymerspec_from_ir(ir)


def extract_monomers_from_ir(ir: BigSmilesIR) -> list[Monomer[Atomistic]]:
    """
    Extract monomers from BigSmilesIR (topology only).

    Supports two formats:
    1. BigSMILES with stochastic objects: {[<]CC[>]}
       - Extracts from repeat_segments
    2. Plain SMILES with atom class ports: CCCCO[*:1]
       - Uses atom class notation as port markers

    Args:
        ir: BigSmilesIR from parser

    Returns:
        List of Monomer[Atomistic] with ports set
    """
    monomers = []

    if ir.chain is None:
        return monomers

    # Try BigSMILES format first (stochastic objects)
    for segment in ir.chain.repeat_segments:
        for obj in segment.stochastic_objects:
            for unit in obj.repeat_units:
                # Skip if unit is a BigSmilesChainIR (nested chains not supported yet)
                if not isinstance(unit, SmilesIR):
                    continue

                monomer = create_monomer_from_unit(
                    unit, obj.left_descriptor, obj.right_descriptor
                )
                monomers.append(monomer)

    # If no stochastic objects found, try atom class port format
    if not monomers:
        monomer = create_monomer_from_atom_class_ports(ir)
        if monomer is not None:
            monomers.append(monomer)

    return monomers


def extract_polymerspec_from_ir(ir: BigSmilesIR) -> PolymerSpec:
    """
    Extract complete polymer specification from BigSmilesIR.

    Analyzes the IR structure to determine:
    - Number of segments (blocks)
    - Composition within each segment (random, alternating, etc.)
    - Overall topology

    Args:
        ir: BigSmilesIR from parser

    Returns:
        PolymerSpec with complete polymer information
    """
    if ir.chain is None:
        return PolymerSpec(segments=[], topology="unknown")

    segments = []

    # Process each repeat_segment
    for repeat_segment in ir.chain.repeat_segments:
        for stochastic_obj in repeat_segment.stochastic_objects:
            segment = create_polymer_segment_from_stochastic_object(stochastic_obj)
            segments.append(segment)

    # Determine overall topology
    topology = determine_polymer_topology(segments)

    return PolymerSpec(segments=segments, topology=topology)


def create_polymer_segment_from_stochastic_object(
    obj: StochasticObjectIR,
) -> PolymerSegment:
    """
    Create PolymerSegment from a stochastic object.

    Args:
        obj: StochasticObjectIR containing repeat units and distribution info

    Returns:
        PolymerSegment with monomers and composition type
    """
    monomers = []

    # Extract monomers from repeat units
    for unit in obj.repeat_units:
        if not isinstance(unit, SmilesIR):
            continue

        monomer = create_monomer_from_unit(
            unit, obj.left_descriptor, obj.right_descriptor
        )
        monomers.append(monomer)

    # Determine composition type
    composition_type = None
    distribution_params = None

    if obj.distribution is not None:
        composition_type = obj.distribution.name
        distribution_params = {"params": obj.distribution.params}
    elif len(monomers) > 1:
        # Multiple repeat units without explicit distribution → assume random
        composition_type = "random"

    # Process end groups if present
    end_groups = []
    if obj.end_groups:
        for eg in obj.end_groups:
            if isinstance(eg, SmilesIR):
                # End groups use same descriptors as main units
                eg_monomer = create_monomer_from_unit(
                    eg, obj.left_descriptor, obj.right_descriptor
                )
                end_groups.append(eg_monomer)

    return PolymerSegment(
        monomers=monomers,
        composition_type=composition_type,
        distribution_params=distribution_params,
        end_groups=end_groups,
    )


def determine_polymer_topology(segments: list[PolymerSegment]) -> str:
    """
    Determine overall polymer topology from segments.

    Rules:
    - 1 segment + 1 monomer → "homopolymer"
    - 1 segment + multiple monomers → "random_copolymer" or based on composition_type
    - Multiple segments → "block_copolymer"

    Args:
        segments: List of polymer segments

    Returns:
        Topology string
    """
    if not segments:
        return "unknown"

    if len(segments) == 1:
        segment = segments[0]
        if len(segment.monomers) == 1:
            return "homopolymer"
        elif segment.composition_type == "alternating":
            return "alternating_copolymer"
        else:
            return "random_copolymer"
    else:
        # Multiple segments
        return "block_copolymer"


def create_monomer_from_unit(
    unit: SmilesIR, left_desc: BondDescriptorIR, right_desc: BondDescriptorIR
) -> Monomer[Atomistic]:
    """
    Create Monomer from repeat unit (topology only).

    By BigSMILES convention:
    - left_descriptor connects to first atom (index 0)
    - right_descriptor connects to last atom (index -1)

    Args:
        unit: SmilesIR of repeat unit (pure chemical structure, no descriptors)
        left_desc: Left terminal descriptor from stochastic_object
        right_desc: Right terminal descriptor from stochastic_object

    Returns:
        Monomer[Atomistic] with ports set
    """
    # 1. Create Atomistic (topology only, no positions)
    atomistic = Atomistic()

    # Add atoms without positions
    for atom_ir in unit.atoms:
        from molpy.core.element import Element

        # Get atomic number and element from symbol
        atomic_num = None
        element_symbol = None
        if atom_ir.symbol:
            try:
                element_symbol = atom_ir.symbol.upper()
                atomic_num = Element(element_symbol).number
            except (ValueError, AttributeError):
                element_symbol = atom_ir.symbol

        atomistic.def_atom(
            symbol=atom_ir.symbol,
            element=element_symbol,
            atomic_num=atomic_num,
            charge=atom_ir.charge,
            # NO xyz parameter!
        )

    # Add bonds
    atoms = atomistic.atoms
    for bond_ir in unit.bonds:
        i = unit.atoms.index(bond_ir.start)
        j = unit.atoms.index(bond_ir.end)
        atomistic.def_bond(atoms[i], atoms[j], kind=bond_ir.kind)

    # 2. Create Monomer and set ports
    monomer = Monomer(atomistic)

    # By convention: left descriptor → first atom, right descriptor → last atom
    if len(atoms) > 0:
        left_port_name = descriptor_to_port_name(left_desc)
        monomer.set_port(left_port_name, atoms[0])

        right_port_name = descriptor_to_port_name(right_desc)
        monomer.set_port(right_port_name, atoms[-1])

    return monomer


def find_adjacent_atom_index(ir: SmilesIR, descriptor_idx: int) -> int:
    """
    Find the index of the atom adjacent to a bond descriptor.

    Searches bonds to find which real atom connects to this descriptor.

    Args:
        ir: SmilesIR containing the descriptor
        descriptor_idx: Index of descriptor in ir.atoms

    Returns:
        Index of adjacent AtomIR in ir.atoms

    Raises:
        ValueError: If no adjacent atom found
    """
    from molpy.parser.smiles import AtomIR

    descriptor = ir.atoms[descriptor_idx]

    for bond in ir.bonds:
        if bond.start == descriptor and isinstance(bond.end, AtomIR):
            return ir.atoms.index(bond.end)
        if bond.end == descriptor and isinstance(bond.start, AtomIR):
            return ir.atoms.index(bond.start)

    raise ValueError(f"No adjacent atom found for descriptor at index {descriptor_idx}")


def descriptor_to_port_name(desc: BondDescriptorIR) -> str:
    """
    Convert bond descriptor to standardized port name.

    Naming rules:
    - [<]   → "in"
    - [>]   → "out"
    - [$]   → "branch"
    - [<1]  → "in_1"
    - [>2]  → "out_2"
    - [$3]  → "branch_3"

    Args:
        desc: Bond descriptor IR

    Returns:
        Standardized port name

    Examples:
        >>> descriptor_to_port_name(BondDescriptorIR(symbol="<"))
        "in"
        >>> descriptor_to_port_name(BondDescriptorIR(symbol="<", index=1))
        "in_1"
    """
    symbol_map = {"<": "in", ">": "out", "$": "branch"}

    base = symbol_map.get(desc.symbol or "", "port")

    if desc.index is not None:
        return f"{base}_{desc.index}"
    return base


def create_monomer_from_atom_class_ports(ir: BigSmilesIR) -> Monomer[Atomistic] | None:
    """
    Create Monomer from plain SMILES with atom class notation as ports.

    Atom class notation [*:n] is interpreted as port markers:
    - [*:1] → port "port_1" points to the atom connected to [*:1]
    - [*:2] → port "port_2" points to the atom connected to [*:2]
    - etc.

    The [*:n] atoms themselves are REMOVED from the final structure,
    and ports point to the real atoms they were connected to.

    Args:
        ir: BigSmilesIR from plain SMILES (no stochastic objects)

    Returns:
        Monomer[Atomistic] with ports from atom classes, or None if no ports found

    Examples:
        >>> parser = SmilesParser()
        >>> ir = parser.parse_bigsmiles("CCCCO[*:1]")
        >>> monomer = create_monomer_from_atom_class_ports(ir)
        >>> monomer.port_names()
        ['port_1']
        >>> # Port points to O atom (connected to [*:1])

        >>> ir = parser.parse_bigsmiles("CC(C[*:2])O[*:3]")
        >>> monomer = create_monomer_from_atom_class_ports(ir)
        >>> set(monomer.port_names())
        {'port_2', 'port_3'}
    """
    # Find atoms with class_ attribute (atom class ports)
    port_markers: dict[int, AtomIR] = {}  # class_ -> AtomIR (the [*:n] atom)
    port_connections: dict[int, AtomIR] = {}  # class_ -> connected real atom

    for atom_ir in ir.atoms:
        if isinstance(atom_ir, AtomIR) and atom_ir.class_ is not None:
            port_markers[atom_ir.class_] = atom_ir

    if not port_markers:
        return None  # No ports found

    # Find which atoms are connected to each port marker
    for class_num, marker_atom in port_markers.items():
        for bond_ir in ir.bonds:
            if not (
                isinstance(bond_ir.start, AtomIR) and isinstance(bond_ir.end, AtomIR)
            ):
                continue

            if bond_ir.start == marker_atom:
                port_connections[class_num] = bond_ir.end
                break
            elif bond_ir.end == marker_atom:
                port_connections[class_num] = bond_ir.start
                break

    # Filter out port marker atoms - only keep real atoms
    real_atoms = [
        a for a in ir.atoms if isinstance(a, AtomIR) and a not in port_markers.values()
    ]

    # Build mapping from original AtomIR to actual Atom object
    # CRITICAL: atomistic.atoms returns items from a set (unordered!)
    # So we must store Atom objects as they're created
    atomir_to_atom: dict[int, Atom] = {}  # id(AtomIR) -> Atom object

    # Map port connections to AtomIR
    port_atomirs: dict[int, AtomIR] = {}  # class_ -> connected AtomIR
    for class_num, connected_atom in port_connections.items():
        port_atomirs[class_num] = connected_atom

    # Filter bonds - remove bonds connected to port markers
    real_bonds = [
        b
        for b in ir.bonds
        if isinstance(b.start, AtomIR)
        and isinstance(b.end, AtomIR)
        and b.start not in port_markers.values()
        and b.end not in port_markers.values()
    ]

    # Create Atomistic (topology only, no positions)
    atomistic = Atomistic()

    # Add real atoms and store references immediately
    for atom_ir in real_atoms:
        from molpy.core.element import Element

        # Get atomic number and element from symbol
        atomic_num = None
        element_symbol = None
        if atom_ir.symbol:
            try:
                element_symbol = atom_ir.symbol.upper()
                atomic_num = Element(element_symbol).number
            except (ValueError, AttributeError):
                element_symbol = atom_ir.symbol

        atom = atomistic.def_atom(
            symbol=atom_ir.symbol,
            element=element_symbol,
            atomic_num=atomic_num,
            charge=atom_ir.charge,
            # NO xyz parameter!
        )
        atomir_to_atom[id(atom_ir)] = atom

    # Add bonds using stored atom references
    for bond_ir in real_bonds:
        atom_i = atomir_to_atom[id(bond_ir.start)]
        atom_j = atomir_to_atom[id(bond_ir.end)]
        atomistic.def_bond(atom_i, atom_j, kind=bond_ir.kind)

    # Create Monomer and set ports using stored atom references
    monomer = Monomer(atomistic)

    for class_num, connected_atomir in port_atomirs.items():
        port_name = f"port_{class_num}"
        port_atom = atomir_to_atom[id(connected_atomir)]
        monomer.set_port(port_name, port_atom)

    return monomer
