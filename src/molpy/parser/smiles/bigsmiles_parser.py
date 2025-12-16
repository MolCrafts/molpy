"""BigSMILES parser implementation backed by the unified grammar."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from lark import Token, Transformer

from molpy.parser.base import GrammarConfig, GrammarParserBase

from .bigsmiles_ir import (
    BigSmilesMoleculeIR,
    BigSmilesSubgraphIR,
    BondingDescriptorIR,
    EndGroupIR,
    RepeatUnitIR,
    StochasticObjectIR,
    TerminalDescriptorIR,
)
from .smiles_ir import BondOrder, SmilesAtomIR, SmilesBondIR, SmilesGraphIR


@dataclass
class SmilesSegment:
    """Intermediate result for a SMILES fragment inside the unified grammar."""

    graph: SmilesGraphIR = field(default_factory=SmilesGraphIR)
    descriptors: list[BondingDescriptorIR] = field(default_factory=list)
    stochastic_objects: list[StochasticObjectIR] = field(default_factory=list)


def _token_text(value: Token | str | None) -> str:
    """Extract text from Token or string."""
    if isinstance(value, Token):
        return str(value.value)
    if value is None:
        return ""
    return str(value)


def _coerce_int(value: Token | int | str | None) -> int:
    """Convert Token/str to int, return 0 if empty."""
    text = _token_text(value)
    return int(text) if text else 0


def _coerce_float(value: Token | int | float | str | None) -> float:
    """Convert Token/str to float, return 0.0 if empty."""
    text = _token_text(value)
    if text == "":
        return 0.0
    return float(text)


def _bond_from_symbol(
    symbol: str | None,
) -> tuple[BondOrder, Literal["/", "\\"] | None]:
    """
    Convert bond symbol to (order, stereo) tuple.

    Args:
        symbol: Bond symbol (-, =, #, :, /, \\, or None)

    Returns:
        (bond_order, stereo_direction) tuple
    """
    if symbol in ("/", "\\"):
        return 1, symbol  # stereo single bond
    if symbol == "=":
        return 2, None
    if symbol == "#":
        return 3, None
    if symbol == ":":
        return "ar", None
    return 1, None


class BigSmilesTransformer(Transformer):
    """
    Lark transformer that converts parse trees to BigSMILES IR.

    This transformer maps the unified grammar (supporting SMILES, BigSMILES,
    and gBigSMILES) into structural BigSMILES intermediate representation.

    The transformer handles:
    - Atom parsing (bracket atoms, organic subsets, aromatic atoms)
    - Bond formation (single, double, triple, aromatic, stereo)
    - Ring closures with validation
    - Branch handling
    - Bonding descriptors ([<], [>], [$], etc.)
    - Stochastic objects (polymer repeat units)
    - Molecular weight distributions (gBigSMILES only)
    - System size annotations (gBigSMILES only)

    Architecture:
        The transformer uses a stateful approach to track:
        - ring_openings: Dict of open ring closures
        - _system_size: Global system size annotation
        - _dot_size: Dot notation size
        - _dot_present: Whether dot notation is used

    Args:
        allow_generative: If True, enables gBigSMILES features
                         (distributions, weights, system sizes)

    Raises:
        ValueError: If gBigSMILES features used without allow_generative=True
        ValueError: If rings are unclosed after parsing

    Examples:
        >>> transformer = BigSmilesTransformer()
        >>> tree = parser.parse("{[<]CC[>]}")
        >>> molecule = transformer.transform(tree)
        >>> print(len(molecule.stochastic_objects))
        1
    """

    def __init__(self, *, allow_generative: bool = False):
        super().__init__()
        self.allow_generative = allow_generative
        self.ring_openings: dict[str, tuple[SmilesAtomIR, str | None]] = {}
        self._system_size: float | None = None
        self._dot_size: float | None = None
        self._dot_present = False
        self._chain_distribution: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------
    def INT(self, value: Token) -> int:
        return int(value)

    def NUMBER(self, value: Token) -> float:
        return float(value)

    # ------------------------------------------------------------------
    # Atom / element helpers
    # ------------------------------------------------------------------
    def atom_symbol(self, children: list) -> str:
        return _token_text(children[0] if children else None)

    def element_symbol(self, children: list) -> str:
        return _token_text(children[0] if children else None)

    def aliphatic_organic(self, children: list) -> SmilesAtomIR:
        text = _token_text(children[0] if children else None)
        return SmilesAtomIR(element=text, aromatic=False)

    def aromatic_organic(self, children: list) -> SmilesAtomIR:
        text = _token_text(children[0] if children else None)
        return SmilesAtomIR(element=text.upper(), aromatic=True)

    def aromatic_symbol(self, children: list) -> str:
        return _token_text(children[0] if children else None)

    def big_smiles_fragment_declaration(self, children: list) -> SmilesAtomIR:
        name = "".join(
            _token_text(child) for child in children if not isinstance(child, Token)
        )
        return SmilesAtomIR(element=None, extras={"fragment": name})

    def atom(self, children: list) -> SmilesAtomIR:
        return children[0]

    def isotope(self, children: list) -> tuple[str, int]:
        return ("isotope", _coerce_int(children[0] if children else None))

    def chiral(self, children: list) -> tuple[str, str]:
        return ("chiral", "".join(_token_text(tok) for tok in children))

    def h_count(self, children: list) -> tuple[str, int]:
        if len(children) > 1:
            return ("h_count", _coerce_int(children[1]))
        return ("h_count", 1)

    def atom_charge(self, children: list) -> tuple[str, int]:
        if not children:
            return ("charge", 0)
        sign = -1 if _token_text(children[0]) == "-" else 1
        if len(children) == 1:
            return ("charge", sign)
        second = children[1]
        if isinstance(second, Token) and second.type in {"PLUSPLUS", "MINUSMINUS"}:
            return ("charge", 2 * sign)
        return ("charge", sign * _coerce_int(second))

    def atom_class(self, children: list) -> tuple[str, int]:
        return ("class_", _coerce_int(children[-1]))

    def bracket_atom(self, children: list) -> SmilesAtomIR:
        symbol: SmilesAtomIR | str | None = None
        props: dict[str, Any] = {}
        charge = None
        hydrogens = None
        for child in children:
            if isinstance(child, SmilesAtomIR):
                symbol = child
            elif isinstance(child, tuple):
                key, value = child
                if key == "charge":
                    charge = value
                elif key == "h_count":
                    hydrogens = value
                else:
                    props[key] = value
            elif isinstance(child, Token):
                continue
            else:
                symbol = child
        if isinstance(symbol, SmilesAtomIR):
            atom = symbol
        else:
            symbol_text = _token_text(symbol)
            aromatic = symbol_text.islower()
            element = symbol_text.upper() if aromatic else symbol_text
            atom = SmilesAtomIR(element=element, aromatic=aromatic)
        atom.charge = charge
        atom.hydrogens = hydrogens
        atom.extras.update(props)
        return atom

    # ------------------------------------------------------------------
    # Descriptors
    # ------------------------------------------------------------------
    def bond_descriptor_symbol(self, children: list) -> str:
        return _token_text(children[0] if children else None)

    def bond_descriptor_symbol_idx(
        self, children: list
    ) -> tuple[str | None, int | None]:
        if not children:
            return None, None
        symbol = self.bond_descriptor_symbol(children[:1])
        label = None
        if len(children) > 1:
            label = _coerce_int(children[1])
        return symbol, label

    def bond_descriptor_generation(self, children: list) -> list[float]:
        if not self.allow_generative:
            raise ValueError("Bond descriptor weight annotations require gBigSMILES")
        floats = [
            float(child) if isinstance(child, (int, float)) else _coerce_float(child)
            for child in children
            if not isinstance(child, Token)
        ]
        return floats

    def inner_bond_descriptor(self, children: list) -> BondingDescriptorIR:
        symbol, label = children[0] if children else (None, None)
        generation = children[1] if len(children) > 1 else None
        desc = BondingDescriptorIR(symbol=symbol, label=label)
        if generation:
            desc.extras["generation_weights"] = generation
        return desc

    def inner_non_covalent_descriptor(self, children: list) -> BondingDescriptorIR:
        symbol = self.bond_descriptor_symbol(children[:1])
        label = None
        context = None
        remaining = children[1:]
        if remaining and isinstance(remaining[0], int):
            label = remaining[0]
            remaining = remaining[1:]
        if remaining:
            context = remaining[0]
        desc = BondingDescriptorIR(symbol=symbol, label=label)
        desc.non_covalent_context = context
        return desc

    def _descriptor_from_child(
        self, child: BondingDescriptorIR | list | None
    ) -> BondingDescriptorIR:
        if isinstance(child, BondingDescriptorIR):
            return child
        if isinstance(child, list):
            for item in child:
                if isinstance(item, BondingDescriptorIR):
                    return item
        return BondingDescriptorIR()

    def simple_bond_descriptor(self, children: list) -> BondingDescriptorIR:
        filtered = [child for child in children if not isinstance(child, Token)]
        return self._descriptor_from_child(filtered[0] if filtered else None)

    def non_covalent_bond_descriptor(self, children: list) -> BondingDescriptorIR:
        filtered = [child for child in children if not isinstance(child, Token)]
        return self._descriptor_from_child(filtered[0] if filtered else None)

    def ladder_bond_descriptor(self, children: list) -> BondingDescriptorIR:
        filtered = [child for child in children if not isinstance(child, Token)]
        if not filtered:
            return BondingDescriptorIR()
        primary = self._descriptor_from_child(filtered[0])
        if len(filtered) > 1 and isinstance(filtered[1], BondingDescriptorIR):
            partner = filtered[1]
            primary.extras["ladder_partner"] = {
                "symbol": partner.symbol,
                "label": partner.label,
            }
        if len(filtered) > 2 and isinstance(filtered[2], int):
            primary.extras["ladder_count"] = filtered[2]
        return primary

    def bond_descriptor(self, children: list) -> BondingDescriptorIR:
        return self._descriptor_from_child(children[0] if children else None)

    def terminal_bond_descriptor(self, children: list) -> TerminalDescriptorIR:
        filtered = [child for child in children if not isinstance(child, Token)]
        descriptors: list[BondingDescriptorIR] = []
        if filtered:
            if isinstance(filtered[0], tuple):
                symbol, label = filtered[0]
                descriptors.append(BondingDescriptorIR(symbol=symbol, label=label))
            elif isinstance(filtered[0], BondingDescriptorIR):
                descriptors.append(filtered[0])
        if len(filtered) > 1:
            generation = filtered[1]
            if generation:
                if not self.allow_generative:
                    raise ValueError("Terminal descriptor weights require gBigSMILES")
                for descriptor in descriptors:
                    descriptor.extras["generation_weights"] = generation
        return TerminalDescriptorIR(descriptors=descriptors)

    # ------------------------------------------------------------------
    # Branch / ring helpers
    # ------------------------------------------------------------------
    def bond_symbol(self, children: list) -> str:
        return _token_text(children[0] if children else None)

    def ring_bond(self, children: list) -> tuple[str, str | None, str]:
        bond_symbol = None
        digits: list[str] = []
        for child in children:
            if isinstance(child, Token) and child.type == "DIGIT":
                digits.append(child.value)
            elif isinstance(child, str):
                bond_symbol = child
        ring_id = "".join(digits) or _token_text(children[-1])
        return ("ring", ring_id, bond_symbol)

    def branch(self, children: list) -> tuple[str, SmilesSegment, str | None]:
        bond = None
        segment: SmilesSegment | None = None
        for child in children:
            if isinstance(child, Token) and child.type in {"LPAR", "RPAR"}:
                continue
            if isinstance(child, str):
                bond = child
            elif isinstance(child, SmilesSegment):
                segment = child
        if segment is None:
            segment = SmilesSegment()
        return ("branch", segment, bond)

    def atom_assembly(self, children: list) -> tuple[str, str | None, tuple]:
        if len(children) == 1:
            return ("asm", None, children[0])
        return ("asm", children[0], children[1])

    def branched_atom(
        self, children: list
    ) -> tuple[Any, list[tuple[str, str | None]], list[tuple]]:
        node = children[0]
        rings: list[tuple[str, str | None]] = []
        branches: list[tuple] = []
        for item in children[1:]:
            if isinstance(item, tuple):
                tag = item[0]
                if tag == "ring":
                    rings.append((item[1], item[2]))
                elif tag == "branch":
                    branches.append(item)
        return node, rings, branches

    # ------------------------------------------------------------------
    # SMILES assembly
    # ------------------------------------------------------------------
    def smiles(self, children: list) -> SmilesSegment:
        segment = SmilesSegment()
        if not children:
            return segment
        seq: list[Any] = []
        seq.append(children[0])
        for item in children[1:]:
            if isinstance(item, tuple) and item and item[0] == "asm":
                _, bond, branched = item
                if bond is not None:
                    seq.append(bond)
                seq.append(branched)
            else:
                seq.append(item)
        active_atom: SmilesAtomIR | None = None
        pending_bond: str | None = None
        # Track descriptors that appeared before any atom (prefix case like [$]O)
        pending_descriptors: list[BondingDescriptorIR] = []

        for entry in seq:
            if isinstance(entry, str):
                pending_bond = entry
                continue
            if not isinstance(entry, tuple) or len(entry) != 3:
                continue
            node, rings, branches = entry
            if isinstance(node, BondingDescriptorIR):
                if active_atom is not None:
                    # Postfix case: descriptor after atom (like C[$])
                    # Use pending_bond if present, otherwise default
                    order, _ = _bond_from_symbol(pending_bond)
                    node.bond_order = order
                    node.anchor_atom = active_atom
                else:
                    # Prefix case: descriptor before atom (like [$]O)
                    # Store and link to next atom
                    pending_descriptors.append(node)
                segment.descriptors.append(node)
                pending_bond = None
                continue
            if isinstance(node, StochasticObjectIR):
                segment.stochastic_objects.append(node)
                pending_bond = None
                continue
            if not isinstance(node, SmilesAtomIR):
                continue

            # Link any pending prefix descriptors to this atom
            # Also assign bond order from the bond between descriptor and this atom
            for descriptor in pending_descriptors:
                descriptor.anchor_atom = node
                # For prefix descriptors like [$]=C, the pending_bond is the bond
                # FROM descriptor TO this atom - assign it as bond_order
                if pending_bond is not None:
                    order, _ = _bond_from_symbol(pending_bond)
                    descriptor.bond_order = order
            pending_descriptors = []

            segment.graph.atoms.append(node)
            if active_atom is not None and active_atom is not node:
                order, stereo = _bond_from_symbol(pending_bond)
                segment.graph.bonds.append(
                    SmilesBondIR(
                        atom_i=active_atom, atom_j=node, order=order, stereo=stereo
                    )
                )
            active_atom = node
            pending_bond = None
            for ring_id, bond_symbol in rings or []:
                if ring_id in self.ring_openings:
                    start_atom, start_symbol = self.ring_openings.pop(ring_id)
                    if start_atom is not node:
                        order, stereo = _bond_from_symbol(bond_symbol or start_symbol)
                        segment.graph.bonds.append(
                            SmilesBondIR(
                                atom_i=start_atom,
                                atom_j=node,
                                order=order,
                                stereo=stereo,
                            )
                        )
                else:
                    self.ring_openings[ring_id] = (node, bond_symbol)
            for branch in branches or []:
                if not isinstance(branch, tuple) or len(branch) != 3:
                    continue
                _, branch_segment, branch_bond = branch
                if not isinstance(branch_segment, SmilesSegment):
                    continue
                if not branch_segment.graph.atoms:
                    continue
                order, stereo = _bond_from_symbol(branch_bond)
                head = branch_segment.graph.atoms[0]
                if active_atom is not None and active_atom is not head:
                    segment.graph.bonds.append(
                        SmilesBondIR(
                            atom_i=active_atom, atom_j=head, order=order, stereo=stereo
                        )
                    )
                segment.graph.atoms.extend(branch_segment.graph.atoms)
                segment.graph.bonds.extend(branch_segment.graph.bonds)
                segment.descriptors.extend(branch_segment.descriptors)
                segment.stochastic_objects.extend(branch_segment.stochastic_objects)
        return segment

    # ------------------------------------------------------------------
    # Repeat units and stochastic objects
    # ------------------------------------------------------------------
    def _segment_to_subgraph(self, segment: SmilesSegment) -> BigSmilesSubgraphIR:
        return BigSmilesSubgraphIR(
            atoms=list(segment.graph.atoms),
            bonds=list(segment.graph.bonds),
            descriptors=list(segment.descriptors),
        )

    def _as_subgraph(
        self, item: SmilesSegment | BigSmilesMoleculeIR
    ) -> BigSmilesSubgraphIR:
        if isinstance(item, SmilesSegment):
            return self._segment_to_subgraph(item)
        if isinstance(item, BigSmilesMoleculeIR):
            return item.backbone
        raise TypeError(f"Unsupported repeat unit type: {type(item)!r}")

    def _mark_terminal_role(
        self, terminal: TerminalDescriptorIR
    ) -> TerminalDescriptorIR:
        """Mark all descriptors in terminal with unified 'terminal' role."""
        for descriptor in terminal.descriptors:
            descriptor.role = "terminal"
        return terminal

    def _convert_repeat_items(self, items: Sequence[Any]) -> list[RepeatUnitIR]:
        result: list[RepeatUnitIR] = []
        for item in items:
            if isinstance(item, (SmilesSegment, BigSmilesMoleculeIR)):
                result.append(RepeatUnitIR(graph=self._as_subgraph(item)))
        return result

    def _convert_end_groups(self, items: Sequence[Any]) -> list[EndGroupIR]:
        result: list[EndGroupIR] = []
        for item in items:
            if isinstance(item, (SmilesSegment, BigSmilesMoleculeIR)):
                subgraph = self._as_subgraph(item)
                for descriptor in subgraph.descriptors:
                    descriptor.role = "end_group"
                result.append(EndGroupIR(graph=subgraph))
        return result

    def _repeat_units(self, children: list) -> list[Any]:
        flattened: list[Any] = []
        for child in children:
            if isinstance(child, list):
                flattened.extend(child)
            else:
                flattened.append(child)
        return flattened

    def monomer_list(self, children: list) -> list[Any]:
        return [child for child in children if not isinstance(child, Token)]

    def _repeat_unit_item(self, children: list) -> Any:
        for child in children:
            if isinstance(child, (SmilesSegment, BigSmilesMoleculeIR)):
                return child
        return SmilesSegment()

    def _end_group(self, children: list) -> tuple[str, list[Any]]:
        items = [child for child in children if not isinstance(child, Token)]
        return ("end_group", items)

    def _distribution_from_call(self, name: str, args: Sequence[Any]) -> dict[str, Any]:
        params = {f"p{i}": float(arg) for i, arg in enumerate(args)}
        return {"__distribution__": True, "name": name, "params": params}

    def flory_schulz(self, children: list) -> dict[str, Any]:
        numbers = [float(child) for child in children if not isinstance(child, Token)]
        return self._distribution_from_call("flory_schulz", numbers)

    def schulz_zimm(self, children: list) -> dict[str, Any]:
        numbers = [float(child) for child in children if not isinstance(child, Token)]
        return self._distribution_from_call("schulz_zimm", numbers)

    def gauss(self, children: list) -> dict[str, Any]:
        numbers = [float(child) for child in children if not isinstance(child, Token)]
        return self._distribution_from_call("gauss", numbers)

    def uniform(self, children: list) -> dict[str, Any]:
        numbers = [float(child) for child in children if not isinstance(child, Token)]
        return self._distribution_from_call("uniform", numbers)

    def log_normal(self, children: list) -> dict[str, Any]:
        numbers = [float(child) for child in children if not isinstance(child, Token)]
        return self._distribution_from_call("log_normal", numbers)

    def poisson(self, children: list) -> dict[str, Any]:
        numbers = [float(child) for child in children if not isinstance(child, Token)]
        return self._distribution_from_call("poisson", numbers)

    def stochastic_generation(self, children: list) -> dict[str, Any]:
        if not self.allow_generative:
            raise ValueError("Stochastic distributions require gBigSMILES")
        return (
            children[0]
            if children
            else {"__distribution__": True, "name": "unknown", "params": {}}
        )

    def stochastic_object(self, children: list) -> StochasticObjectIR:
        """Parse stochastic object with unified terminal handling.

        Per BigSMILES v1.1: terminal bonding descriptors at stochastic object
        boundaries connect internal structure to external SMILES. Both terminals
        serve the same semantic role, so we merge them into a single `terminals` field.

        Per BigSMILES v1.1 syntax: { [terminal] repeat_units ; end_groups [terminal] }
        Both terminal descriptors are REQUIRED, even if empty [] (explicitly indicating
        no external connection). Empty [] must be explicitly written.

        Terminal descriptors are linked to atoms:
        - First terminal descriptors attach to first atom of first repeat unit
        - Last terminal descriptors attach to last atom of last repeat unit
        """
        filtered = [child for child in children if not isinstance(child, Token)]
        if len(filtered) < 2:
            raise ValueError(
                "Stochastic object requires two terminal descriptors (even if empty []). "
                "Per BigSMILES v1.1: { [terminal] repeat_units ; end_groups [terminal] }"
            )

        # Collect all terminal descriptors (both first and last are REQUIRED per BigSMILES v1.1)
        first_terminal = filtered[0]
        if not isinstance(first_terminal, TerminalDescriptorIR):
            raise ValueError(
                "Stochastic object must start with a terminal descriptor (even if empty []). "
                "Per BigSMILES v1.1: { [terminal] repeat_units ; end_groups [terminal] }"
            )

        # Last terminal descriptor is REQUIRED per BigSMILES v1.1 syntax
        if not isinstance(filtered[-1], TerminalDescriptorIR):
            raise ValueError(
                "Stochastic object must end with a terminal descriptor (even if empty []). "
                "Per BigSMILES v1.1: { [terminal] repeat_units ; end_groups [terminal] }"
                " Example: {[]C(COCCO[>])(COCCO[>])COCCO[>][]} instead of "
                "{C(COCCO[>])(COCCO[>])COCCO[>]}"
            )
        last_terminal = filtered[-1]
        body = filtered[1:-1]

        # Parse body content
        distribution = None
        repeat_items: list[Any] = []
        end_group_items: list[Any] = []
        for part in body:
            if isinstance(part, dict) and part.get("__distribution__"):
                distribution = part
            elif isinstance(part, tuple) and part and part[0] == "end_group":
                end_group_items.extend(part[1])
            elif isinstance(part, list):
                repeat_items.extend(part)
            else:
                repeat_items.append(part)

        # Collect all atoms from repeat units to link terminal descriptors
        all_atoms: list[SmilesAtomIR] = []
        for item in repeat_items:
            if isinstance(item, SmilesSegment):
                all_atoms.extend(item.graph.atoms)

        # Link first terminal descriptors to first atom
        if all_atoms and first_terminal.descriptors:
            for descriptor in first_terminal.descriptors:
                if descriptor.anchor_atom is None:
                    descriptor.anchor_atom = all_atoms[0]

        # Link last terminal descriptors to last atom
        if all_atoms and last_terminal.descriptors:
            for descriptor in last_terminal.descriptors:
                if descriptor.anchor_atom is None:
                    descriptor.anchor_atom = all_atoms[-1]

        # Collect terminal symbols for pruning
        terminal_symbols = {
            descriptor.symbol
            for descriptor in first_terminal.descriptors + last_terminal.descriptors
            if descriptor.symbol is not None
        }

        # Extract unanchored terminal descriptors from repeat units
        extracted_descriptors = self._prune_terminal_descriptors(
            repeat_items, terminal_symbols
        )

        # Merge all terminal descriptors into unified terminals field
        all_terminal_descriptors = (
            first_terminal.descriptors
            + last_terminal.descriptors
            + extracted_descriptors
        )
        unified_terminals = TerminalDescriptorIR(
            descriptors=all_terminal_descriptors,
            extras={**first_terminal.extras, **last_terminal.extras},
        )
        self._mark_terminal_role(unified_terminals)

        sobj = StochasticObjectIR(
            terminals=unified_terminals,
            repeat_units=self._convert_repeat_items(repeat_items),
            end_groups=self._convert_end_groups(end_group_items),
        )
        if distribution is not None:
            sobj.extras["distribution"] = distribution
        return sobj

    def _prune_terminal_descriptors(
        self, items: Sequence[Any], terminal_symbols: set[str | None]
    ) -> list[BondingDescriptorIR]:
        """Extract unanchored terminal descriptors from repeat units.

        Per BigSMILES v1.1: bonding descriptors without anchor atoms (not attached
        to any atom in the repeat unit) are terminal bonding descriptors that
        connect the stochastic object to external SMILES.

        Returns:
            List of extracted terminal descriptors (unanchored < and > symbols)
        """
        extracted_descriptors: list[BondingDescriptorIR] = []
        # Always consider < and > as potential terminal symbols
        all_terminal_symbols = terminal_symbols | {"<", ">"}

        for item in items:
            if isinstance(item, SmilesSegment):
                kept: list[BondingDescriptorIR] = []
                for descriptor in item.descriptors:
                    # Extract unanchored descriptors with terminal symbols
                    if (
                        descriptor.anchor_atom is None
                        and descriptor.symbol in all_terminal_symbols
                    ):
                        extracted_descriptors.append(descriptor)
                    else:
                        kept.append(descriptor)
                item.descriptors = kept
        return extracted_descriptors

    # ------------------------------------------------------------------
    # Chain assembly
    # ------------------------------------------------------------------
    def big_smiles_repeat(self, children: list) -> list[Any]:
        result: list[Any] = []
        for child in children:
            if isinstance(child, list):
                result.extend(child)
            else:
                result.append(child)
        return result

    def big_smiles_molecule(self, children: list) -> BigSmilesMoleculeIR:
        segments: list[SmilesSegment] = []
        stochastic_objects: list[StochasticObjectIR] = []
        for child in children:
            if isinstance(child, SmilesSegment):
                segments.append(child)
                stochastic_objects.extend(child.stochastic_objects)
            elif isinstance(child, StochasticObjectIR):
                stochastic_objects.append(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, StochasticObjectIR):
                        stochastic_objects.append(item)
                    elif isinstance(item, SmilesSegment):
                        segments.append(item)
                        stochastic_objects.extend(item.stochastic_objects)
        backbone = BigSmilesSubgraphIR()
        for segment in segments:
            backbone.atoms.extend(segment.graph.atoms)
            backbone.bonds.extend(segment.graph.bonds)
            backbone.descriptors.extend(segment.descriptors)
        return BigSmilesMoleculeIR(
            backbone=backbone, stochastic_objects=stochastic_objects
        )

    # ------------------------------------------------------------------
    # Generative decorations handled at gBigSMILES layer
    # ------------------------------------------------------------------
    def dot_generation(self, children: list) -> None:
        if not self.allow_generative:
            raise ValueError("Dot generation annotations require gBigSMILES")
        size = None
        for child in children:
            if isinstance(child, dict) and child.get("__system_size__"):
                size = child["value"]
            elif isinstance(child, (int, float)):
                size = float(child)
        self._dot_present = True
        if size is not None:
            self._dot_size = size
        return None

    def dot_system_size(self, children: list) -> dict[str, Any]:
        value = None
        for child in children:
            if isinstance(child, (int, float)):
                value = float(child)
            elif isinstance(child, Token) and child.type == "NUMBER":
                value = float(child)
        return {"__system_size__": True, "value": value}

    def system_size(self, children: list) -> None:
        if not self.allow_generative:
            raise ValueError("System size annotations require gBigSMILES")
        value = None
        for child in children:
            if isinstance(child, (int, float)):
                value = float(child)
            elif isinstance(child, Token) and child.type == "NUMBER":
                value = float(child)
        self._system_size = value
        return None

    @property
    def system_size_value(self) -> float | None:
        return self._system_size

    @property
    def dot_size_value(self) -> float | None:
        return self._dot_size

    @property
    def dot_present_flag(self) -> bool:
        return self._dot_present


class BigSmilesParserImpl(GrammarParserBase):
    """
    Parser that produces BigSmilesMoleculeIR from BigSMILES strings.

    Uses Lark parser with Earley algorithm and BigSmilesTransformer
    to convert BigSMILES notation into structured IR.

    Examples:
        >>> parser = BigSmilesParserImpl()
        >>> molecule = parser.parse("{[<]CC[>]}")
        >>> print(len(molecule.stochastic_objects))
        1
    """

    def __init__(self):
        config = GrammarConfig(
            grammar_path=Path(__file__).parent / "grammars" / "gbigsmiles_new.lark",
            start="big_smiles_molecule",
            parser="earley",
            propagate_positions=True,
            maybe_placeholders=False,
            auto_reload=True,
        )
        super().__init__(config)

    def parse(self, src: str) -> BigSmilesMoleculeIR:
        if not src:
            return BigSmilesMoleculeIR()
        tree = self.parse_tree(src)
        transformer = BigSmilesTransformer(allow_generative=False)
        molecule = transformer.transform(tree)
        if transformer.ring_openings:
            raise ValueError(
                f"Unclosed rings: {list(transformer.ring_openings.keys())}"
            )
        return molecule
