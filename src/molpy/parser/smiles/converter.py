"""Convert BigSmilesIR to Atomistic and PolymerSpec.

This module provides conversion functions from the new BigSmilesIR structure
to MolPy Atomistic structures and PolymerSpec objects.
"""

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

from molpy.core.atomistic import Atom, Atomistic
from molpy.core.element import Element

from .bigsmiles_ir import (
    BigSmilesMoleculeIR,
    BigSmilesSubgraphIR,
    BondingDescriptorIR,
    EndGroupIR,
    RepeatUnitIR,
    StochasticObjectIR,
)
from .smiles_ir import SmilesAtomIR, SmilesGraphIR


def _convert_bond_order_to_kind(order: int | float) -> str:
    """Convert numeric bond order to kind symbol string.

    Args:
        order: Bond order (1, 2, 3, or 1.5 for aromatic)

    Returns:
        Bond kind symbol ("-", "=", "#", or ":")
    """
    kind_map = {1: "-", 2: "=", 3: "#", 1.5: ":"}
    return kind_map.get(float(order), "-")


@dataclass
class PolymerSegment:
    """Polymer segment specification."""

    monomers: list[Atomistic]
    composition_type: str | None = None
    distribution_params: dict | None = None
    end_groups: list[Atomistic] = field(default_factory=list)
    # Store original IR repeat units for direct access
    repeat_units_ir: list[BigSmilesSubgraphIR] = field(default_factory=list)
    end_groups_ir: list[BigSmilesSubgraphIR] = field(default_factory=list)


@dataclass
class PolymerSpec:
    """Complete polymer specification."""

    segments: list[PolymerSegment]
    topology: str
    # Store start and end groups from GBigSmilesIR
    start_group_ir: SmilesGraphIR | None = None
    end_group_ir: SmilesGraphIR | None = None

    def __post_init__(self):
        """Ensure segments is initialized."""
        if not hasattr(self, "segments") or self.segments is None:
            object.__setattr__(self, "segments", [])

    def all_monomers(self) -> list[Atomistic]:
        """Get all structures from all segments."""
        return [struct for segment in self.segments for struct in segment.monomers]


def bigsmilesir_to_monomer(ir: BigSmilesMoleculeIR) -> Atomistic:
    """
    Convert BigSmilesMoleculeIR to Atomistic structure (topology only).

    Single responsibility: IR → Atomistic conversion only.
    Parsing should be done separately.

    Supports BigSMILES with stochastic object: {[<]CC[>]} (ONE repeat unit only)

    Args:
        ir: BigSmilesMoleculeIR from parser

    Returns:
        Atomistic structure with ports marked on atoms, NO positions

    Raises:
        ValueError: If IR contains multiple repeat units (use bigsmilesir_to_polymerspec instead)

    Examples:
        >>> from molpy.parser.smiles import parse_bigsmiles
        >>> ir = parse_bigsmiles("{[<]CC[>]}")
        >>> struct = bigsmilesir_to_monomer(ir)
        >>> # Ports are marked on atoms: atom["port"] = "<" or ">"
    """
    monomers = []

    # Extract from stochastic objects
    for stoch_obj in ir.stochastic_objects:
        for repeat_unit in stoch_obj.repeat_units:
            monomer = create_monomer_from_repeat_unit(repeat_unit, stoch_obj)
            if monomer is not None:
                monomers.append(monomer)

    if len(monomers) == 1:
        return monomers[0]
    elif len(monomers) > 1:
        raise ValueError(
            f"BigSmilesMoleculeIR contains {len(monomers)} repeat units. "
            "Use bigsmilesir_to_polymerspec() for multiple repeat units."
        )

    raise ValueError(
        "BigSmilesMoleculeIR contains no repeat units. " "Use {[<]...[>]} format."
    )


def bigsmilesir_to_polymerspec(ir: BigSmilesMoleculeIR) -> PolymerSpec:
    """
    Convert BigSmilesIR to complete polymer specification.

    Single responsibility: IR -> PolymerSpec conversion only.
    Parsing should be done separately.

    Extracts monomers and analyzes polymer topology and composition.

    Args:
        ir: BigSmilesIR from parser

    Returns:
        PolymerSpec with segments, topology, and all monomers

    Examples:
        >>> from molpy.parser.smiles import parse_bigsmiles
        >>> ir = parse_bigsmiles("{[<]CC[>]}")
        >>> spec = bigsmilesir_to_polymerspec(ir)
        >>> spec.topology
        'homopolymer'
    """
    return extract_polymerspec_from_ir(ir)


def extract_polymerspec_from_ir(ir: BigSmilesMoleculeIR) -> PolymerSpec:
    """
    Extract complete polymer specification from BigSmilesMoleculeIR.

    Analyzes the IR structure to determine:
    - Number of segments (blocks)
    - Composition within each segment (random, alternating, etc.)
    - Overall topology

    Args:
        ir: BigSmilesMoleculeIR from parser

    Returns:
        PolymerSpec with complete polymer information
    """
    segments = []

    # Process each stochastic object as a segment
    for stochastic_obj in ir.stochastic_objects:
        segment = create_polymer_segment_from_stochastic_object(stochastic_obj)
        segments.append(segment)

    # Determine overall topology
    topology = determine_polymer_topology(segments)

    # Extract start group from backbone if present
    start_group_ir = None
    if ir.backbone.atoms or ir.backbone.bonds:
        # Convert backbone to SmilesGraphIR
        start_group_ir = SmilesGraphIR(
            atoms=list(ir.backbone.atoms),
            bonds=list(ir.backbone.bonds),
        )

    return PolymerSpec(
        segments=segments,
        topology=topology,
        start_group_ir=start_group_ir,
        end_group_ir=None,
    )


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
    for repeat_unit in obj.repeat_units:
        monomer = create_monomer_from_repeat_unit(repeat_unit, obj)
        if monomer is not None:
            monomers.append(monomer)

    # Determine composition type
    composition_type = None
    distribution_params = None

    # Note: Distribution info is now in gBigSMILES layer, not here
    if len(monomers) > 1:
        # Multiple repeat units -> assume random
        composition_type = "random"

    # Process end groups if present
    end_groups = []
    end_groups_ir = []
    if obj.end_groups:
        for end_group in obj.end_groups:
            eg_monomer = create_monomer_from_end_group(end_group, obj)
            if eg_monomer is not None:
                end_groups.append(eg_monomer)
                end_groups_ir.append(end_group.graph)

    # Store original IR repeat units and end groups
    repeat_units_ir = [ru.graph for ru in obj.repeat_units]

    return PolymerSegment(
        monomers=monomers,
        composition_type=composition_type,
        distribution_params=distribution_params,
        end_groups=end_groups,
        repeat_units_ir=repeat_units_ir,
        end_groups_ir=end_groups_ir,
    )


def determine_polymer_topology(segments: list[PolymerSegment]) -> str:
    """
    Determine overall polymer topology from segments.

    Rules:
    - 1 segment + 1 monomer -> "homopolymer"
    - 1 segment + multiple monomers -> "random_copolymer" or based on composition_type
    - Multiple segments -> "block_copolymer"

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


def create_monomer_from_repeat_unit(
    repeat_unit: RepeatUnitIR, stoch_obj: StochasticObjectIR
) -> Atomistic | None:
    """
    Create Atomistic structure from RepeatUnitIR.

    Extracts bonding descriptors from the subgraph to determine port locations.

    Args:
        repeat_unit: RepeatUnitIR containing the subgraph
        stoch_obj: StochasticObjectIR containing terminal descriptors

    Returns:
        Atomistic structure with ports marked on atoms based on descriptors
    """
    graph = repeat_unit.graph

    # Create Atomistic structure (topology only, no positions)
    struct = Atomistic()

    # Add atoms
    for atom_ir in graph.atoms:
        atom_data = asdict(atom_ir)
        # SmilesAtomIR has 'element' but not 'symbol', so copy element to symbol
        if atom_data.get("element") and not atom_data.get("symbol"):
            atom_data["symbol"] = atom_data["element"]
        struct.def_atom(**atom_data)

    # Add bonds
    atoms = list(struct.atoms)
    bonds_added = set()

    # Build atom mapping: SmilesAtomIR -> index in graph.atoms
    atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(graph.atoms)}

    for bond_ir in graph.bonds:
        # Use id() for reliable atom matching (not == which compares by value)
        i = atom_ir_to_idx.get(id(bond_ir.atom_i))
        j = atom_ir_to_idx.get(id(bond_ir.atom_j))

        # Skip if atoms not found or same atom (by identity, not value)
        if i is None or j is None or i == j:
            continue

        if i < len(atoms) and j < len(atoms):
            bond_key = tuple(sorted([i, j]))
            if bond_key not in bonds_added:
                bond_order = bond_ir.order
                bond_kind = _convert_bond_order_to_kind(bond_order)
                struct.def_bond(atoms[i], atoms[j], order=bond_order, kind=bond_kind)
                bonds_added.add(bond_key)

    # Set ports based on descriptors using anchor_atom
    # All descriptors (from terminals and from repeat unit graph) should have anchor_atom set
    all_descriptors = stoch_obj.terminals.descriptors + list(graph.descriptors)

    # Build mapping from SmilesAtomIR to Atomistic Atom
    # atoms list is in same order as graph.atoms
    atomir_to_atom: dict[int, Atom] = {
        id(atom_ir): atoms[i] for i, atom_ir in enumerate(graph.atoms)
    }

    for descriptor in all_descriptors:
        if descriptor.anchor_atom is None:
            # No anchor atom - skip (shouldn't happen with proper parsing)
            continue

        # Find the corresponding Atomistic atom using anchor_atom
        anchor_atomir_id = id(descriptor.anchor_atom)
        if anchor_atomir_id not in atomir_to_atom:
            # anchor_atom not found in mapping - skip
            continue

        atom = atomir_to_atom[anchor_atomir_id]
        if atom.get("port") is not None:
            # Already has a port - skip (shouldn't happen if descriptors are unique)
            continue

        port_name = descriptor_to_port_name(descriptor)
        atom["port"] = port_name
        atom["port_role"] = "terminal"
        atom["port_bond_kind"] = "-"
        atom["port_compat"] = set()
        atom["port_priority"] = 0

    return struct


def create_monomer_from_end_group(end_group: EndGroupIR) -> Atomistic | None:
    """Create Atomistic structure from EndGroupIR."""
    # Similar to create_monomer_from_repeat_unit but for end groups
    graph = end_group.graph

    struct = Atomistic()

    # Add atoms and bonds (same logic as repeat unit)
    for atom_ir in graph.atoms:
        struct.def_atom(**asdict(atom_ir))

    atoms = list(struct.atoms)
    bonds_added = set()

    # Build atom mapping using id() for reliable matching
    atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(graph.atoms)}

    for bond_ir in graph.bonds:
        # Use id() for reliable atom matching
        i = atom_ir_to_idx.get(id(bond_ir.atom_i))
        j = atom_ir_to_idx.get(id(bond_ir.atom_j))

        # Skip if atoms not found or same atom (by identity)
        if i is None or j is None or i == j:
            continue

        if i < len(atoms) and j < len(atoms):
            bond_key = tuple(sorted([i, j]))
            if bond_key not in bonds_added:
                bond_order = bond_ir.order
                bond_kind = _convert_bond_order_to_kind(bond_order)
                struct.def_bond(atoms[i], atoms[j], order=bond_order, kind=bond_kind)
                bonds_added.add(bond_key)

    # Set ports from descriptors
    # All descriptors are unified in terminals - no graph descriptors to process

    return struct


def create_monomer_from_unit(
    unit: SmilesGraphIR, left_desc: BondingDescriptorIR, right_desc: BondingDescriptorIR
) -> Atomistic:
    """
    Create Atomistic structure from SmilesGraphIR with descriptors.

    By BigSMILES convention:
    - left_descriptor connects to first atom (index 0)
    - right_descriptor connects to last atom (index -1)

    Args:
        unit: SmilesGraphIR of repeat unit (pure chemical structure, no descriptors)
        left_desc: Left terminal descriptor
        right_desc: Right terminal descriptor

    Returns:
        Atomistic structure with ports marked on atoms
    """
    # Create Atomistic structure (topology only, no positions)
    struct = Atomistic()

    # Add atoms
    for atom_ir in unit.atoms:

        struct.def_atom(**asdict(atom_ir))

    # Add bonds
    atoms = list(struct.atoms)
    bonds_added = set()

    # Build atom mapping using id() for reliable matching
    atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(unit.atoms)}

    for bond_ir in unit.bonds:
        # Use id() for reliable atom matching
        i = atom_ir_to_idx.get(id(bond_ir.atom_i))
        j = atom_ir_to_idx.get(id(bond_ir.atom_j))

        # Skip if atoms not found or same atom (by identity)
        if i is None or j is None or i == j:
            continue

        if i < len(atoms) and j < len(atoms):
            bond_key = tuple(sorted([i, j]))
            if bond_key not in bonds_added:
                # Set both order (numeric) and kind (symbol) for bond
                bond_order = bond_ir.order
                kind_map = {1: "-", 2: "=", 3: "#", 1.5: ":"}
                bond_kind = kind_map.get(float(bond_order), "-")
                struct.def_bond(atoms[i], atoms[j], order=bond_order, kind=bond_kind)
                bonds_added.add(bond_key)

    # By convention: left descriptor -> first atom, right descriptor -> last atom
    if len(atoms) > 0:
        if left_desc is not None:
            left_port_name = descriptor_to_port_name(left_desc)
            atoms[0]["port"] = left_port_name

        if right_desc is not None:
            right_port_name = descriptor_to_port_name(right_desc)
            atoms[-1]["port"] = right_port_name

    return struct


def descriptor_to_port_name(desc: BondingDescriptorIR) -> str:
    """
    Convert bond descriptor to port name using original symbol.

    Naming rules:
    - [<]   -> "<"
    - [>]   -> ">"
    - [$]   -> "$"
    - [<1]  -> "<1"
    - [>2]  -> ">2"
    - [$3]  -> "$3"

    Args:
        desc: BondingDescriptorIR

    Returns:
        Port name using original descriptor symbol

    Examples:
        >>> from molpy.parser.smiles.bigsmiles_ir import BondingDescriptorIR
        >>> descriptor_to_port_name(BondingDescriptorIR(symbol="<"))
        "<"
        >>> descriptor_to_port_name(BondingDescriptorIR(symbol="<", label=1))
        "<1"
    """
    symbol = desc.symbol or ""

    if desc.label is not None:
        return f"{symbol}{desc.label}"
    return symbol


def create_monomer_from_atom_class_ports(ir: SmilesGraphIR) -> Atomistic | None:
    """
    Create Atomistic structure from SmilesGraphIR with atom class notation as ports.

    Atom class notation [*:n] is interpreted as port markers:
    - [*:1] -> port "port_1" points to the atom connected to [*:1]
    - [*:2] -> port "port_2" points to the atom connected to [*:2]
    - etc.

    The [*:n] atoms themselves are REMOVED from the final structure,
    and ports point to the real atoms they were connected to.

    Args:
        ir: SmilesGraphIR from plain SMILES

    Returns:
        Atomistic structure with ports marked on atoms, or None if no ports found
    """
    # Find atoms with class_ attribute (atom class ports)
    port_markers: dict[int, SmilesAtomIR] = (
        {}
    )  # class_ -> SmilesAtomIR (the [*:n] atom)
    port_connections: dict[int, SmilesAtomIR] = {}  # class_ -> connected real atom

    for atom_ir in ir.atoms:
        if atom_ir.extras.get("class_") is not None:
            class_num = atom_ir.extras["class_"]
            port_markers[class_num] = atom_ir

    if not port_markers:
        return None  # No ports found

    # Find which atoms are connected to each port marker
    for class_num, marker_atom in port_markers.items():
        for bond_ir in ir.bonds:
            if bond_ir.atom_i == marker_atom:
                port_connections[class_num] = bond_ir.atom_j
                break
            elif bond_ir.atom_j == marker_atom:
                port_connections[class_num] = bond_ir.atom_i
                break

    # Filter out port marker atoms - only keep real atoms
    real_atoms = [a for a in ir.atoms if a not in port_markers.values()]

    # Build mapping from original SmilesAtomIR to actual Atom object
    atomir_to_atom: dict[int, Atom] = {}  # id(SmilesAtomIR) -> Atom object

    # Map port connections to SmilesAtomIR
    port_atomirs: dict[int, SmilesAtomIR] = {}  # class_ -> connected SmilesAtomIR
    for class_num, connected_atom in port_connections.items():
        port_atomirs[class_num] = connected_atom

    # Filter bonds - remove bonds connected to port markers
    real_bonds = [
        b
        for b in ir.bonds
        if b.atom_i not in port_markers.values()
        and b.atom_j not in port_markers.values()
    ]

    # Create Atomistic structure (topology only, no positions)
    struct = Atomistic()

    # Add real atoms and store references immediately
    for atom_ir in real_atoms:

        atom = struct.def_atom(**asdict(atom_ir))
        atomir_to_atom[id(atom_ir)] = atom

    # Add bonds using stored atom references
    for bond_ir in real_bonds:
        atom_i = atomir_to_atom[id(bond_ir.atom_i)]
        atom_j = atomir_to_atom[id(bond_ir.atom_j)]
        bond_order = bond_ir.order
        bond_kind = _convert_bond_order_to_kind(bond_order)
        struct.def_bond(atom_i, atom_j, order=bond_order, kind=bond_kind)

    # Set ports by marking atoms
    for class_num, connected_atomir in port_atomirs.items():
        port_name = f"port_{class_num}"
        port_atom = atomir_to_atom[id(connected_atomir)]
        port_atom["port"] = port_name

    return struct
