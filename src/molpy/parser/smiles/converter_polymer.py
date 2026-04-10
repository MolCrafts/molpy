"""Convert BigSmilesIR to PolymerSpec and monomer Atomistic structures.

This module handles polymer topology analysis and monomer extraction
from BigSMILES intermediate representations.
"""

from dataclasses import dataclass, field

from molpy.core.atomistic import Atomistic

from .bigsmiles_ir import (
    BigSmilesMoleculeIR,
    BigSmilesSubgraphIR,
    BondingDescriptorIR,
    EndGroupIR,
    RepeatUnitIR,
    StochasticObjectIR,
)
from .converter_atomistic import _build_atomistic_from_graph
from .smiles_ir import SmilesGraphIR


@dataclass
class PolymerSegment:
    """Polymer segment specification.

    Attributes:
        monomers: List of Atomistic structures for each repeat unit monomer.
        composition_type: Composition type ("random", "alternating", etc.) or None.
        distribution_params: Optional distribution parameters for stochastic generation.
        end_groups: Atomistic structures for end groups.
        repeat_units_ir: Raw BigSmilesSubgraphIR for each repeat unit.
        end_groups_ir: Raw BigSmilesSubgraphIR for each end group.
    """

    monomers: list[Atomistic]
    composition_type: str | None = None
    distribution_params: dict | None = None
    end_groups: list[Atomistic] = field(default_factory=list)
    repeat_units_ir: list[BigSmilesSubgraphIR] = field(default_factory=list)
    end_groups_ir: list[BigSmilesSubgraphIR] = field(default_factory=list)


@dataclass
class PolymerSpec:
    """Complete polymer specification.

    Attributes:
        segments: List of PolymerSegment (one per stochastic object).
        topology: Topology classification (e.g., "homopolymer", "block_copolymer").
        start_group_ir: Optional SmilesGraphIR for the backbone start group.
        end_group_ir: Optional SmilesGraphIR for the backbone end group.
    """

    segments: list[PolymerSegment]
    topology: str
    start_group_ir: SmilesGraphIR | None = None
    end_group_ir: SmilesGraphIR | None = None

    def __post_init__(self):
        if not hasattr(self, "segments") or self.segments is None:
            object.__setattr__(self, "segments", [])

    def all_monomers(self) -> list[Atomistic]:
        """Get all monomer structures from all segments.

        Returns:
            Flat list of Atomistic structures across every segment.
        """
        return [struct for segment in self.segments for struct in segment.monomers]


def bigsmilesir_to_monomer(ir: BigSmilesMoleculeIR) -> Atomistic:
    """
    Convert BigSmilesMoleculeIR to Atomistic structure (topology only).

    Supports BigSMILES with stochastic object: {[<]CC[>]} (ONE repeat unit only)

    Args:
        ir: BigSmilesMoleculeIR from parser

    Returns:
        Atomistic structure with ports marked on atoms, NO positions

    Raises:
        ValueError: If IR contains multiple repeat units (use bigsmilesir_to_polymerspec instead)
    """
    monomers = []

    for stoch_obj in ir.stochastic_objects:
        for repeat_unit in stoch_obj.repeat_units:
            monomer = _create_monomer_from_repeat_unit(repeat_unit, stoch_obj)
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
        "BigSmilesMoleculeIR contains no repeat units. Use {[<]...[>]} format."
    )


def bigsmilesir_to_polymerspec(ir: BigSmilesMoleculeIR) -> PolymerSpec:
    """
    Convert BigSmilesIR to complete polymer specification.

    Extracts monomers and analyzes polymer topology and composition.

    Args:
        ir: BigSmilesIR from parser

    Returns:
        PolymerSpec with segments, topology, and all monomers
    """
    segments = []

    for stochastic_obj in ir.stochastic_objects:
        segment = _create_polymer_segment_from_stochastic_object(stochastic_obj)
        segments.append(segment)

    topology = _determine_polymer_topology(segments)

    start_group_ir = None
    if ir.backbone.atoms or ir.backbone.bonds:
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


def _create_polymer_segment_from_stochastic_object(
    obj: StochasticObjectIR,
) -> PolymerSegment:
    """Create PolymerSegment from a stochastic object."""
    monomers = []

    for repeat_unit in obj.repeat_units:
        monomer = _create_monomer_from_repeat_unit(repeat_unit, obj)
        if monomer is not None:
            monomers.append(monomer)

    composition_type = None
    distribution_params = None

    if len(monomers) > 1:
        composition_type = "random"

    end_groups = []
    end_groups_ir = []
    if obj.end_groups:
        for end_group in obj.end_groups:
            eg_monomer = _create_monomer_from_end_group(end_group)
            if eg_monomer is not None:
                end_groups.append(eg_monomer)
                end_groups_ir.append(end_group.graph)

    repeat_units_ir = [ru.graph for ru in obj.repeat_units]

    return PolymerSegment(
        monomers=monomers,
        composition_type=composition_type,
        distribution_params=distribution_params,
        end_groups=end_groups,
        repeat_units_ir=repeat_units_ir,
        end_groups_ir=end_groups_ir,
    )


def _determine_polymer_topology(segments: list[PolymerSegment]) -> str:
    """Determine overall polymer topology from segments."""
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
        return "block_copolymer"


def _create_monomer_from_repeat_unit(
    repeat_unit: RepeatUnitIR, stoch_obj: StochasticObjectIR
) -> Atomistic | None:
    """Create Atomistic structure from RepeatUnitIR."""
    graph = repeat_unit.graph

    struct, atom_ir_id_to_atom = _build_atomistic_from_graph(graph.atoms, graph.bonds)

    all_descriptors = stoch_obj.terminals.descriptors + list(graph.descriptors)

    for descriptor in all_descriptors:
        if descriptor.anchor_atom is None:
            continue

        anchor_atomir_id = id(descriptor.anchor_atom)
        if anchor_atomir_id not in atom_ir_id_to_atom:
            continue

        atom = atom_ir_id_to_atom[anchor_atomir_id]
        if atom.get("port") is not None:
            continue

        port_name = _descriptor_to_port_name(descriptor)
        atom["port"] = port_name

    return struct


def _create_monomer_from_end_group(end_group: EndGroupIR) -> Atomistic | None:
    """Create Atomistic structure from EndGroupIR."""
    graph = end_group.graph

    struct, _atom_mapping = _build_atomistic_from_graph(graph.atoms, graph.bonds)

    return struct


def _descriptor_to_port_name(desc: BondingDescriptorIR) -> str:
    """Convert bond descriptor to port name using original symbol."""
    symbol = desc.symbol or ""

    if desc.label is not None:
        return f"{symbol}{desc.label}"
    return symbol
