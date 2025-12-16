"""BigSMILES structural IR aligned with the unified grammar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .smiles_ir import BondOrder, SmilesAtomIR, SmilesBondIR

DescriptorSymbol = Literal["$", "<", ">"]
DescriptorRole = Literal["internal", "terminal", "end_group"]  # unified terminal role

_id_counter = 0


def _generate_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


@dataclass(eq=True)
class BondingDescriptorIR:
    """Standalone descriptor node for bonding points.

    Per BigSMILES v1.1: bonding descriptors attach to atoms within repeat units.
    The anchor_atom field tracks which atom this descriptor is attached to.
    If anchor_atom is None, this is a terminal bonding descriptor at the
    stochastic object boundary.
    """

    id: int = field(default_factory=_generate_id, compare=False)
    symbol: DescriptorSymbol | None = None
    label: int | None = None
    bond_order: BondOrder = 1
    role: DescriptorRole = "internal"
    anchor_atom: SmilesAtomIR | None = None  # Atom this descriptor attaches to
    non_covalent_context: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    # Position hint: "first" for first atom, "last" for last atom, None for unspecified
    position_hint: str | None = None  # "first" | "last" | None

    def __hash__(self) -> int:
        return self.id


@dataclass(eq=True)
class BigSmilesSubgraphIR:
    """Structural fragment that carries atoms, bonds, and descriptors."""

    atoms: list[SmilesAtomIR] = field(default_factory=list)
    bonds: list[SmilesBondIR] = field(default_factory=list)
    descriptors: list[BondingDescriptorIR] = field(default_factory=list)


@dataclass(eq=True)
class TerminalDescriptorIR:
    """Terminal brackets that hold descriptors for stochastic objects."""

    descriptors: list[BondingDescriptorIR] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class RepeatUnitIR:
    """Repeat unit captured inside a stochastic object."""

    id: int = field(default_factory=_generate_id, compare=False)
    graph: BigSmilesSubgraphIR = field(default_factory=BigSmilesSubgraphIR)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class EndGroupIR:
    """Optional end-group fragments that terminate stochastic objects."""

    id: int = field(default_factory=_generate_id, compare=False)
    graph: BigSmilesSubgraphIR = field(default_factory=BigSmilesSubgraphIR)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class StochasticObjectIR:
    """Container for repeat units, terminals, and end groups."""

    id: int = field(default_factory=_generate_id, compare=False)
    terminals: TerminalDescriptorIR = field(default_factory=TerminalDescriptorIR)
    repeat_units: list[RepeatUnitIR] = field(default_factory=list)
    end_groups: list[EndGroupIR] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return self.id


@dataclass(eq=True)
class BigSmilesMoleculeIR:
    """Top-level structural IR for BigSMILES strings."""

    backbone: BigSmilesSubgraphIR = field(default_factory=BigSmilesSubgraphIR)
    stochastic_objects: list[StochasticObjectIR] = field(default_factory=list)
