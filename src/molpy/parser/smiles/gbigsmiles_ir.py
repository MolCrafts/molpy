"""gBigSMILES generative IR aligned with unified grammar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bigsmiles_ir import BigSmilesMoleculeIR, BondingDescriptorIR, StochasticObjectIR


@dataclass(eq=True)
class DistributionIR:
    """Generative distribution applied to stochastic objects."""

    name: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(eq=True)
class GBStochasticObjectIR:
    """Wraps a structural stochastic object plus optional distribution."""

    structural: StochasticObjectIR
    distribution: DistributionIR | None = None


@dataclass(eq=True)
class GBBondingDescriptorIR:
    """Weights associated with a bonding descriptor."""

    structural: BondingDescriptorIR
    global_weight: float | None = None
    pair_weights: list[float] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class GBigSmilesMoleculeIR:
    """gBigSMILES molecule = structure + generative metadata."""

    structure: BigSmilesMoleculeIR
    descriptor_weights: list[GBBondingDescriptorIR] = field(default_factory=list)
    stochastic_metadata: list[GBStochasticObjectIR] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class GBigSmilesComponentIR:
    """Single component entry in a gBigSMILES system."""

    molecule: GBigSmilesMoleculeIR
    target_mass: float | None = None
    mass_is_fraction: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class GBigSmilesSystemIR:
    """gBigSMILES system describing an ensemble of molecules."""

    molecules: list[GBigSmilesComponentIR] = field(default_factory=list)
    total_mass: float | None = None


def build_stochastic_metadata(
    structure: BigSmilesMoleculeIR,
) -> list[GBStochasticObjectIR]:
    """Build stochastic metadata from BigSMILES structural IR.

    Extracts distribution annotations from stochastic objects and wraps
    them in ``GBStochasticObjectIR`` containers.

    Args:
        structure: Parsed BigSMILES molecule IR.

    Returns:
        List of generative stochastic-object wrappers.
    """
    metadata: list[GBStochasticObjectIR] = []
    for sobj in structure.stochastic_objects:
        distribution_data = sobj.extras.pop("distribution", None)
        distribution = None
        if distribution_data is not None:
            distribution = DistributionIR(
                name=str(distribution_data.get("name", "unknown")),
                params={
                    k: float(v) for k, v in distribution_data.get("params", {}).items()
                },
            )
        metadata.append(
            GBStochasticObjectIR(structural=sobj, distribution=distribution)
        )
    return metadata
