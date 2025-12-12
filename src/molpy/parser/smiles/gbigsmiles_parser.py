"""gBigSMILES parser implementation built on top of the unified grammar."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from molpy.parser.base import GrammarConfig, GrammarParserBase

from .bigsmiles_ir import BigSmilesMoleculeIR, BondingDescriptorIR, StochasticObjectIR
from .bigsmiles_parser import BigSmilesTransformer
from .gbigsmiles_ir import (
    DistributionIR,
    GBBondingDescriptorIR,
    GBigSmilesComponentIR,
    GBigSmilesMoleculeIR,
    GBigSmilesSystemIR,
    GBStochasticObjectIR,
)


def _iter_descriptors(structure: BigSmilesMoleculeIR) -> Iterable[BondingDescriptorIR]:
    """Iterate over all bonding descriptors in a BigSmilesMoleculeIR structure."""
    yield from structure.backbone.descriptors
    for sobj in structure.stochastic_objects:
        # Use unified terminals field (per BigSMILES v1.1 refactor)
        yield from sobj.terminals.descriptors
        for repeat_unit in sobj.repeat_units:
            yield from repeat_unit.graph.descriptors
        for end_group in sobj.end_groups:
            yield from end_group.graph.descriptors


def _build_descriptor_weights(
    structure: BigSmilesMoleculeIR,
) -> list[GBBondingDescriptorIR]:
    weights: list[GBBondingDescriptorIR] = []
    for descriptor in _iter_descriptors(structure):
        generation = descriptor.extras.pop("generation_weights", None)
        if generation is None:
            continue
        gb = GBBondingDescriptorIR(structural=descriptor)
        if len(generation) == 1:
            gb.global_weight = float(generation[0])
        else:
            gb.pair_weights = [float(value) for value in generation]
        weights.append(gb)
    return weights


def _build_stochastic_metadata(
    structure: BigSmilesMoleculeIR,
) -> list[GBStochasticObjectIR]:
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


class GBigSmilesTransformer:
    """Wrapper transformer that reuses BigSmilesTransformer for structural parsing."""

    def __init__(self):
        self.structural = BigSmilesTransformer(allow_generative=True)
        self._chain_distribution: dict[str, Any] | None = None

    def _extract_chain_distribution(self, tree) -> None:
        """Extract chain-level distribution from parse tree."""
        # Recursively search for stochastic_generation
        if not hasattr(tree, "data"):
            return

        # Check if this node is stochastic_generation
        if tree.data == "stochastic_generation":
            # Extract distribution from stochastic_generation
            # Find stochastic_distribution child
            for child in tree.children:
                if hasattr(child, "data") and child.data == "stochastic_distribution":
                    # Find the actual distribution node (schulz_zimm, etc.) inside
                    for dist_child in child.children:
                        if hasattr(dist_child, "data"):
                            # Transform the distribution node (e.g., schulz_zimm)
                            dist_dict = self.structural.transform(dist_child)
                            if isinstance(dist_dict, dict) and dist_dict.get(
                                "__distribution__"
                            ):
                                self._chain_distribution = dist_dict
                                return

        # Recursively search in children
        if hasattr(tree, "children"):
            for child in tree.children:
                if hasattr(child, "data") or hasattr(child, "children"):
                    self._extract_chain_distribution(child)

    def transform(self, tree) -> GBigSmilesMoleculeIR | GBigSmilesSystemIR:
        # First extract chain-level distribution from tree
        self._extract_chain_distribution(tree)

        # Then transform with structural transformer
        structure: BigSmilesMoleculeIR = self.structural.transform(tree)

        # Apply chain-level distribution to first stochastic_object if present
        if self._chain_distribution is not None and structure.stochastic_objects:
            # Apply to first stochastic_object
            structure.stochastic_objects[0].extras[
                "distribution"
            ] = self._chain_distribution

        descriptor_weights = _build_descriptor_weights(structure)
        stochastic_metadata = _build_stochastic_metadata(structure)
        molecule = GBigSmilesMoleculeIR(
            structure=structure,
            descriptor_weights=descriptor_weights,
            stochastic_metadata=stochastic_metadata,
        )

        system_size = self.structural.system_size_value
        dot_size = self.structural.dot_size_value
        dot_present = self.structural.dot_present_flag

        if system_size is not None or dot_present:
            target = system_size if system_size is not None else dot_size
            component = GBigSmilesComponentIR(molecule=molecule, target_mass=target)
            total_mass = target if target is not None else 0.0
            return GBigSmilesSystemIR(molecules=[component], total_mass=total_mass)

        return molecule


class GBigSmilesParserImpl(GrammarParserBase):
    """Parser that produces gBigSMILES IR from the unified grammar."""

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

    def parse(self, src: str) -> GBigSmilesSystemIR:
        if not src:
            empty = BigSmilesMoleculeIR()
            molecule = GBigSmilesMoleculeIR(structure=empty)
            component = GBigSmilesComponentIR(molecule=molecule, target_mass=None)
            return GBigSmilesSystemIR(molecules=[component], total_mass=0.0)
        tree = self.parse_tree(src)
        transformer = GBigSmilesTransformer()
        ir = transformer.transform(tree)
        if transformer.structural.ring_openings:
            raise ValueError(
                f"Unclosed rings: {list(transformer.structural.ring_openings.keys())}"
            )
        # Always return GBigSmilesSystemIR
        if isinstance(ir, GBigSmilesSystemIR):
            return ir
        else:
            # Wrap GBigSmilesMoleculeIR in GBigSmilesSystemIR
            component = GBigSmilesComponentIR(molecule=ir, target_mass=None)
            return GBigSmilesSystemIR(molecules=[component], total_mass=0.0)
