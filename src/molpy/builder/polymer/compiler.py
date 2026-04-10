"""G-BigSMILES compiler: IR to builder execution pipeline.

Compiles a ``GBigSmilesSystemIR`` into a sequence of builder calls,
producing a list of Atomistic chain structures.

The compiler orchestrates:
1. Monomer preparation (IR → Atomistic with 3D)
2. Distribution creation (IR → DPDistribution | MassDistribution)
3. System planning (distribution + sequence generator → chain plan)
4. Chain building (plan + library + reacter → Atomistic chains)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.parser.smiles.converter_polymer import (
    _create_monomer_from_repeat_unit,
)
from molpy.parser.smiles.gbigsmiles_ir import (
    GBigSmilesComponentIR,
    GBigSmilesMoleculeIR,
    GBigSmilesSystemIR,
    GBStochasticObjectIR,
    build_stochastic_metadata,
)
from molpy.reacter.base import Reacter

from .distributions import (
    DPDistribution,
    MassDistribution,
    UniformPolydisperse,
    create_polydisperse_from_ir,
)
from .presets import ReactionPresets
from .sequences import WeightedSequenceGenerator
from .system import Chain, PolydisperseChainGenerator, SystemPlan, SystemPlanner


@dataclass(frozen=True)
class CompilerConfig:
    """Configuration for the G-BigSMILES compiler.

    Attributes:
        reaction_preset: Name of reaction preset for chain building.
        add_hydrogens: Add explicit hydrogens during monomer 3D generation.
        optimize_geometry: Optimize monomer geometry via RDKit.
        random_seed: Random seed for reproducibility.
    """

    reaction_preset: str = "dehydration"
    add_hydrogens: bool = True
    optimize_geometry: bool = True
    random_seed: int | None = None


class GBigSmilesCompiler:
    """Compile G-BigSMILES IR into built polymer chains.

    The compiler walks the IR tree and delegates to the class-level APIs:
    - ``_prepare_monomers`` → extract and optionally 3D-embed monomers
    - ``_create_distribution`` → build distribution from IR metadata
    - ``_plan_component`` → plan chains for a single component
    - ``_build_chains`` → assemble chains from a plan

    Example::

        from molpy.parser.smiles import parse_gbigsmiles

        ir = parse_gbigsmiles("{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|")
        compiler = GBigSmilesCompiler(CompilerConfig(random_seed=42))
        chains = compiler.compile_and_build(ir)
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        self._config = config or CompilerConfig()

    @property
    def config(self) -> CompilerConfig:
        return self._config

    def compile_and_build(self, system_ir: GBigSmilesSystemIR) -> list[Atomistic]:
        """Full pipeline: IR → monomer preparation → system planning → chain building.

        Args:
            system_ir: Parsed G-BigSMILES system IR.

        Returns:
            List of Atomistic structures (one per chain).
        """
        rng = np.random.default_rng(self._config.random_seed)
        reacter = ReactionPresets.get(self._config.reaction_preset)
        all_chains: list[Atomistic] = []

        for component in system_ir.molecules:
            chains = self._compile_component(component, reacter, rng)
            all_chains.extend(chains)

        return all_chains

    def _compile_component(
        self,
        component: GBigSmilesComponentIR,
        reacter: Reacter,
        rng: np.random.Generator,
    ) -> list[Atomistic]:
        """Compile a single G-BigSMILES component into chains."""
        molecule = component.molecule
        stochastic_metadata = molecule.stochastic_metadata
        if not stochastic_metadata:
            stochastic_metadata = build_stochastic_metadata(molecule.structure)

        library, label_order = self._prepare_monomers(molecule)

        if not library:
            return []

        distribution = self._create_distribution(stochastic_metadata)
        target_mass = component.target_mass

        if distribution is None and target_mass is not None and target_mass > 0:
            # No distribution: treat the annotation value as fixed DP
            dp = max(1, int(target_mass))
            plan = self._plan_fixed_dp(dp, label_order, library)
        elif target_mass is not None and target_mass > 0 and distribution is not None:
            # Has distribution + mass target → full system planning
            plan = self._plan_component(
                stochastic_metadata, label_order, library, target_mass, rng
            )
        else:
            # No annotation → build a single chain with default DP
            plan = self._plan_single_chain(
                stochastic_metadata, label_order, library, rng
            )

        return self._build_chains(plan, library, reacter)

    def _prepare_monomers(
        self, molecule: GBigSmilesMoleculeIR
    ) -> tuple[dict[str, Atomistic], list[str]]:
        """Extract and prepare monomers from molecule IR.

        Returns:
            Tuple of (label→Atomistic library, ordered label list).
        """
        library: dict[str, Atomistic] = {}
        label_order: list[str] = []

        for stoch_meta in molecule.stochastic_metadata:
            stoch_obj = stoch_meta.structural
            for i, repeat_unit in enumerate(stoch_obj.repeat_units):
                monomer = _create_monomer_from_repeat_unit(repeat_unit, stoch_obj)
                if monomer is None:
                    continue

                label = f"M{len(library)}"
                monomer = self._try_generate_3d(monomer)
                library[label] = monomer
                label_order.append(label)

        if not molecule.stochastic_metadata:
            for stoch_obj in molecule.structure.stochastic_objects:
                for repeat_unit in stoch_obj.repeat_units:
                    monomer = _create_monomer_from_repeat_unit(repeat_unit, stoch_obj)
                    if monomer is None:
                        continue
                    label = f"M{len(library)}"
                    monomer = self._try_generate_3d(monomer)
                    library[label] = monomer
                    label_order.append(label)

        return library, label_order

    def _try_generate_3d(self, monomer: Atomistic) -> Atomistic:
        """Attempt 3D generation via RDKit if available."""
        try:
            from molpy.adapter.rdkit import RDKitAdapter
            from molpy.tool.rdkit import Generate3D
        except ImportError:
            return monomer

        adapter = RDKitAdapter(monomer)
        gen3d = Generate3D(
            add_hydrogens=self._config.add_hydrogens,
            optimize=self._config.optimize_geometry,
        )
        adapter = gen3d(adapter)
        return adapter.get_internal()

    def _create_distribution(
        self,
        stochastic_metadata: list[GBStochasticObjectIR],
    ) -> DPDistribution | MassDistribution | None:
        """Create distribution from the first stochastic object that has one."""
        for meta in stochastic_metadata:
            if meta.distribution is not None:
                return create_polydisperse_from_ir(
                    meta.distribution,
                    random_seed=self._config.random_seed,
                )
        return None

    def _estimate_monomer_mass(self, library: dict[str, Atomistic]) -> dict[str, float]:
        """Estimate monomer masses from atom counts (rough: sum of atomic masses)."""
        _ATOMIC_MASS = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "S": 32.06,
            "P": 30.974,
            "Cl": 35.45,
            "Br": 79.904,
            "Si": 28.085,
        }
        masses: dict[str, float] = {}
        for label, monomer in library.items():
            mass = 0.0
            for atom in monomer.atoms:
                symbol = atom.get("symbol", "C")
                mass += _ATOMIC_MASS.get(symbol, 12.0)
            masses[label] = mass
        return masses

    def _plan_fixed_dp(
        self,
        dp: int,
        label_order: list[str],
        library: dict[str, Atomistic],
    ) -> SystemPlan:
        """Create a plan for a single chain with fixed DP."""
        monomer_mass = self._estimate_monomer_mass(library)
        # Round-robin through label_order for the sequence
        monomers = [label_order[i % len(label_order)] for i in range(dp)]
        mass = sum(monomer_mass.get(m, 0.0) for m in monomers)
        chain = Chain(dp=dp, monomers=monomers, mass=mass)
        return SystemPlan(chains=[chain], total_mass=mass, target_mass=mass)

    def _plan_single_chain(
        self,
        stochastic_metadata: list[GBStochasticObjectIR],
        label_order: list[str],
        library: dict[str, Atomistic],
        rng: np.random.Generator,
    ) -> SystemPlan:
        """Plan a single chain when no target mass is specified."""
        distribution = self._create_distribution(stochastic_metadata)
        if distribution is None:
            distribution = UniformPolydisperse(min_dp=10, max_dp=10)

        monomer_mass = self._estimate_monomer_mass(library)
        weights = {label: 1.0 for label in label_order}
        seq_gen = WeightedSequenceGenerator(weights)

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass=monomer_mass,
            distribution=distribution,
        )
        chain = chain_gen.build_chain(rng)
        return SystemPlan(
            chains=[chain],
            total_mass=chain.mass,
            target_mass=chain.mass,
        )

    def _plan_component(
        self,
        stochastic_metadata: list[GBStochasticObjectIR],
        label_order: list[str],
        library: dict[str, Atomistic],
        target_mass: float,
        rng: np.random.Generator,
    ) -> SystemPlan:
        """Plan a multi-chain system for a component with a mass target."""
        distribution = self._create_distribution(stochastic_metadata)
        if distribution is None:
            distribution = UniformPolydisperse(min_dp=10, max_dp=10)

        monomer_mass = self._estimate_monomer_mass(library)
        weights = {label: 1.0 for label in label_order}
        seq_gen = WeightedSequenceGenerator(weights)

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass=monomer_mass,
            distribution=distribution,
        )
        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=target_mass,
        )
        return planner.plan_system(rng)

    def _build_chains(
        self,
        plan: SystemPlan,
        library: dict[str, Atomistic],
        reacter: Reacter,
    ) -> list[Atomistic]:
        """Build Atomistic chains from a system plan.

        For each chain in the plan, constructs a CGSmiles string from the
        monomer sequence and delegates to PolymerBuilder.
        """
        from .core import PolymerBuilder

        placer = self._make_placer(library)

        built_chains: list[Atomistic] = []
        for chain in plan.chains:
            if not chain.monomers:
                continue

            cgsmiles = self._chain_to_cgsmiles(chain)
            builder = PolymerBuilder(
                library=library,
                reacter=reacter,
                placer=placer,
            )
            result = builder.build(cgsmiles)
            built_chains.append(result.polymer)

        return built_chains

    @staticmethod
    def _make_placer(library: dict[str, Atomistic]):
        """Create a Placer only if monomers have 3D coordinates."""
        from .placer import CovalentSeparator, LinearOrienter, Placer

        for monomer in library.values():
            for atom in monomer.atoms:
                if "x" not in atom:
                    return None
        return Placer(CovalentSeparator(), LinearOrienter())

    @staticmethod
    def _chain_to_cgsmiles(chain: Chain) -> str:
        """Convert a Chain to CGSmiles notation.

        For a chain with monomers ["M0", "M0", "M1", "M0"], produces:
        ``"{[#M0][#M0][#M1][#M0]}"``
        """
        nodes = "".join(f"[#{m}]" for m in chain.monomers)
        return "{" + nodes + "}"
