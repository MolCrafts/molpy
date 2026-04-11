"""Polymer building tools.

Tools that wrap the parser, adapter, builder, and reacter modules
into single-call workflows for common polymer construction tasks.

Tools (auto-registered in ToolRegistry):
- ``PrepareMonomer`` — BigSMILES → 3D Atomistic with ports
- ``BuildPolymer`` — CGSmiles + library → assembled chain
- ``PlanSystem`` — distribution parameters → chain plan (no atoms)
- ``BuildSystem`` — G-BigSMILES → list of built chains

Convenience functions:
- ``polymer()`` — auto-detect notation, build single chain
- ``polymer_system()`` — G-BigSMILES → multi-chain system
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from molpy.core.atomistic import Atomistic

from .base import Tool

# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

Backend = Literal["default", "amber"]


@dataclass(frozen=True)
class PrepareMonomer(Tool):
    """Parse a BigSMILES monomer string and produce an Atomistic structure.

    Pipeline: parse BigSMILES → convert to Atomistic with port markers →
    generate 3D coordinates via RDKit (if available) → compute angles/dihedrals.

    Preferred for:
        - Preparing monomers for BuildPolymer or polymer().
        - One-step SMILES-to-3D when you need port annotations.

    Avoid when:
        - You already have an Atomistic struct (use RDKit adapter directly).
        - You need custom 3D embedding parameters (use Generate3D).

    Attributes:
        add_hydrogens: Add explicit hydrogens during 3D generation.
        optimize: Optimize geometry after 3D embedding.
        gen_topology: Compute angles and dihedrals.

    Related:
        - BuildPolymer
        - polymer
        - Generate3D
    """

    add_hydrogens: bool = True
    optimize: bool = True
    gen_topology: bool = True

    def run(self, smiles: str) -> Atomistic:
        """Prepare a monomer from a BigSMILES string.

        Args:
            smiles: BigSMILES string (e.g. ``"{[<]CCOCC[>]}"``).

        Returns:
            Atomistic structure with ports marked and optional 3D coordinates.
        """
        from molpy.parser import parse_monomer

        monomer = parse_monomer(smiles)

        monomer = self._try_generate_3d(monomer)

        if self.gen_topology:
            monomer = monomer.get_topo(gen_angle=True, gen_dihe=True)

        return monomer

    def _try_generate_3d(self, monomer: Atomistic) -> Atomistic:
        """Attempt 3D generation via RDKit if available."""
        try:
            from molpy.adapter.rdkit import RDKitAdapter
            from molpy.tool.rdkit import Generate3D
        except ImportError:
            return monomer

        adapter = RDKitAdapter(monomer)
        gen3d = Generate3D(
            add_hydrogens=self.add_hydrogens,
            optimize=self.optimize,
        )
        adapter = gen3d(adapter)
        return adapter.get_internal()


@dataclass(frozen=True)
class BuildPolymer(Tool):
    """Build a polymer chain from CGSmiles notation and a monomer library.

    Preferred for:
        - Assembling a single chain from pre-prepared monomers.
        - Iterating over a system plan to build chains one at a time.

    Avoid when:
        - You want end-to-end build from a string (use polymer() or BuildSystem).
        - You need custom reaction logic (use PolymerBuilder directly).

    Attributes:
        reaction_preset: Name of reaction preset (default ``"dehydration"``).
        use_placer: Enable geometric placement of monomers.

    Related:
        - PrepareMonomer
        - PlanSystem
        - polymer
        - PolymerBuilder
    """

    reaction_preset: str = "dehydration"
    use_placer: bool = True

    def run(self, cgsmiles: str, library: dict[str, Atomistic]) -> dict[str, Any]:
        """Build a polymer chain.

        Args:
            cgsmiles: CGSmiles notation (e.g. ``"{[#EO]|10}"``).
            library: Mapping from label to prepared Atomistic monomer.

        Returns:
            Dict with ``"polymer"`` (Atomistic), ``"total_steps"`` (int),
            and ``"connection_history"`` (list).
        """
        from molpy.builder.polymer.core import PolymerBuilder
        from molpy.builder.polymer.placer import (
            CovalentSeparator,
            LinearOrienter,
            Placer,
        )
        from molpy.builder.polymer.presets import ReactionPresets

        reacter = ReactionPresets.get(self.reaction_preset)

        placer = None
        if self.use_placer and self._has_coords(library):
            placer = Placer(CovalentSeparator(), LinearOrienter())

        builder = PolymerBuilder(
            library=library,
            reacter=reacter,
            placer=placer,
        )
        result = builder.build(cgsmiles)

        return {
            "polymer": result.polymer,
            "total_steps": result.total_steps,
            "connection_history": result.connection_history,
        }

    @staticmethod
    def _has_coords(library: dict) -> bool:
        """Check whether every monomer in library has 3D coordinates."""
        for monomer in library.values():
            for atom in monomer.atoms:
                if "x" not in atom:
                    return False
        return True


@dataclass(frozen=True)
class PlanSystem(Tool):
    """Plan a polydisperse polymer system from distribution parameters.

    Returns chain specifications (DP, monomer sequence, mass) without
    creating any atoms. Use this to validate distribution parameters
    before committing to an expensive build.

    Preferred for:
        - Previewing system composition before building.
        - Iterating on distribution parameters cheaply.

    Avoid when:
        - You want chains built directly (use BuildSystem or polymer_system).

    Attributes:
        random_seed: Random seed for reproducibility.

    Related:
        - BuildPolymer
        - BuildSystem
        - polymer_system
    """

    random_seed: int | None = None

    def run(
        self,
        monomer_weights: dict[str, float],
        monomer_mass: dict[str, float],
        distribution_type: str,
        distribution_params: dict[str, float],
        target_total_mass: float,
        end_group_mass: float = 0.0,
        max_rel_error: float = 0.02,
    ) -> dict[str, Any]:
        """Plan a polydisperse polymer system.

        Args:
            monomer_weights: Weight fractions for each monomer label.
            monomer_mass: Molar mass (g/mol) per monomer label.
            distribution_type: Distribution name (e.g. ``"schulz_zimm"``).
            distribution_params: Distribution parameters as ``{"p0": ..., "p1": ...}``.
            target_total_mass: Target total system mass (g/mol).
            end_group_mass: Mass of end groups per chain (g/mol).
            max_rel_error: Maximum relative error for total mass.

        Returns:
            Dict with ``"chains"`` (list of chain dicts), ``"total_mass"``,
            and ``"target_mass"``.
        """
        import numpy as np

        from molpy.builder.polymer.distributions import create_polydisperse_from_ir
        from molpy.builder.polymer.sequences import WeightedSequenceGenerator
        from molpy.builder.polymer.system import (
            PolydisperseChainGenerator,
            SystemPlanner,
        )
        from molpy.parser.smiles.gbigsmiles_ir import DistributionIR

        distribution = create_polydisperse_from_ir(
            DistributionIR(name=distribution_type, params=distribution_params),
            random_seed=self.random_seed,
        )

        seq_gen = WeightedSequenceGenerator(monomer_weights)
        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass=monomer_mass,
            end_group_mass=end_group_mass,
            distribution=distribution,
        )
        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=target_total_mass,
            max_rel_error=max_rel_error,
        )

        rng = np.random.default_rng(self.random_seed)
        plan = planner.plan_system(rng)

        return {
            "chains": [
                {"dp": c.dp, "monomers": c.monomers, "mass": c.mass}
                for c in plan.chains
            ],
            "total_mass": plan.total_mass,
            "target_mass": plan.target_mass,
        }


@dataclass(frozen=True)
class BuildSystem(Tool):
    """End-to-end polymer system construction from G-BigSMILES.

    Parses a G-BigSMILES string and delegates to the GBigSmilesCompiler
    to produce a list of Atomistic chains.

    Preferred for:
        - Building a complete polydisperse system in one call.
        - When you do not need to inspect the system plan before building.

    Avoid when:
        - You need to inspect or modify the plan first (use PlanSystem + BuildPolymer).
        - You need the Amber backend (use BuildPolymerAmber).

    Attributes:
        reaction_preset: Name of reaction preset.
        add_hydrogens: Add explicit hydrogens during monomer preparation.
        optimize: Optimize monomer geometry.
        random_seed: Random seed for reproducibility.

    Related:
        - PlanSystem
        - polymer_system
        - BuildPolymerAmber
    """

    reaction_preset: str = "dehydration"
    add_hydrogens: bool = True
    optimize: bool = True
    random_seed: int | None = None

    def run(self, gbigsmiles: str) -> list[Atomistic]:
        """Build a polymer system from a G-BigSMILES string.

        Args:
            gbigsmiles: G-BigSMILES notation
                (e.g. ``"{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|"``).

        Returns:
            List of Atomistic structures (one per chain).
        """
        from molpy.builder.polymer.compiler import CompilerConfig, GBigSmilesCompiler
        from molpy.parser.smiles import parse_gbigsmiles

        system_ir = parse_gbigsmiles(gbigsmiles)
        config = CompilerConfig(
            reaction_preset=self.reaction_preset,
            add_hydrogens=self.add_hydrogens,
            optimize_geometry=self.optimize,
            random_seed=self.random_seed,
        )
        compiler = GBigSmilesCompiler(config)
        return compiler.compile_and_build(system_ir)


# ---------------------------------------------------------------------------
# Amber backend tool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildPolymerAmber(Tool):
    """Build a polymer chain using the AmberTools backend.

    Uses antechamber, parmchk2, prepgen, and tleap to assemble
    a polymer from a CGSmiles string and a monomer library. Returns
    both MolPy structures and AMBER topology/coordinate files.

    Preferred for:
        - Polymer systems that need AMBER force field parameters (GAFF/GAFF2).
        - Workflows that feed into AMBER or LAMMPS with AMBER-style inputs.

    Avoid when:
        - You do not need force field parameters (use BuildPolymer).
        - AmberTools is not installed.

    Attributes:
        reaction_preset: Named preset for leaving group detection.
            When None, hydrogen atoms bonded to port atoms are
            auto-detected.
        force_field: Amber force field (``"gaff"`` or ``"gaff2"``).
        charge_method: Antechamber charge method.
        conda_env: Conda environment containing AmberTools.
        work_dir: Directory for intermediate files.

    Related:
        - BuildPolymer
        - polymer (with backend="amber")
    """

    reaction_preset: str | None = "dehydration"
    force_field: str = "gaff2"
    charge_method: str = "bcc"
    conda_env: str | None = None
    work_dir: str = "amber_work"

    def run(self, cgsmiles: str, library: dict[str, Atomistic]) -> dict[str, Any]:
        """Build a polymer using AmberTools.

        Args:
            cgsmiles: CGSmiles notation (e.g. ``"{[#EO]|10}"``).
            library: Mapping from label to prepared Atomistic monomer.
                Each monomer must have ``port="<"`` (head) and
                ``port=">"`` (tail) annotations.

        Returns:
            Dict with ``"frame"``, ``"forcefield"``, ``"prmtop_path"``,
            ``"inpcrd_path"``, ``"pdb_path"``, ``"monomer_count"``.
        """
        from pathlib import Path

        from molpy.builder.polymer.ambertools import AmberPolymerBuilder

        builder = AmberPolymerBuilder(
            library=library,
            force_field=self.force_field,  # type: ignore[arg-type]
            charge_method=self.charge_method,
            work_dir=Path(self.work_dir),
            reaction_preset=self.reaction_preset,
            env=self.conda_env if self.conda_env is not None else None,
            env_manager="conda" if self.conda_env is not None else None,
        )
        result = builder.build(cgsmiles)

        return {
            "frame": result.frame,
            "forcefield": result.forcefield,
            "prmtop_path": result.prmtop_path,
            "inpcrd_path": result.inpcrd_path,
            "pdb_path": result.pdb_path,
            "monomer_count": result.monomer_count,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def polymer(
    spec: str,
    *,
    library: Mapping[str, Atomistic] | None = None,
    reaction_preset: str = "dehydration",
    use_placer: bool = True,
    add_hydrogens: bool = True,
    optimize: bool = True,
    random_seed: int | None = None,
    backend: Backend = "default",
    amber_config: Any = None,
) -> Atomistic | Any:
    """Build a single polymer chain from a string specification.

    Auto-detects notation type (for the default backend):

    - **G-BigSMILES** (contains ``|`` annotation):
      ``polymer("{[<]CCOCC[>]}|10|")``
    - **CGSmiles + inline fragments** (contains ``.{#``):
      ``polymer("{[#EO]|10}.{#EO=[<]COC[>]}")``
    - **Pure CGSmiles** (requires ``library`` kwarg):
      ``polymer("{[#EO]|10}", library={"EO": eo_monomer})``

    For the Amber backend:

    - ``polymer("{[#EO]|10}", library={"EO": eo}, backend="amber")``

    Args:
        spec: Polymer specification string.
        library: Monomer library (required for pure CGSmiles and Amber).
        reaction_preset: Reaction preset name.
        use_placer: Enable geometric placement (default backend only).
        add_hydrogens: Add hydrogens during 3D generation.
        optimize: Optimize geometry.
        random_seed: Random seed for reproducibility.
        backend: Builder backend — ``"default"`` or ``"amber"``.
        amber_config: Optional ``AmberPolymerBuilderConfig`` for fine-grained
            control of the Amber backend. When None, defaults are used.

    Returns:
        Atomistic (default backend) or AmberBuildResult (amber backend).

    Related:
        - polymer_system
        - PrepareMonomer
        - BuildPolymer
        - BuildPolymerAmber
    """
    if backend == "amber":
        if library is None:
            raise TypeError(
                "Amber backend requires 'library' kwarg with port-annotated monomers."
            )
        return _build_amber(
            spec,
            library=library,
            reaction_preset=reaction_preset,
            amber_config=amber_config,
        )

    notation = _detect_notation(spec)

    if notation == "gbigsmiles":
        return _build_from_gbigsmiles(
            spec,
            reaction_preset=reaction_preset,
            add_hydrogens=add_hydrogens,
            optimize=optimize,
            random_seed=random_seed,
        )

    if notation == "cgsmiles_fragments":
        return _build_from_cgsmiles_with_fragments(
            spec,
            reaction_preset=reaction_preset,
            use_placer=use_placer,
            add_hydrogens=add_hydrogens,
            optimize=optimize,
        )

    # Pure CGSmiles
    if library is None:
        raise TypeError(
            "Pure CGSmiles notation requires 'library' kwarg. "
            "Provide a dict mapping labels to Atomistic monomers, "
            "or use G-BigSMILES notation with inline monomer definitions."
        )
    return _build_from_cgsmiles(
        spec,
        library=library,
        reaction_preset=reaction_preset,
        use_placer=use_placer,
    )


def polymer_system(
    spec: str,
    *,
    reaction_preset: str = "dehydration",
    add_hydrogens: bool = True,
    optimize: bool = True,
    random_seed: int | None = None,
) -> list[Atomistic]:
    """Build a multi-chain polymer system from G-BigSMILES.

    Example::

        chains = polymer_system(
            "{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|",
            random_seed=42,
        )

    Args:
        spec: G-BigSMILES specification string.
        reaction_preset: Reaction preset name.
        add_hydrogens: Add hydrogens during 3D generation.
        optimize: Optimize geometry.
        random_seed: Random seed for reproducibility.

    Returns:
        List of Atomistic structures (one per chain).

    Related:
        - polymer
        - BuildSystem
        - PlanSystem
    """
    from molpy.builder.polymer.compiler import CompilerConfig, GBigSmilesCompiler
    from molpy.parser.smiles import parse_gbigsmiles

    system_ir = parse_gbigsmiles(spec)
    config = CompilerConfig(
        reaction_preset=reaction_preset,
        add_hydrogens=add_hydrogens,
        optimize_geometry=optimize,
        random_seed=random_seed,
    )
    compiler = GBigSmilesCompiler(config)
    return compiler.compile_and_build(system_ir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_notation(spec: str) -> str:
    """Detect which notation the spec string uses.

    Returns:
        ``"gbigsmiles"``, ``"cgsmiles_fragments"``, or ``"cgsmiles"``.
    """
    if "}|" in spec:
        return "gbigsmiles"
    if ".{#" in spec:
        return "cgsmiles_fragments"
    return "cgsmiles"


def _build_from_gbigsmiles(
    spec: str,
    *,
    reaction_preset: str,
    add_hydrogens: bool,
    optimize: bool,
    random_seed: int | None,
) -> Atomistic:
    """Build single chain from G-BigSMILES."""
    from molpy.builder.polymer.compiler import CompilerConfig, GBigSmilesCompiler
    from molpy.parser.smiles import parse_gbigsmiles

    system_ir = parse_gbigsmiles(spec)
    config = CompilerConfig(
        reaction_preset=reaction_preset,
        add_hydrogens=add_hydrogens,
        optimize_geometry=optimize,
        random_seed=random_seed,
    )
    compiler = GBigSmilesCompiler(config)
    chains = compiler.compile_and_build(system_ir)

    if not chains:
        raise ValueError(f"G-BigSMILES compilation produced no chains: {spec!r}")

    return chains[0]


def _build_from_cgsmiles_with_fragments(
    spec: str,
    *,
    reaction_preset: str,
    use_placer: bool,
    add_hydrogens: bool,
    optimize: bool,
) -> Atomistic:
    """Build from CGSmiles with inline fragment definitions."""
    parts = spec.split(".{#")
    cgsmiles_part = parts[0]
    fragments: dict[str, str] = {}
    for frag_part in parts[1:]:
        frag_part = frag_part.rstrip("}")
        eq_pos = frag_part.index("=")
        label = frag_part[:eq_pos]
        smiles_body = frag_part[eq_pos + 1 :]
        fragments[label] = smiles_body

    preparer = PrepareMonomer(
        add_hydrogens=add_hydrogens,
        optimize=optimize,
    )
    library: dict[str, Atomistic] = {}
    for label, smiles_body in fragments.items():
        bigsmiles_str = "{" + smiles_body + "}"
        library[label] = preparer.run(bigsmiles_str)

    build_tool = BuildPolymer(
        reaction_preset=reaction_preset,
        use_placer=use_placer,
    )
    result = build_tool.run(cgsmiles_part, library)
    return result["polymer"]


def _build_from_cgsmiles(
    spec: str,
    *,
    library: Mapping[str, Atomistic],
    reaction_preset: str,
    use_placer: bool,
) -> Atomistic:
    """Build from pure CGSmiles with external library."""
    build_tool = BuildPolymer(
        reaction_preset=reaction_preset,
        use_placer=use_placer,
    )
    result = build_tool.run(spec, dict(library))
    return result["polymer"]


def _build_amber(
    spec: str,
    *,
    library: Mapping[str, Atomistic],
    reaction_preset: str,
    amber_config: Any,
) -> Any:
    """Build polymer using the AmberTools backend."""
    from molpy.builder.polymer.ambertools import AmberPolymerBuilder

    kwargs: dict = {"reaction_preset": reaction_preset}
    if amber_config is not None:
        # Accept dict-like overrides for backwards compat
        if hasattr(amber_config, "__dict__"):
            kwargs.update(
                {k: v for k, v in amber_config.__dict__.items() if v is not None}
            )

    builder = AmberPolymerBuilder(library=library, **kwargs)
    return builder.build(spec)
