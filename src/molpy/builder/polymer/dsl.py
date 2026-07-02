"""Declarative polymer-building entry functions.

Start here for polymer construction:

- ``polymer()`` — auto-detect notation, build a single chain
- ``polymer_system()`` — G-BigSMILES → polydisperse multi-chain system
- ``prepare_monomer()`` — BigSMILES → 3D Atomistic with ports
- ``generate_3d()`` — embed an existing Atomistic in 3D
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from molpy.core.atomistic import Atomistic

Backend = Literal["default", "amber"]


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
        amber_config: Optional object whose attributes override
            :class:`~molpy.builder.polymer.ambertools.AmberPolymerBuilder`
            keyword arguments (e.g. force_field, charge_method). When
            None, defaults are used.

    Returns:
        Atomistic (default backend) or AmberBuildResult (amber backend).

    Related:
        - polymer_system
        - prepare_monomer
        - PolymerBuilder
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
        - prepare_monomer
        - ReactionPresets
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


def prepare_monomer(
    bigsmiles: str,
    typifier: Any = None,
    *,
    add_hydrogens: bool = True,
    optimize: bool = True,
    gen_angle: bool = True,
    gen_dihe: bool = True,
) -> Atomistic:
    """Parse, embed in 3D, augment topology, and optionally typify a monomer.

    Bundles the four-step pattern that appears in every polymer-building
    workflow::

        m = mp.parser.parse_monomer(bigsmiles)
        m = generate_3d(m, add_hydrogens=True, optimize=True)
        m = m.get_topo(gen_angle=True, gen_dihe=True)
        m = typifier.typify(m)

    Args:
        bigsmiles: BigSMILES string (e.g. ``"{[][<]OCCOCCOCCO[>][]}"``).
        typifier: Optional typifier instance (e.g. ``OplsTypifier``).
            When provided, force-field types are assigned before returning.
        add_hydrogens: Add implicit hydrogens during 3D generation.
        optimize: Run force-field geometry optimisation after embedding.
        gen_angle: Generate angle interactions from bonds.
        gen_dihe: Generate dihedral interactions from bonds.

    Returns:
        Fully prepared Atomistic monomer ready for reactions or export.
    """
    from molpy.parser import parse_monomer

    mol = parse_monomer(bigsmiles)
    mol = generate_3d(mol, add_hydrogens=add_hydrogens, optimize=optimize)
    if gen_angle or gen_dihe:
        mol = mol.get_topo(gen_angle=gen_angle, gen_dihe=gen_dihe)
    if typifier is not None:
        mol = typifier.typify(mol)
    return mol


def generate_3d(
    mol: Atomistic,
    add_hydrogens: bool = True,
    optimize: bool = True,
) -> Atomistic:
    """Generate 3D coordinates for a molecular structure via RDKit.

    Thin re-export of :func:`molpy.adapter.rdkit.generate_3d` for use inside
    polymer-building workflows.

    Args:
        mol: Atomistic structure (typically from parser.parse_molecule)
        add_hydrogens: Add implicit hydrogens before embedding
        optimize: Run force-field geometry optimization after embedding

    Returns:
        New Atomistic with 3D coordinates and (optionally) explicit hydrogens

    Raises:
        ImportError: if RDKit is not installed
    """
    try:
        from molpy.adapter.rdkit import generate_3d as _rdkit_generate_3d
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "RDKit is required for 3D coordinate generation. "
            "Install with: pip install rdkit"
        ) from exc

    return _rdkit_generate_3d(mol, add_hydrogens=add_hydrogens, optimize=optimize)


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

    from molpy.builder.polymer.tools import BuildPolymer, PrepareMonomer

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
    from molpy.builder.polymer.tools import BuildPolymer

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
        # Accept attribute-bearing config objects; plain dicts are NOT
        # read (attributes only) — see AmberPolymerBuilder for kwargs.
        if hasattr(amber_config, "__dict__"):
            kwargs.update(
                {k: v for k, v in amber_config.__dict__.items() if v is not None}
            )

    builder = AmberPolymerBuilder(library=library, **kwargs)
    return builder.build(spec)
