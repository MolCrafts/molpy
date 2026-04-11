"""Tools and compute operations for molecular modeling.

Tools are high-level workflows that wrap multiple MolPy modules into
single-call operations. Compute classes are analysis operations
on trajectory data.

Tool examples::

    from molpy.tool import PrepareMonomer, BuildPolymer, polymer

    prep = PrepareMonomer()
    eo = prep.run("{[<]CCO[>]}")

    chain = polymer("{[<]CCO[>]}|10|")

Compute examples::

    from molpy.tool import MSD, DisplacementCorrelation

    msd = MSD(max_lag=3000)
    msd_values = msd(unwrapped_coords)           # -> NDArray (max_lag,)

    xdc = DisplacementCorrelation(max_lag=3000)
    corr = xdc(cation_coords, anion_coords)       # -> NDArray (max_lag,)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic

# Base classes
from .base import Compute, Tool, ToolRegistry

# Compute operations (analysis)
from .cross_correlation import DisplacementCorrelation, displacement_correlation
from .msd import MSD
from .msd import msd as compute_msd_shorthand
from .time_series import TimeAverage, TimeCache, compute_acf, compute_msd

# Tool operations (polymer-building workflows)
from .polymer import (
    BuildPolymer,
    BuildPolymerAmber,
    BuildSystem,
    PlanSystem,
    PrepareMonomer,
    polymer,
    polymer_system,
)

# Optional RDKit compute nodes
try:  # pragma: no cover
    from .rdkit import Generate3D, OptimizeGeometry

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing
    _HAS_RDKIT = False
    Generate3D = None  # type: ignore[assignment]
    OptimizeGeometry = None  # type: ignore[assignment]


def prepare_monomer(
    bigsmiles: str,
    typifier=None,
    *,
    add_hydrogens: bool = True,
    optimize: bool = True,
    gen_angle: bool = True,
    gen_dihe: bool = True,
) -> "Atomistic":
    """Parse, embed in 3D, augment topology, and optionally typify a monomer.

    Bundles the four-step pattern that appears in every polymer-building
    workflow::

        m = mp.parser.parse_monomer(bigsmiles)
        m = mp.tool.generate_3d(m, add_hydrogens=True, optimize=True)
        m = m.get_topo(gen_angle=True, gen_dihe=True)
        m = typifier.typify(m)

    Args:
        bigsmiles: BigSMILES string (e.g. ``"{[][<]OCCOCCOCCO[>][]}"``).
        typifier: Optional typifier instance (e.g. ``OplsAtomisticTypifier``).
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
    mol: "Atomistic",
    add_hydrogens: bool = True,
    optimize: bool = True,
) -> "Atomistic":
    """Generate 3D coordinates for a molecular structure via RDKit.

    Wraps RDKitAdapter + Generate3D into a single convenience call.

    Args:
        mol: Atomistic structure (typically from parser.parse_molecule)
        add_hydrogens: Add implicit hydrogens before embedding
        optimize: Run force-field geometry optimization after embedding

    Returns:
        New Atomistic with 3D coordinates and (optionally) explicit hydrogens

    Raises:
        ImportError: if RDKit is not installed
    """
    from molpy.adapter import RDKitAdapter

    if RDKitAdapter is None or Generate3D is None:
        raise ImportError(
            "RDKit is required for 3D coordinate generation. "
            "Install with: pip install rdkit"
        )

    adapter = RDKitAdapter(internal=mol)
    compute = Generate3D(
        add_hydrogens=add_hydrogens,
        embed=True,
        optimize=optimize,
        update_internal=True,
    )
    adapter = compute(adapter)
    return adapter.get_internal()


__all__ = [
    # Base
    "Compute",
    "Tool",
    "ToolRegistry",
    # Compute operations
    "MSD",
    "DisplacementCorrelation",
    "TimeCache",
    "TimeAverage",
    "compute_msd",
    "compute_msd_shorthand",
    "compute_acf",
    "displacement_correlation",
    "generate_3d",
    "prepare_monomer",
    # Tool operations (polymer building)
    "PrepareMonomer",
    "BuildPolymer",
    "PlanSystem",
    "BuildSystem",
    "BuildPolymerAmber",
    "polymer",
    "polymer_system",
]

if _HAS_RDKIT:
    __all__ += ["Generate3D", "OptimizeGeometry"]
