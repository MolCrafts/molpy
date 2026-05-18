"""Compute abstraction for molecular computations.

This module provides a unified interface for defining and executing
computational operations on molecular structures. The core abstractions are:

- Result, TimeSeriesResult, MCDResult, PMSDResult, ACFResult,
  SpectralResult, DielectricResult, DielectricSusceptibilityResult:
  result dataclasses returned by compute operations.
- Compute: Base class for computation operations.
- MCDCompute, PMSDCompute, RDF, NeighborList: structural / trajectory
  analyses.
- ACFAnalyzer, SpectralAnalyzer, DielectricSusceptibility: dielectric
  spectroscopy from MD trajectories (Einstein-Helfand & Green-Kubo).

Example: diffusion via mean displacement correlation:
    >>> from molpy.compute import MCDCompute
    >>> from molpy.io import read_h5_trajectory
    >>> traj = read_h5_trajectory("trajectory.h5")
    >>> mcd = MCDCompute(tags=["1"], max_dt=30.0, dt=0.01)
    >>> result = mcd(traj)
    >>> print(result.correlations["1"])  # MSD values at each time lag

Example: dielectric spectrum:
    >>> from molpy.compute import DielectricSusceptibility
    >>> dc = DielectricSusceptibility(
    ...     dt=0.001,                # ps
    ...     temperature=300.0,       # K
    ...     max_correlation_time=200,  # frames
    ...     routes=["einstein-helfand", "green-kubo"],
    ... )
    >>> result = dc(trajectory)
    >>> eh = result.results["EH-full"]
    >>> # eh.frequency: rad/ps, eh.epsilon_real/imag: dimensionless
"""

from .base import Compute
from .cluster import Cluster, ClusterCenters
from .decomposition import DescriptorRow, KMeans, Pca
from .dielectric import ACFAnalyzer, DielectricSusceptibility, SpectralAnalyzer
from .mcd import MCDCompute
from .msd import MSD
from .neighborlist import NeighborList
from .pmsd import PMSDCompute
from .rdf import RDF
from .result import (
    ACFResult,
    DielectricResult,
    DielectricSusceptibilityResult,
    MCDResult,
    PMSDResult,
    Result,
    SpectralResult,
    TimeSeriesResult,
)
from .shape import (
    CenterOfMass,
    GyrationTensor,
    InertiaTensor,
    RadiusOfGyration,
)
from .time_series import TimeAverage, TimeCache, compute_acf, compute_msd
from .workflow import (
    Workflow,
    WorkflowCycleError,
    WorkflowDuplicateNodeError,
    WorkflowError,
    WorkflowMissingInputError,
)

# Optional RDKit compute nodes (slated for removal in Phase 3 — replaced by
# the molrs-backed embed pipeline).
try:  # pragma: no cover
    from .rdkit import Generate3D, OptimizeGeometry

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing
    _HAS_RDKIT = False
    Generate3D = None  # type: ignore[assignment]
    OptimizeGeometry = None  # type: ignore[assignment]

__all__ = [
    "Compute",
    "Result",
    "TimeSeriesResult",
    "MCDResult",
    "PMSDResult",
    "ACFResult",
    "SpectralResult",
    "DielectricResult",
    "DielectricSusceptibilityResult",
    "MCDCompute",
    "PMSDCompute",
    "ACFAnalyzer",
    "SpectralAnalyzer",
    "DielectricSusceptibility",
    "NeighborList",
    "RDF",
    "MSD",
    "Cluster",
    "ClusterCenters",
    "CenterOfMass",
    "GyrationTensor",
    "InertiaTensor",
    "RadiusOfGyration",
    "DescriptorRow",
    "Pca",
    "KMeans",
    "TimeCache",
    "TimeAverage",
    "Workflow",
    "WorkflowCycleError",
    "WorkflowDuplicateNodeError",
    "WorkflowError",
    "WorkflowMissingInputError",
    "compute_msd",
    "compute_acf",
]

if _HAS_RDKIT:
    __all__ += ["Generate3D", "OptimizeGeometry"]
