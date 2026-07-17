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
    >>> from molpy.io import read_lammps_trajectory
    >>> traj = read_lammps_trajectory("dump.lammpstrj")
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
from .cluster import Cluster, ClusterCenters, ClusterProperties
from .decomposition import DescriptorRow, KMeans, Pca
from .dielectric import (
    ACFAnalyzer,
    DielectricSusceptibility,
    IonicConductivity,
    SpectralAnalyzer,
)
from .density import GaussianDensity, LocalDensity
from .diffraction import StaticStructureFactorDebye
from .environment import BondOrder
from .order import Hexatic, Nematic, SolidLiquid, Steinhardt
from .pmft import PMFTXY
from .mcd import MCDCompute
from .msd import MSD
from .neighborlist import NeighborList
from .onsager import Onsager
from .jacf import JACF
from .persist import Persist
from .pmsd import PMSDCompute
from .rdf import RDF
from .result import (
    ACFResult,
    ConductivityResult,
    DebyeFit,
    DielectricResult,
    DielectricSusceptibilityResult,
    JACFResult,
    MCDResult,
    OnsagerResult,
    PersistResult,
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

# analysis-parity computes (geometric / combined / spatial distributions, Van Hove,
# reorientation, hydrogen bonds, radical Voronoi, vibrational spectra). The
# numerical kernels live in molrs; these are thin typed shells.
from .distribution import (
    AngleDistribution,
    CombinedDistribution,
    DihedralDistribution,
    DistanceDistribution,
)
from .spatial import SpatialDistribution
from .van_hove import VanHove
from .reorientation import LegendreReorientation
from .hbond import HBondCriterion, HBonds
from .voronoi import (
    RadicalVoronoi,
    VoronoiCells,
    VoronoiIntegration,
    voronoi_domains,
    voronoi_voids,
)
from .spectra import (
    IRSpectrum,
    PowerSpectrum,
    RamanSpectrum,
    ResonanceRamanSpectrum,
    RoaSpectrum,
    VcdSpectrum,
)

__all__ = [
    "Compute",
    "Result",
    "TimeSeriesResult",
    "MCDResult",
    "PMSDResult",
    "OnsagerResult",
    "JACFResult",
    "PersistResult",
    "ACFResult",
    "SpectralResult",
    "DielectricResult",
    "DielectricSusceptibilityResult",
    "ConductivityResult",
    "DebyeFit",
    "MCDCompute",
    "PMSDCompute",
    "Onsager",
    "JACF",
    "Persist",
    "ACFAnalyzer",
    "SpectralAnalyzer",
    "DielectricSusceptibility",
    "IonicConductivity",
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
    "Steinhardt",
    "Hexatic",
    "Nematic",
    "SolidLiquid",
    "LocalDensity",
    "GaussianDensity",
    "StaticStructureFactorDebye",
    "BondOrder",
    "PMFTXY",
    "ClusterProperties",
    # analysis-parity computes
    "DistanceDistribution",
    "AngleDistribution",
    "DihedralDistribution",
    "CombinedDistribution",
    "SpatialDistribution",
    "VanHove",
    "LegendreReorientation",
    "HBonds",
    "HBondCriterion",
    "RadicalVoronoi",
    "VoronoiCells",
    "VoronoiIntegration",
    "voronoi_domains",
    "voronoi_voids",
    "PowerSpectrum",
    "IRSpectrum",
    "RamanSpectrum",
    "VcdSpectrum",
    "RoaSpectrum",
    "ResonanceRamanSpectrum",
]
