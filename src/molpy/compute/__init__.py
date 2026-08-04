"""Trajectory and structure analyses.

Configure a compute, call it on frames or pre-assembled arrays, read typed
fields. Analysis time is femtoseconds (LAMMPS real units).

Transport and dielectric quantities are composed explicitly::

    raw curve  →  Fit  →  optional SI scale in your script

Example::

    >>> from molpy.compute import Onsager, EinsteinConductivity, LinearFit
    >>> L = Onsager.correlation(P_i, P_j, dt=10.0, max_correlation_time=500)
    >>> raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=500)
    >>> fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])

"""

from .base import Compute
from .cluster import Cluster, ClusterCenters, ClusterProperties
from .decomposition import DescriptorRow, KMeans, Pca
from .dielectric import (
    CumulativeTrapezoid,
    DebyeFit,
    DebyeRelaxation,
    Dielectric,
    EinsteinHelfandSpectrum,
    GreenKuboSpectrum,
    LinearFit,
    acf_fft,
    apply_window,
    frequency_grid,
)
from .density import GaussianDensity, LocalDensity
from .diffraction import StaticStructureFactorDebye
from .environment import BondOrder
from .order import Hexatic, Nematic, SolidLiquid, Steinhardt
from .pmft import PMFTXY
from .msd import MSD
from .neighborlist import NeighborList
from .onsager import Onsager
from .jacf import GreenKuboConductivity
from .persist import Persist
from .pmsd import EinsteinConductivity
from .rdf import RDF
from .result import (
    ACFResult,
    ConductivityResult,
    DebyeSpectrumFit,
    DielectricResult,
    DielectricSusceptibilityResult,
    JACFResult,
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
from .workflow import (
    Workflow,
    WorkflowCycleError,
    WorkflowDuplicateNodeError,
    WorkflowError,
    WorkflowMissingInputError,
)

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

from molrs.compute.cluster import (
    CenterOfMassResult,
    ClusterCentersResult,
    ClusterResult,
)
from molrs.compute.density import RDFResult
from molrs.compute.dynamics import Acf, AcfResult
from molrs.compute.ml import KMeansResult, Pca2, PcaResult
from molrs.compute.msd import MSDResult, MSDTimeSeries
from molrs.compute.spectroscopy import (
    conductivity_sum_rule,
    kramers_kronig,
    polarizability_finite_field,
    route_agreement,
)
from molrs.compute.voronoi import DensityGrid, MolecularMoments

from . import signal
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
    "PMSDResult",
    "OnsagerResult",
    "JACFResult",
    "PersistResult",
    "ACFResult",
    "SpectralResult",
    "DielectricResult",
    "DielectricSusceptibilityResult",
    "ConductivityResult",
    "DebyeSpectrumFit",
    "EinsteinConductivity",
    "Onsager",
    "GreenKuboConductivity",
    "Persist",
    "Dielectric",
    "DebyeRelaxation",
    "DebyeFit",
    "EinsteinHelfandSpectrum",
    "GreenKuboSpectrum",
    "LinearFit",
    "CumulativeTrapezoid",
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
    "Workflow",
    "WorkflowCycleError",
    "WorkflowDuplicateNodeError",
    "WorkflowError",
    "WorkflowMissingInputError",
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
    "DensityGrid",
    "MolecularMoments",
    "polarizability_finite_field",
    "conductivity_sum_rule",
    "kramers_kronig",
    "route_agreement",
    "Acf",
    "AcfResult",
    "Dielectric",
    "CenterOfMassResult",
    "ClusterCentersResult",
    "ClusterResult",
    "KMeansResult",
    "Pca2",
    "PcaResult",
    "MSDResult",
    "MSDTimeSeries",
    "RDFResult",
    "signal",
    "voronoi_domains",
    "voronoi_voids",
    "PowerSpectrum",
    "IRSpectrum",
    "RamanSpectrum",
    "VcdSpectrum",
    "RoaSpectrum",
    "ResonanceRamanSpectrum",
]
