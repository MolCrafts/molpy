"""Trajectory and structure analyses.

Every analysis here is a native class re-exported by identity
(``molpy.compute.RDF is molrs.compute.density.RDF``); molpy adds no wrapper.
Configure a compute, run ``.compute(...)`` on frames or pre-assembled arrays,
read typed fields. Analysis time is femtoseconds (LAMMPS real units).

Pair-based analyses (``RDF``, ``LocalDensity``, ``Steinhardt``, ``BondOrder``,
``PMFTXY``, ``Cluster``, …) take neighbour tables from the core
``molpy.NeighborList``::

    >>> nl = mp.NeighborList(cutoff)
    >>> nl.build(frame.coords, frame.box)
    >>> g = mp.compute.RDF(n_bins, cutoff).compute([frame], [nl.neighbors()])

Transport and dielectric quantities are composed explicitly::

    raw curve  →  Fit  →  optional SI scale in your script

Example::

    >>> from molpy.compute import Onsager, EinsteinConductivity, LinearFit
    >>> L = Onsager.correlation(P_i, P_j, dt=10.0, max_correlation_time=500)
    >>> raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=500)
    >>> fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])

"""

from molrs import signal
from molrs.compute import Compute
from molrs.compute.cluster import (
    CenterOfMass,
    CenterOfMassResult,
    Cluster,
    ClusterCenters,
    ClusterCentersResult,
    ClusterProperties,
    ClusterResult,
    GyrationTensor,
    InertiaTensor,
    RadiusOfGyration,
)
from molrs.compute.density import (
    RDF,
    GaussianDensity,
    LocalDensity,
    RDFResult,
    SpatialDistribution,
    SpatialDistributionResult,
)
from molrs.compute.dielectric import Dielectric
from molrs.compute.diffraction import StaticStructureFactorDebye
from molrs.compute.distribution import (
    AngleDistribution,
    CombinedDistribution,
    CombinedDistributionResult,
    DihedralDistribution,
    DistanceDistribution,
    DistributionResult,
)
from molrs.compute.dynamics import Acf, AcfResult, VanHove, VanHoveResult
from molrs.compute.environment import BondOrder
from molrs.compute.fitting import CumulativeTrapezoid, LinearFit, Plateau
from molrs.compute.hbond import HBondCriterion, HBonds, HBondsResult
from molrs.compute.ml import DescriptorRow, KMeans, KMeansResult, Pca2, PcaResult
from molrs.compute.msd import MSD, MSDResult, MSDTimeSeries
from molrs.compute.order import (
    Hexatic,
    LegendreReorientation,
    LegendreReorientationResult,
    Nematic,
    SolidLiquid,
    Steinhardt,
)
from molrs.compute.pmft import PMFTXY
from molrs.compute.spectroscopy import (
    EinsteinHelfandSpectrum,
    GreenKuboSpectrum,
    IRSpectrum,
    PowerSpectrum,
    RamanSpectrum,
    ResonanceRamanSpectrum,
    RoaSpectrum,
    VcdSpectrum,
    conductivity_sum_rule,
    kramers_kronig,
    polarizability_finite_field,
    route_agreement,
)
from molrs.compute.transport import (
    VACF,
    DebyeFit,
    DebyeRelaxation,
    EinsteinConductivity,
    EinsteinDiffusion,
    GreenKuboConductivity,
    GreenKuboDiffusion,
    Onsager,
    Persist,
)
from molrs.compute.voronoi import (
    DensityGrid,
    MolecularMoments,
    RadicalVoronoi,
    VoronoiCells,
    VoronoiIntegration,
    voronoi_domains,
    voronoi_voids,
)

__all__ = [
    "Acf",
    "AcfResult",
    "AngleDistribution",
    "BondOrder",
    "CenterOfMass",
    "CenterOfMassResult",
    "Cluster",
    "ClusterCenters",
    "ClusterCentersResult",
    "ClusterProperties",
    "ClusterResult",
    "CombinedDistribution",
    "CombinedDistributionResult",
    "Compute",
    "CumulativeTrapezoid",
    "DebyeFit",
    "DebyeRelaxation",
    "DensityGrid",
    "DescriptorRow",
    "Dielectric",
    "DihedralDistribution",
    "DistanceDistribution",
    "DistributionResult",
    "EinsteinConductivity",
    "EinsteinDiffusion",
    "EinsteinHelfandSpectrum",
    "GaussianDensity",
    "GreenKuboConductivity",
    "GreenKuboDiffusion",
    "GreenKuboSpectrum",
    "GyrationTensor",
    "HBondCriterion",
    "HBonds",
    "HBondsResult",
    "Hexatic",
    "IRSpectrum",
    "InertiaTensor",
    "KMeans",
    "KMeansResult",
    "LegendreReorientation",
    "LegendreReorientationResult",
    "LinearFit",
    "LocalDensity",
    "MSD",
    "MSDResult",
    "MSDTimeSeries",
    "MolecularMoments",
    "Nematic",
    "Onsager",
    "PMFTXY",
    "Pca2",
    "PcaResult",
    "Persist",
    "Plateau",
    "PowerSpectrum",
    "RDF",
    "RDFResult",
    "RadicalVoronoi",
    "RadiusOfGyration",
    "RamanSpectrum",
    "ResonanceRamanSpectrum",
    "RoaSpectrum",
    "SolidLiquid",
    "SpatialDistribution",
    "SpatialDistributionResult",
    "StaticStructureFactorDebye",
    "Steinhardt",
    "VACF",
    "VanHove",
    "VanHoveResult",
    "VcdSpectrum",
    "VoronoiCells",
    "VoronoiIntegration",
    "conductivity_sum_rule",
    "kramers_kronig",
    "polarizability_finite_field",
    "route_agreement",
    "signal",
    "voronoi_domains",
    "voronoi_voids",
]
