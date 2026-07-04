"""Polymer assembly — start here.

One-call entry points (most users need only these):

- :func:`polymer` — build a single chain from G-BigSMILES / CGSmiles
- :func:`polymer_system` — build a polydisperse multi-chain system
- :func:`prepare_monomer` — BigSMILES → 3D Atomistic with ports
- :func:`generate_3d` — embed an existing Atomistic in 3D

Step-by-step (advanced) API:

- :class:`PolymerBuilder` + :class:`Connector` — direct graph assembly
  with explicit port mapping and reaction control
- :class:`Placer` family (:class:`CovalentSeparator`,
  :class:`VdWSeparator`, :class:`LinearOrienter`) — geometric placement
- :class:`ReactionPresets` / :class:`ReactionPresetSpec` — named
  reaction chemistry; ``ReactionPresets.register()`` is the extension
  point for custom chemistries
- Sequence generators and polydispersity distributions for system
  planning

Internal machinery (importable but not part of the public surface):
``GBigSmilesCompiler``, ``SystemPlanner``, ``PolydisperseChainGenerator``.
"""

from .connectors import (
    Connector,
    ConnectorContext,
)
from .core import PolymerBuilder, PolymerBuildResult
from .distributions import (
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
)
from .dsl import (
    generate_3d,
    polymer,
    polymer_system,
    prepare_monomer,
)
from .placer import CovalentSeparator, LinearOrienter, Placer, VdWSeparator
from .presets import ReactionPresets, ReactionPresetSpec
from .sequences import (
    AlternatingSequenceGenerator,
    BlockSequenceGenerator,
    SequenceGenerator,
    WeightedSequenceGenerator,
)
from .system import (
    Chain,
    PolydisperseChainGenerator,
    SystemPlan,
    SystemPlanner,
)

__all__ = [
    # One-call entry functions
    "polymer",
    "polymer_system",
    "prepare_monomer",
    "generate_3d",
    # Step-by-step builder
    "PolymerBuilder",
    "PolymerBuildResult",
    "Connector",
    "ConnectorContext",
    # Reaction presets (public extension point)
    "ReactionPresets",
    "ReactionPresetSpec",
    # Placer
    "CovalentSeparator",
    "LinearOrienter",
    "Placer",
    "VdWSeparator",
    # Sequence generators
    "AlternatingSequenceGenerator",
    "BlockSequenceGenerator",
    "SequenceGenerator",
    "WeightedSequenceGenerator",
    # Distributions
    "DPDistribution",
    "MassDistribution",
    "FlorySchulzPolydisperse",
    "PoissonPolydisperse",
    "SchulzZimmPolydisperse",
    "UniformPolydisperse",
    # System planning data types
    "Chain",
    "SystemPlan",
]
