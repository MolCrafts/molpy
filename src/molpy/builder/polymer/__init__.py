"""Polymer assembly — compose the real builder classes directly.

There is no ``polymer()`` convenience dispatcher. You build a chain by
calling the engine classes yourself, which keeps the data flow explicit:

1. Prepare monomers — parse a BigSMILES repeat unit and embed it in 3D with
   molpy's native (molrs) conformer generator::

       from molpy.conformer import Conformer
       from molpy.parser import parse_monomer

       eo, _ = Conformer(add_hydrogens=True, seed=42).generate(parse_monomer("{[<]CCO[>]}"))

2. Assemble a chain with :class:`PolymerBuilder` — feed it a monomer
   ``library`` + a :class:`~molpy.reacter.Reacter` (from
   :class:`ReactionPresets`), then either a CGSmiles string or a plain
   label sequence::

       from molpy.builder.polymer import PolymerBuilder, ReactionPresets

       builder = PolymerBuilder({"EO": eo}, reacter=ReactionPresets.get("dehydration"))
       chain = builder.build_sequence(["EO"] * 10).polymer

For a **polydisperse system**, drive :class:`PolymerBuilder` from the
distribution + planner primitives yourself — sample a plan, then loop::

       import numpy as np
       from molpy.builder.polymer import (
           PolydisperseChainGenerator, SchulzZimmPolydisperse,
           SystemPlanner, WeightedSequenceGenerator,
       )

       planner = SystemPlanner(
           PolydisperseChainGenerator(
               WeightedSequenceGenerator({"EO": 1.0}),
               {"EO": 44.05},
               distribution=SchulzZimmPolydisperse(1500, 3000),
           ),
           target_total_mass=5e5,
       )
       plan = planner.plan_system(np.random.default_rng(42))
       chains = [builder.build_sequence(c.monomers).polymer for c in plan.chains]

The AmberTools-backed build (:class:`AmberPolymerBuilder`) lives in
:mod:`molpy.builder.polymer.ambertools`.
"""

from .connectors import Connector
from .core import PolymerBuilder, PolymerBuildResult
from .distributions import (
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
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
    # Chain assembly engine
    "PolymerBuilder",
    "PolymerBuildResult",
    "Connector",
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
    # System planning
    "Chain",
    "SystemPlan",
    "SystemPlanner",
    "PolydisperseChainGenerator",
]
