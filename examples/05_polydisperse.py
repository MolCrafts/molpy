"""Polydisperse polymer system: sample a chain-length distribution.

A real melt is not monodisperse. You sample chain lengths from a
molecular-weight distribution, then build each chain. The distribution +
planner primitives do the sampling; you compose them directly (no wrapper):

    distribution + sequence generator -> PolydisperseChainGenerator
    -> SystemPlanner.plan_system() -> a list of Chain (dp, monomers, mass)

Each ``Chain`` carries its monomer sequence, ready to feed straight into
``PolymerBuilder.build_sequence`` (see 02_build_polymer).

Guide: docs/user-guide/05_polydisperse_systems.md
Run:   python 05_polydisperse.py
"""

import numpy as np

from molpy.builder.polymer import (
    PolydisperseChainGenerator,
    SchulzZimmPolydisperse,
    SystemPlanner,
    WeightedSequenceGenerator,
)


def main() -> None:
    # A styrene / methyl-acrylate copolymer, 80:20 by weight. Monomer molar
    # masses (g/mol) set the chain masses the planner accumulates.
    monomer_mass = {"Sty": 104.15, "MA": 86.09}
    seq_gen = WeightedSequenceGenerator(monomer_weights={"Sty": 8.0, "MA": 2.0})

    # Schulz-Zimm distribution in molecular-weight space (target Mn, Mw).
    distribution = SchulzZimmPolydisperse(Mn=1400, Mw=1500)

    planner = SystemPlanner(
        PolydisperseChainGenerator(seq_gen, monomer_mass, distribution=distribution),
        target_total_mass=5e5,  # keep sampling chains until the box mass is hit
        max_rel_error=0.02,
    )

    # Deterministic sampling via a seeded rng passed at plan time.
    plan = planner.plan_system(np.random.default_rng(42))

    mw = np.array([c.mass for c in plan.chains])
    Mn = float(mw.mean())
    Mw = float((mw**2).sum() / mw.sum())
    print(f"sampled {len(plan.chains)} chains for target mass {plan.target_mass:.0f}")
    print(f"  Mn={Mn:.0f}  Mw={Mw:.0f}  PDI={Mw / Mn:.3f}")

    # Each Chain is (dp, monomer sequence, mass) — hand it to build_sequence.
    first = plan.chains[0]
    print(f"  first chain: dp={first.dp}, sequence[:8]={first.monomers[:8]}")


if __name__ == "__main__":
    main()
