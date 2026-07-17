"""C7 — Prepolymer + tetrafunctional agent.

Linear EO chains keep free hydroxyl ends (SITE a/b). A small-molecule agent
with four SITE-a hydroxyls is packed in; ether condensation couples ends to
the agent through the same ETHER reaction used for chain growth.

Guide: docs/user-guide/topology/11_prepolymer_agent.md
Run:   python topology/11_prepolymer_agent.py
"""

import molpy as mp
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas
from molpy.core import fields

from eo_kit import ETHER, eo_builder, full_library, report


def main() -> None:
    lib = full_library()
    chain = eo_builder().build_linear("EO", 5)
    agent = lib["X4"].copy()

    world = Replicas(chain).times(4, spacing=8.0)
    for i in range(2):
        a = agent.copy()
        a.move([i * 3.0, 2.0, 0.0], entity_type=mp.Atom)
        for atom in a.atoms:
            atom[fields.MOL_ID] = 100 + i
        world.merge(a)

    n0 = world.n_atoms
    cured = GraphAssembler(mp.Reaction(ETHER)).assemble(
        world,
        ExhaustiveSelector(cutoff=10.0, exclude_same_molecule=True),
    )
    print(f"prepolymer+agent: {n0} → {cured.n_atoms} atoms")
    print(f"  Δatoms (leaving groups removed): {n0 - cured.n_atoms}")
    report("cured", cured)


if __name__ == "__main__":
    main()
