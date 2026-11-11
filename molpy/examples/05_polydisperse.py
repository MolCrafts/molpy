"""Polydisperse polymer system: sample a chain-size distribution.

A real melt is not monodisperse. A generative-BigSMILES string states the
whole ensemble — the repeat units and their weights, the end groups, the
chain-size distribution and the system total — and ``to_system()`` reads it
into the polymer model of ``mp.builder``:

    Fragment (ported template + port weights)
    -> StochasticObject (repeat units, end groups, ChainSize)
    -> Molecule -> System.sample(seed) -> a site graph + one record per chain

Each sampled chain is a site graph: ``Molecule.sample(seed)`` (or the whole
system's ``sample.sites``) is what ``mp.builder.Assembler(library,
mp.builder.GrowthPlacer())`` grows into atoms (see topology/02_block.py).

Guide: docs/user-guide/05_polydisperse_systems.md
Run:   python 05_polydisperse.py
"""

import numpy as np

import molpy as mp

# A styrene / methyl-acrylate copolymer, 80:20 by port weight, methyl caps,
# Schulz-Zimm sizes (Mw 1500, Mn 1400) and a 5e5 g/mol system — every size
# counted in heavy-atom mass, as generative BigSMILES writes it.
TEXT = (
    "{[] [<|8|]CC(c1ccccc1)[>|8|], [<|2|]CC(C(=O)OC)[>|2|]; C[>], [<]C []}"
    "|schulz_zimm(1500, 1400)|.|5e5|"
)


def main() -> None:
    system = mp.io.bigsmiles.read_bigsmiles(TEXT).to_system()

    # Deterministic in the seed.
    sample = system.sample(42)

    heavy = np.array([c.heavy_atom_mass for c in sample.molecules])
    Mn = float(heavy.mean())
    Mw = float((heavy**2).sum() / heavy.sum())
    print(f"sampled {len(sample.molecules)} chains for {system.targets()[0]:.0f} g/mol")
    print(f"  heavy-atom Mn={Mn:.0f}  Mw={Mw:.0f}  PDI={Mw / Mn:.3f}")

    # Every chain is a connected component of one site graph; a site's
    # bead_type names the fragment placed there.
    first = sample.molecules[0]
    print(
        f"  first chain: {first.n_repeat_units} repeat units, "
        f"{first.n_fragments} fragments, {first.mass:.1f} g/mol with hydrogens"
    )
    print(f"  fragments: {sorted(system.library())}")


if __name__ == "__main__":
    main()
