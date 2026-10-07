"""Shared ethylene-oxide kit for every topology example.

Guide: docs/user-guide/topology/index.md

Every unit is one CGsmiles fragment whose bonding descriptors are its ports:
``<`` joins ``>``, and a label (``<g`` / ``>g``) joins only the same label.
Every topology is a CGsmiles string too; ``to_coarsegrain()`` turns it into
the site graph that ``mp.builder.Assembler`` grows with ``mp.builder.GrowthPlacer`` into an
``mp.Atomistic``.
"""

from __future__ import annotations

import molpy as mp

UNITS = {
    "EO": "[<]OCC[>]",  # -O-CH2-CH2-
    "PO": "[<]OC(C)C[>]",  # -O-CH(CH3)-CH2-
    "CAPA": "C[>]",  # CH3- : starts a chain on its `<` end
    "CAPB": "[<]OC",  # -O-CH3 : ends a chain on its `>` end
    "X3": "C(C[>])(C[>])C[>]",  # three-arm core
    "BR": "[<]OCC(C[>g])[>]",  # backbone unit with a labelled graft port
    "GR": "[<g]OCC[>]",  # first graft unit
}


def library(*, seed: int = 42) -> dict[str, mp.Atomistic]:
    """Every unit of :data:`UNITS` as a 3D molecule with hydrogens and ports."""
    conformer = mp.conformer.Conformer(seed=seed)
    return {
        name: conformer.generate(
            mp.io.smiles.SmilesIR.from_fragment(body).to_template()
        )[0]
        for name, body in UNITS.items()
    }


def report(name: str, world: mp.Atomistic) -> None:
    """Atoms, bonds, units and the ports left open."""
    atoms = world.to_frame()["atoms"]
    n_units = len(set(atoms["frag_id"].tolist()))
    n_bonds = world.to_frame()["bonds"].nrows
    print(
        f"{name:14s} units={n_units:3d}  atoms={world.n_atoms:4d}  "
        f"bonds={n_bonds:4d}  open ports={world.n_ports}"
    )
