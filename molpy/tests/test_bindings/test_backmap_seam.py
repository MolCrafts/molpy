"""The operator's backmap script, run through every seam it crosses.

Regression example (trace-assembly-07, ac-008). This is the operator-ruled
exception to "no multi-stage chains under ``tests/``" (notes.md 2026-09-26):
one public-API script composing the backmap classes exactly as the binding
script ``backmap_pe_pma/backmap.py`` does, steps 1 (the tuple-key coordinate
write only) and 3-5,

    atoms["x", "y", "z"] = arr -> SubgraphMatcher.find -> Coarsener.coarsen
    -> Assembler(lib, SitePlacer(), AxisOrienter()).assemble(sites)
    -> ElementTypifier().typify(world).

The library ships only the classes; the composition lives here. Every number
asserted is a hand count on the fixtures below, not a re-derived geometry:
matching, coarsening, port assignment, placement, linking and typing are proven
by the Rust suites.

Fixtures, built by hand in process (no third-party software):

- a CG frame: an 8-bead head-to-head chain, bead types ``4 1 1 1 1 1 1 4``,
  bead ``i`` at ``(4 i, 2 (i mod 2), 0)`` Å (a zigzag, so each site axis
  crosses the chain), ``mass = 72.0`` g/mol, bonded ``(i, i + 1)``,
  plus one type-``2`` bead at ``x = 40`` Å with ``mass = 7.0`` g/mol and no
  bond; no ``mol_id`` column;
- the library: the ported monomer ``H0-C0-C1-H1`` (an ``Atomistic``, ports
  ``(C0, H0, ">")`` and ``(C1, H1, "<")``, hydrogens off the C-C line; C 12.011, H 1.008 g/mol) under
  ``"M"`` and a one-atom Li ``Atomistic`` (6.94 g/mol) under ``"Li"``.

Hand counts: ``1-1-1-4`` embeds in the chain once from each end (beads 0-3
and 4-7, disjoint) and ``2`` once, so 3 groups and 3 sites; the chain's
bond 3-4 joins the two chain sites, so the site graph is one 2-site chain and
one lone site. Two monomer copies are 8 atoms and 4
ports; their one link removes one hydrogen per joined port, leaving 6 atoms
and 2 ports; the Li adds 1 atom: 7 atoms, 2 ports, ``mol_id`` 1 (the chain)
and 2 (the Li), and every atom type is an element of C, H or Li.
"""

from __future__ import annotations

import molpy as mp
import numpy as np
from molpy.core import CoarseGrain
from molpy.builder import Assembler, AxisOrienter, SitePlacer
from molpy.ff.typifier import ElementTypifier
from molpy.io.cgsmiles import read_cgsmiles
from molpy.builder import Coarsener
from molpy.perceive import SubgraphMatcher

N_CHAIN = 8


def _cg_frame() -> mp.core.Frame:
    """The CG chain plus one lone ``2`` bead, coordinates still zero."""
    n = N_CHAIN + 1
    index = np.arange(N_CHAIN)
    return mp.core.Frame(
        {
            "atoms": {
                "type": np.array(["4", "1", "1", "1", "1", "1", "1", "4", "2"]),
                "x": np.zeros(n, dtype=np.float64),
                "y": np.zeros(n, dtype=np.float64),
                "z": np.zeros(n, dtype=np.float64),
                "mass": np.array([72.0] * N_CHAIN + [7.0], dtype=np.float64),
            },
            "bonds": {
                "atomi": index[:-1],
                "atomj": index[1:],
            },
        }
    )


def _coordinates() -> np.ndarray:
    """``(9, 3)``: chain bead ``i`` at ``(4 i, 2 (i mod 2), 0)`` Å, the lone
    bead at ``x = 40`` Å."""
    xyz = np.zeros((N_CHAIN + 1, 3), dtype=np.float64)
    xyz[:N_CHAIN, 0] = 4.0 * np.arange(N_CHAIN)
    xyz[:N_CHAIN, 1] = 2.0 * (np.arange(N_CHAIN) % 2)
    xyz[N_CHAIN, 0] = 40.0
    return xyz


def _ported_monomer() -> mp.core.Atomistic:
    """``H0-C0-C1-H1`` with ports ``(C0, H0, ">")`` and ``(C1, H1, "<")``.

    Kept beside the script, not imported from another test module, so this
    regression example runs on its own.
    """
    fragment = mp.core.Atomistic()
    c0 = fragment.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0, mass=12.011)
    c1 = fragment.add_atom(atomic_number=6, x=1.54, y=0.0, z=0.0, mass=12.011)
    h0 = fragment.add_atom(atomic_number=1, x=-0.5, y=0.9, z=0.0, mass=1.008)
    h1 = fragment.add_atom(atomic_number=1, x=2.04, y=0.9, z=0.0, mass=1.008)
    fragment.add_bond(c0, c1)
    fragment.add_bond(c0, h0)
    fragment.add_bond(c1, h1)
    fragment.add_port(c0, h0, ">")
    fragment.add_port(c1, h1, "<")
    return fragment


def _lithium() -> mp.core.Atomistic:
    mol = mp.core.Atomistic()
    mol.add_atom(atomic_number=3, x=0.0, y=0.0, z=0.0, mass=6.94)
    return mol


def test_operator_backmap_script_crosses_every_seam() -> None:
    # 1. CG input: coordinates written through the tuple key.
    frame = _cg_frame()
    atoms = frame["atoms"]
    atoms["x", "y", "z"] = _coordinates()
    cg = CoarseGrain.from_frame(frame)

    # 2. Library: name -> one molecule, with ports where it bonds.
    lib = {"M": _ported_monomer(), "Li": _lithium()}

    # 3. Rules: bead-group pattern -> molecule name; each group is one site.
    rules = {"{[#1][#1][#1][#4]}": "M", "{[#2]}": "Li"}
    groups: list[list[mp.core.Bead]] = []
    names: list[str] = []
    for pattern, name in rules.items():
        found = SubgraphMatcher(read_cgsmiles(pattern).to_coarsegrain()).find(cg)
        groups += found
        names += [name] * len(found)
    assert len(groups) == 3
    sites = Coarsener(cg).coarsen(groups, names)
    assert type(sites) is mp.core.CoarseGrain
    assert sites.n_beads == 3

    # 4. Assemble: each copy turned onto its site and placed on it; ports join
    #    the bonded sites.
    world = Assembler(lib, SitePlacer(), AxisOrienter()).assemble(sites)
    assert type(world) is mp.core.Atomistic
    assert world.n_atoms == 7
    assert world.n_ports == 2
    mol_id = np.asarray(world.to_frame()["atoms"]["mol_id"])
    assert set(mol_id.tolist()) == {1, 2}

    # 5. Element types for the writer.
    typed = ElementTypifier().typify(world)
    assert type(typed) is mp.core.Atomistic
    assert set(typed.to_frame()["atoms"]["type"]) <= {"C", "H", "Li"}
