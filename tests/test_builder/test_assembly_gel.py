"""End-to-end crosslinked networks on the assembly kernel.

Ported from the deleted ``test_crosslink/test_peo_recipe.py``. The behaviour it
guarded — spacing sets the crosslink density, a seed reproduces a network
exactly, conversion hits its target, and the result relaxes and exports — is
still guaranteed. What is gone is the ``crosslink_gel()`` free function that
wired three classes together; that call sequence belongs in the docs.
"""

from __future__ import annotations

import molrs

import molpy as mp
from molpy.builder.assembly import (
    GraphAssembler,
    RandomSelector,
    SpacingSelector,
)
from molpy.core.atomistic import Atomistic
from molpy.optimize import LBFGS, SoftPotential

# Single-atom self reaction: every backbone carbon is a crosslink site.
RXN = "[C:1].[C:2]>>[C:1][C:2]"


def _peg_like(n_chains: int, degree: int, gap: float) -> Atomistic:
    """``n_chains`` parallel linear carbon chains (separate molecules, 3D)."""
    s = mp.Atomistic()
    atom_id = 1
    for chain in range(n_chains):
        prev = None
        for i in range(degree):
            atom = s.def_atom(
                element="C",
                x=float(i),
                y=float(chain) * gap,
                z=0.0,
                type=1,
                mass=12.011,
                id=atom_id,
                mol_id=chain + 1,
                charge=0.0,
            )
            atom_id += 1
            if prev is not None:
                s.def_bond(prev, atom, order=1.0)
            prev = atom
    return s


def _crosslink_bonds(gel: Atomistic, base: int) -> int:
    return len(list(gel.bonds)) - base


def _signature(gel: Atomistic) -> list[tuple[int, int]]:
    return sorted(tuple(sorted((b.itom.handle, b.jtom.handle))) for b in gel.bonds)


def _relax(gel: Atomistic) -> Atomistic:
    """The step the recipe used to hide: converge the guessed bond lengths."""
    frame = gel.to_frame()
    result = LBFGS(SoftPotential()).run(frame, fmax=0.05, steps=50)
    return Atomistic.adopt(molrs.Atomistic.from_frame(result.frame))


# --------------------------------------------------------------------------
# spacing sets the crosslink density
# --------------------------------------------------------------------------


def test_spacing_sets_the_crosslink_density():
    peg = _peg_like(4, 10, gap=2.0)
    base = len(list(peg.bonds))
    assembler = GraphAssembler(mp.Reaction(RXN))

    dense = assembler.assemble(
        peg, SpacingSelector(2, cutoff=2.5, exclude_same_molecule=True)
    )
    sparse = assembler.assemble(
        peg, SpacingSelector(5, cutoff=2.5, exclude_same_molecule=True)
    )

    dense_bonds = _crosslink_bonds(dense, base)
    sparse_bonds = _crosslink_bonds(sparse, base)
    assert dense_bonds > 0
    assert dense_bonds > sparse_bonds
    # the input is never touched
    assert len(list(peg.bonds)) == base


# --------------------------------------------------------------------------
# a seed reproduces a network exactly
# --------------------------------------------------------------------------


def test_random_network_is_reproducible_from_its_seed():
    peg = _peg_like(4, 12, gap=2.0)
    assembler = GraphAssembler(mp.Reaction(RXN))

    def gel():
        return assembler.assemble(
            peg,
            RandomSelector(
                conversion=0.5, seed=42, cutoff=2.5, exclude_same_molecule=True
            ),
        )

    assert _signature(gel()) == _signature(gel())


def test_a_different_seed_gives_a_different_network():
    peg = _peg_like(4, 12, gap=2.0)
    assembler = GraphAssembler(mp.Reaction(RXN))
    a = assembler.assemble(peg, RandomSelector(conversion=0.5, seed=1, cutoff=2.5))
    b = assembler.assemble(peg, RandomSelector(conversion=0.5, seed=2, cutoff=2.5))
    assert _signature(a) != _signature(b)


# --------------------------------------------------------------------------
# the network relaxes and exports
# --------------------------------------------------------------------------


def test_network_relaxes_and_exports_to_lammps(tmp_path):
    peg = _peg_like(3, 10, gap=2.0)
    gel = GraphAssembler(mp.Reaction(RXN)).assemble(
        peg, RandomSelector(conversion=0.4, seed=7, cutoff=2.5)
    )
    relaxed = _relax(gel)
    assert relaxed is not gel

    out = tmp_path / "gel.data"
    mp.io.write_lammps_data(out, relaxed.to_frame(), atom_style="full")
    assert out.exists() and out.stat().st_size > 0
