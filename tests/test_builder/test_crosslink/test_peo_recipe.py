"""End-to-end PEO-gel recipe tests (crosslink-03).

Build a linear multi-chain PEG-like structure, crosslink it (uniform ``spacing``
or random ``conversion``), relax, and export to LAMMPS — the recipe only
composes existing pieces. Relaxation is real: the recipe runs
:class:`molpy.optimize.LBFGS` on the crosslinked frame, defaulting to the
force-field-free :class:`~molpy.optimize.SoftPotential`.
"""

import molpy as mp
from molpy.builder.crosslink import (
    DeterministicCrosslinker,
    RandomCrosslinker,
    crosslink_gel,
    write_lammps,
)

# Single-atom self reaction: every backbone carbon is a crosslink site.
RXN = "[C:1].[C:2]>>[C:1][C:2]"


def _peg_like(n_chains, degree, gap):
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


def _crosslink_bonds(gel, base):
    return len(list(gel.bonds)) - base


def _signature(gel):
    return sorted(tuple(sorted((b.itom.handle, b.jtom.handle))) for b in gel.bonds)


# --------------------------------------------------------------------------
# ac-003 — uniform (spacing) PEO network recipe, runnable + exportable
# --------------------------------------------------------------------------


def test_uniform_network_recipe_runs_and_exports(tmp_path):
    peg = _peg_like(n_chains=4, degree=20, gap=2.0)
    base = len(list(peg.bonds))

    dense = crosslink_gel(
        peg,
        DeterministicCrosslinker(
            RXN, spacing=2, cutoff=2.5, exclude_same_molecule=True
        ),
    )
    sparse = crosslink_gel(
        peg,
        DeterministicCrosslinker(
            RXN, spacing=6, cutoff=2.5, exclude_same_molecule=True
        ),
    )

    dense_bonds = _crosslink_bonds(dense, base)
    sparse_bonds = _crosslink_bonds(sparse, base)
    assert dense_bonds > 0
    assert dense_bonds > sparse_bonds  # spacing sets crosslink density

    # Input untouched (immutable recipe).
    assert len(list(peg.bonds)) == base

    out = tmp_path / "peo_gel_uniform.data"
    write_lammps(dense, out)
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------
# ac-004 — random (conversion) PEO network recipe, reproducible + exportable
# --------------------------------------------------------------------------


def test_random_network_recipe_reproducible_and_exports(tmp_path):
    peg = _peg_like(n_chains=4, degree=20, gap=2.0)
    base = len(list(peg.bonds))

    def build():
        return crosslink_gel(
            peg,
            RandomCrosslinker(RXN, conversion=0.7, seed=42, exclude_same_molecule=True),
        )

    gel1 = build()
    gel2 = build()

    assert _signature(gel1) == _signature(gel2)  # same seed -> same network

    # limiting reactant = 80 carbons / 2 (A x A) = 40 sites; 0.7 -> ~28 bonds.
    assert abs(_crosslink_bonds(gel1, base) - 28) <= 2

    out = tmp_path / "peo_gel_random.data"
    write_lammps(gel1, out)
    assert out.exists() and out.stat().st_size > 0


def test_recipe_relaxes_via_lbfgs_soft_potential():
    """Default relax runs LBFGS+SoftPotential: coords move, topology preserved."""
    import numpy as np

    peg = _peg_like(n_chains=2, degree=10, gap=2.0)
    xlink = DeterministicCrosslinker(RXN, cutoff=2.5, exclude_same_molecule=True)

    unrelaxed = crosslink_gel(peg, xlink, relax=False)
    relaxed = crosslink_gel(peg, xlink, relax=True)

    # Relaxation returns a distinct, independent structure.
    assert relaxed is not unrelaxed
    # Same topology (relaxation only moves atoms).
    assert len(list(relaxed.atoms)) == len(list(unrelaxed.atoms))
    assert len(list(relaxed.bonds)) == len(list(unrelaxed.bonds))
    # Coordinates actually changed under relaxation.
    assert not np.allclose(relaxed.xyz, unrelaxed.xyz)
    # Input polymer untouched (immutable recipe).
    assert np.allclose(peg.xyz, _peg_like(n_chains=2, degree=10, gap=2.0).xyz)


def test_relaxed_network_exports(tmp_path):
    peg = _peg_like(n_chains=3, degree=12, gap=2.0)
    gel = crosslink_gel(
        peg,
        DeterministicCrosslinker(
            RXN, spacing=2, cutoff=2.5, exclude_same_molecule=True
        ),
        relax=True,
    )
    out = tmp_path / "peo_gel_relaxed.data"
    write_lammps(gel, out)
    assert out.exists() and out.stat().st_size > 0
