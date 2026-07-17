"""Cross-layer polymer assembly workflows; intentionally not unit tests."""

from __future__ import annotations

import molrs
import pytest

import molpy as mp
from molpy.builder.assembly import (
    ExhaustiveSelector,
    GraphAssembler,
    MonomerLibrary,
    PolymerBuilder,
    RandomSelector,
    Replicas,
    SiteMap,
)
from molpy.core import fields
from molpy.core.atomistic import Atomistic
from molpy.optimize import LBFGS, SoftPotential

pytestmark = pytest.mark.integration

ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"
CARBON_LINK = "[C:1].[C:2]>>[C:1][C:2]"


def _eo() -> Atomistic:
    struct = mp.Atomistic()
    heavy = [
        struct.def_atom(element=element, x=float(index), y=0.0, z=0.0)
        for index, element in enumerate("OCCO")
    ]
    for left, right in zip(heavy, heavy[1:], strict=False):
        struct.def_bond(left, right)
    for oxygen in (heavy[0], heavy[-1]):
        struct.def_bond(
            oxygen,
            struct.def_atom(element="H", x=oxygen["x"], y=1.0, z=0.0),
        )
    SiteMap(struct).label_elements("O", "a", "b")
    return struct


def _carbon_chains(n_chains: int, degree: int, gap: float) -> Atomistic:
    world = mp.Atomistic()
    atom_id = 1
    for chain_index in range(n_chains):
        previous = None
        for index in range(degree):
            atom = world.def_atom(
                element="C",
                x=float(index),
                y=float(chain_index) * gap,
                z=0.0,
                type=1,
                mass=12.011,
                id=atom_id,
                mol_id=chain_index + 1,
                charge=0.0,
            )
            atom_id += 1
            if previous is not None:
                world.def_bond(previous, atom, order=1.0)
            previous = atom
    return world


class TestPolymerAssemblyWorkflow:
    def test_build_replicate_and_crosslink(self):
        strand = PolymerBuilder(
            MonomerLibrary({"EO": _eo()}), mp.Reaction(ETHER)
        ).build_linear("EO", 3)
        hydroxyls = [
            atom
            for atom in strand.atoms
            if atom.get(fields.ELEMENT) == "O"
            and any(
                neighbor.get(fields.ELEMENT) == "H"
                for neighbor in strand.get_neighbors(atom)
            )
        ]
        SiteMap(strand).every_nth(hydroxyls, 1, "x", leaving="h", fold_charge=False)
        melt = Replicas(strand).times(2, spacing=2.0)

        gel = GraphAssembler(
            mp.Reaction("[O;%x:1][H;%h].[O;%x:2][H;%h]>>[O:1][O:2]")
        ).assemble(
            melt,
            ExhaustiveSelector(cutoff=5.0, exclude_same_molecule=True),
        )

        assert gel.n_atoms < melt.n_atoms
        assert len(list(gel.bonds)) > len(list(melt.bonds)) - 4

    def test_crosslink_relax_and_export_lammps(self, tmp_path):
        precursor = _carbon_chains(3, 10, gap=2.0)
        gel = GraphAssembler(mp.Reaction(CARBON_LINK)).assemble(
            precursor,
            RandomSelector(conversion=0.4, seed=7, cutoff=2.5),
        )
        optimized = LBFGS(SoftPotential()).run(gel.to_frame(), fmax=0.05, steps=50)
        relaxed = Atomistic.adopt(molrs.Atomistic.from_frame(optimized.frame))

        # This workflow intentionally writes explicit topology. The toy soft
        # potential has no force-field assigner, so give every emitted relation
        # a valid local type id instead of asking the writer to omit it.
        for collection in (relaxed.bonds, relaxed.angles, relaxed.dihedrals):
            for relation in collection:
                relation[fields.TYPE] = 1

        output = tmp_path / "gel.data"
        mp.io.write_lammps_data(output, relaxed.to_frame(), atom_style="full")
        assert output.exists()
        assert output.stat().st_size > 0
