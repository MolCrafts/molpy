"""A three-atom water with a two-type force field every emitter can write."""

import pytest

import molpy as mp


@pytest.fixture
def water() -> mp.Atomistic:
    mol = mp.Atomistic()
    o = mol.def_atom(
        name="O",
        element="O",
        type="OW",
        charge=-0.834,
        mass=15.999,
        x=0.0,
        y=0.0,
        z=0.0,
        mol_id=1,
    )
    h1 = mol.def_atom(
        name="H1",
        element="H",
        type="HW",
        charge=0.417,
        mass=1.008,
        x=0.96,
        y=0.0,
        z=0.0,
        mol_id=1,
    )
    h2 = mol.def_atom(
        name="H2",
        element="H",
        type="HW",
        charge=0.417,
        mass=1.008,
        x=-0.24,
        y=0.93,
        z=0.0,
        mol_id=1,
    )
    mol.def_bond(o, h1, type="OW-HW")
    mol.def_bond(o, h2, type="OW-HW")
    return mol


@pytest.fixture
def tip3p() -> mp.ForceField:
    ff = mp.ForceField(name="tip3p", units="real")
    atoms = ff.def_style("atom", "full")
    ow = atoms.def_type("OW", mass=15.999, charge=-0.834, element="O")
    hw = atoms.def_type("HW", mass=1.008, charge=0.417, element="H")
    ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=450.0, r0=0.9572)
    # No style-level cutoff: that is a LAMMPS pair_style argument the GROMACS
    # writer rightly refuses (it has no .top directive); cutoffs live in the mdp.
    pairs = ff.def_style("pair", "lj/cut")
    pairs.def_type("OW-OW", ow, ow, epsilon=0.1521, sigma=3.1507)
    pairs.def_type("HW-HW", hw, hw, epsilon=0.046, sigma=0.4)
    return ff
