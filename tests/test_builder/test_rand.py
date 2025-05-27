import pytest
from molpy.builder.rand import RandomLatticeBuilder, RandomStructBuilder

def test_random_lattice_reproducible():
    rl = RandomLatticeBuilder()
    coords1 = rl.create_sites(n_steps=5, step_size=0.5, seed=123)
    coords2 = rl.create_sites(n_steps=5, step_size=0.5, seed=123)
    assert coords1 == coords2
    assert len(coords1) == 6  # 0 + 5 kroków

def test_random_struct_builder():
    sites = [(0,0,0),(1,1,1)]
    rs = RandomStructBuilder()
    mols = rs.populate(sites, monomer='X')
    assert len(mols) == 2
    assert all(isinstance(m, dict) for m in mols)
    assert mols[0]['monomer'] == 'X'
    assert mols[1]['position'] == (1,1,1)
