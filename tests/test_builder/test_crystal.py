import pytest
from molpy.builder.crystal import CrystalLatticeBuilder, CrystalStructBuilder

@pytest.fixture
def lattice_builder():
    return CrystalLatticeBuilder()

def test_crystal_lattice_default(lattice_builder):
    # domyślnie nx=ny=nz=1, a=1.0
    sites = lattice_builder.create_sites()
    assert sites == [(0.0, 0.0, 0.0)]

@pytest.mark.parametrize(
    "nx, ny, nz, a, expected",
    [
        (2, 1, 1, 2.5, [(0.0, 0.0, 0.0), (2.5, 0.0, 0.0)]),
        (1, 2, 1, 1.0, [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
    ]
)
def test_crystal_lattice_custom(lattice_builder, nx, ny, nz, a, expected):
    sites = lattice_builder.create_sites(nx=nx, ny=ny, nz=nz, a=a)
    assert sorted(sites) == sorted(expected)

def test_crystal_struct_builder():
    coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    struct_builder = CrystalStructBuilder()
    atoms = struct_builder.populate(coords, element='Fe')

    # spodziewamy się listy słowników z kluczami 'type' i 'position'
    assert isinstance(atoms, list)
    assert all(isinstance(a, dict) for a in atoms)

    types = {a['type'] for a in atoms}
    positions = {a['position'] for a in atoms}

    assert types == {'Fe'}
    assert positions == {(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)}
