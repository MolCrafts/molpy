import numpy as np

from molpy.builder.bulk import FCCBuilder, BCCBuilder
from molpy.core.region import BoxRegion
from molpy.core.struct import AtomicStructure


def simple_atom():
    mol = AtomicStructure()
    mol.def_atom(name="X", element="X", xyz=[0.0, 0.0, 0.0])
    return mol


def test_fcc_unit_cell():
    a = 1.0
    region = BoxRegion([a, a, a], [0, 0, 0])
    builder = FCCBuilder(a)
    struct = builder.build(region, simple_atom())
    assert len(struct.atoms) == 4
    coords = np.array([atom.xyz for atom in struct.atoms])
    expected = np.array(
        [
            [0, 0, 0],
            [0.5 * a, 0.5 * a, 0],
            [0.5 * a, 0, 0.5 * a],
            [0, 0.5 * a, 0.5 * a],
        ]
    )
    for e in expected:
        assert any(np.allclose(coords[i], e) for i in range(len(coords)))


def test_bcc_unit_cell():
    a = 1.0
    region = BoxRegion([a, a, a], [0, 0, 0])
    builder = BCCBuilder(a)
    struct = builder.build(region, simple_atom())
    assert len(struct.atoms) == 2
    coords = np.array([atom.xyz for atom in struct.atoms])
    expected = np.array([[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]])
    for e in expected:
        assert any(np.allclose(coords[i], e) for i in range(len(coords)))
