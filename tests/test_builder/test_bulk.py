import numpy as np

from molpy.builder.bulk import FCCBuilder, BCCBuilder
from molpy.core.region import BoxRegion
from molpy.core.atomistic import AtomicStructure


def simple_atom():
    mol = AtomicStructure()
    mol.def_atom(name="X", element="X", xyz=[0.0, 0.0, 0.0])
    return mol


def test_fcc_unit_cell():
    a = 1.0
    region = BoxRegion([a, a, a], [0, 0, 0])  # 精确原胞
    builder = FCCBuilder(a)
    struct = builder.fill_basis(region, simple_atom())
    assert len(struct.atoms) == 4  # FCC原胞应有4个原子
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
    region = BoxRegion([a, a, a], [0, 0, 0])  # 精确原胞
    builder = BCCBuilder(a)
    struct = builder.fill_basis(region, simple_atom())
    assert len(struct.atoms) == 2  # BCC原胞应有2个原子
    coords = np.array([atom.xyz for atom in struct.atoms])
    expected = np.array([[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]])
    for e in expected:
        assert any(np.allclose(coords[i], e) for i in range(len(coords)))


def test_fcc_cell_fill():
    a = 1.0
    region = BoxRegion([a - 0.1, a - 0.1, a - 0.1], [0, 0, 0])  # 稍小region，避免边界误差
    builder = FCCBuilder(a)
    struct = builder.fill_lattice(region, simple_atom())
    # 只在unit cell原点填充template
    assert len(struct.atoms) == 1
    coords = np.array([atom.xyz for atom in struct.atoms])
    expected = np.array([[0, 0, 0]])
    for e in expected:
        assert any(np.allclose(coords[i], e) for i in range(len(coords)))


def test_bcc_cell_fill():
    a = 1.0
    region = BoxRegion([a - 0.1, a - 0.1, a - 0.1], [0, 0, 0])  # 稍小region，避免边界误差
    builder = BCCBuilder(a)
    struct = builder.fill_lattice(region, simple_atom())
    assert len(struct.atoms) == 1
    coords = np.array([atom.xyz for atom in struct.atoms])
    expected = np.array([[0, 0, 0]])
    for e in expected:
        assert any(np.allclose(coords[i], e) for i in range(len(coords)))
