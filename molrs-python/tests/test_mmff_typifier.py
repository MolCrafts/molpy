"""Python-binding coverage for MMFF graph typification."""

import molrs


def _ethane() -> "molrs.core.Atomistic":
    mol = molrs.core.Atomistic()
    c1 = mol.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    c2 = mol.add_atom(atomic_number=6, x=1.54, y=0.0, z=0.0)
    hpos = [
        (c1, (-0.36, 1.03, 0.0)),
        (c1, (-0.36, -0.51, 0.89)),
        (c1, (-0.36, -0.51, -0.89)),
        (c2, (1.90, 1.03, 0.0)),
        (c2, (1.90, -0.51, 0.89)),
        (c2, (1.90, -0.51, -0.89)),
    ]
    for c, (x, y, z) in hpos:
        h = mol.add_atom(atomic_number=1, x=x, y=y, z=z)
        mol.add_bond(c, h)
    mol.add_bond(c1, c2)
    return mol


def test_mmff_typify_returns_typed_atomistic_topology():
    typifier = molrs.ff.typifier.Mmff94Typifier()
    typed = typifier.typify(_ethane())
    assert isinstance(typed, molrs.core.Atomistic)

    frame = typed.to_frame()
    assert frame["atoms"].n_rows == 8
    assert frame["bonds"].n_rows == 7
    assert frame["angles"].n_rows == 12
    assert frame["dihedrals"].n_rows == 9
    assert len(frame["atoms"]["charge"]) == 8
