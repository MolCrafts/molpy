"""``molrs.ff.typifier.ElementTypifier`` FFI seam: labels cross to the frame,
and the class has one public home. The labelling rule is proven by the Rust
suite.
"""

from __future__ import annotations

import molrs
import molrs.ff
import molrs.ff.typifier


def _water() -> molrs.core.Atomistic:
    mol = molrs.core.Atomistic()
    o = mol.add_atom(atomic_number=8, x=0.0, y=0.0, z=0.0)
    h1 = mol.add_atom(atomic_number=1, x=0.96, y=0.0, z=0.0)
    h2 = mol.add_atom(atomic_number=1, x=-0.24, y=0.93, z=0.0)
    mol.add_bond(o, h1)
    mol.add_bond(o, h2)
    return mol


def test_element_typifier_labels_atoms_and_bonds_by_element() -> None:
    typed = molrs.ff.typifier.ElementTypifier().typify(_water())

    frame = typed.to_frame()
    assert list(frame["atoms"]["type"]) == ["O", "H", "H"]
    assert list(frame["bonds"]["type"]) == ["H-O", "H-O"]


def test_element_typifier_lives_only_in_the_typifier_module() -> None:
    assert isinstance(molrs.ff.typifier.ElementTypifier(), molrs.ff.typifier.Typifier)
    assert not hasattr(molrs.ff, "ElementTypifier")
