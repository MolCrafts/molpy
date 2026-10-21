"""Seam tests for the subclassable ``molrs.ff.typifier.Typifier`` base and ``TypeAssignment``.

The base owns one output force field per instance. For a Python subclass,
``typify`` copies the graph, calls the subclass ``assign`` on the copy, and hands
the returned ``TypeAssignment`` to the Rust ``TypeAssignment::apply_to``; for a native typifier
it runs ``Typing::typify``. The science of both paths (validation, stamping,
definition order) is proven by the Rust unit tests in ``ff/typifier/mod.rs``;
these tests only cover the binding seam: construction, the subclass path,
positional mapping across the boundary, and error mapping.
"""

from __future__ import annotations

import itertools

import molrs
import pytest
from molrs.core import Dihedral, Improper
from molrs.ff.typifier import TypeAssignment, Mmff94Typifier, Typifier

_SPECIAL_LJ = (0.0, 0.0, 0.5)
_SPECIAL_COUL = (0.0, 0.0, 0.75)


def _pair() -> molrs.core.Atomistic:
    """Two bonded carbon atoms, nothing typed."""
    mol = molrs.core.Atomistic()
    a = mol.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    b = mol.add_atom(atomic_number=6, x=1.54, y=0.0, z=0.0)
    mol.add_bond(a, b)
    return mol


def _chain_with_improper() -> molrs.core.Atomistic:
    """Five named atoms; two dihedrals with one improper created between them.

    The improper is created after the first dihedral and before the second, so
    a mapping that mixed impropers into the dihedral positions
    (``links.bucket(Dihedral)``) would disagree with the kind's own rows.
    """
    mol = molrs.core.Atomistic()
    atoms = [
        mol.add_atom(atomic_number=6, name=f"a{i}", x=1.5 * i, y=0.0, z=0.0)
        for i in range(5)
    ]
    for left, right in itertools.pairwise(atoms):
        mol.add_bond(left, right)
    mol.add_dihedral(atoms[0], atoms[1], atoms[2], atoms[3])
    mol.add_improper(atoms[1], atoms[0], atoms[2], atoms[4])
    mol.add_dihedral(atoms[1], atoms[2], atoms[3], atoms[4])
    return mol


def _ethane() -> molrs.core.Atomistic:
    """Ethane (C2H6) with explicit hydrogens, as the native typifiers expect."""
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


def _atom_rows(ff: molrs.ff.forcefield.ForceField) -> dict[str, dict]:
    """``{name: params}`` of the ``atom``/``full`` style of ``ff``."""
    return {t.name: t.params for t in ff.get_style("atom", "full").get_types()}


def _assert_declares_library_special_bonds(ff: molrs.ff.forcefield.ForceField) -> None:
    """``ff`` declares exactly ``_SPECIAL_LJ`` / ``_SPECIAL_COUL``: merging an
    equal declaration is accepted, a different one refused."""
    same = molrs.ff.forcefield.ForceField("same")
    same.set_special_bonds(list(_SPECIAL_LJ), list(_SPECIAL_COUL))
    assert ff.merge(same) is ff
    other = molrs.ff.forcefield.ForceField("other")
    other.set_special_bonds([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        ff.merge(other)


def _endpoint_label(link: Dihedral) -> str:
    return "-".join(str(atom["name"]) for atom in link.endpoints)


class _FirstAtomX(Typifier):
    """Types the first atom ``X`` (style ``full``) with ``mass``; the second gets nothing."""

    def __init__(self, mass: float = 1.0) -> None:
        self.mass = mass

    def assign(self, graph: molrs.core.Atomistic) -> TypeAssignment:
        return TypeAssignment(
            [{"type": ("full", "X", (), {"mass": self.mass})}, {}],
            styles=[("atom", "full", {})],
        )


class _DihedralTagger(Typifier):
    """Stamps each node and each dihedral with a label derived from the element itself."""

    def assign(self, graph: molrs.core.Atomistic) -> TypeAssignment:
        nodes = [{"seen": str(atom["name"])} for atom in graph.atoms]
        dihedrals = [{"tag": _endpoint_label(link)} for link in graph.dihedrals]
        return TypeAssignment(nodes, {Dihedral: dihedrals})


class _SpecialBondsLibrary(Typifier):
    """A stamp-only typifier whose library declares special_bonds."""

    def source_forcefield(self) -> molrs.ff.forcefield.ForceField:
        lib = molrs.ff.forcefield.ForceField("lib")
        lib.set_special_bonds(list(_SPECIAL_LJ), list(_SPECIAL_COUL))
        return lib

    def assign(self, graph: molrs.core.Atomistic) -> TypeAssignment:
        return TypeAssignment([{} for _ in graph.atoms])


def test_a_type_annotation_without_endpoints_raises_type_error() -> None:
    """A type annotation carries its endpoints: ``(style, name, params)`` is not
    a form — a name is never read for endpoints."""
    with pytest.raises(TypeError, match="endpoints"):
        TypeAssignment([{"type": ("harmonic", "C-C", {"k": 1.0})}])


class TestTypifierSubclass:
    def test_typify_stamps_a_new_graph_and_defines_the_type(self) -> None:
        mol = _pair()
        before = [dict(atom.items()) for atom in mol.atoms]
        typifier = _FirstAtomX()

        typed = typifier.typify(mol)

        assert isinstance(typed, molrs.core.Atomistic)
        assert typed is not mol
        first, second = typed.atoms[0], typed.atoms[1]
        assert first["type"] == "X"
        assert first["mass"] == 1.0
        assert "type" not in second
        assert "mass" not in second
        assert [dict(atom.items()) for atom in mol.atoms] == before
        assert _atom_rows(typifier.forcefield()) == {"X": {"mass": 1.0}}

    def test_conflicting_second_typify_raises_value_error(self) -> None:
        typifier = _FirstAtomX(mass=1.0)
        typifier.typify(_pair())
        typifier.mass = 2.0

        with pytest.raises(ValueError):
            typifier.typify(_pair())

        assert _atom_rows(typifier.forcefield()) == {"X": {"mass": 1.0}}

    def test_match_positions_follow_nodes_and_their_own_kind(self) -> None:
        typed = _DihedralTagger().typify(_chain_with_improper())

        for atom in typed.atoms:
            assert atom["seen"] == atom["name"]
        dihedrals = typed.dihedrals
        assert len(dihedrals) == 2
        for link in dihedrals:
            assert link["tag"] == _endpoint_label(link)
        impropers = typed.impropers
        assert len(impropers) == 1
        for link in impropers:
            assert "tag" not in link

    def test_library_special_bonds_reach_the_output(self) -> None:
        typifier = _SpecialBondsLibrary()

        typifier.typify(_pair())

        _assert_declares_library_special_bonds(typifier.forcefield())

    def test_forcefield_before_typify_is_the_seeded_empty_output(self) -> None:
        output = _SpecialBondsLibrary().forcefield()

        assert output.name == "lib"
        _assert_declares_library_special_bonds(output)
        assert output.get_styles(molrs.ff.forcefield.Style) == []

    def test_subclass_defining_typify_is_rejected_at_class_creation(self) -> None:
        with pytest.raises(TypeError):

            class _Overrides(Typifier):
                def typify(self, mol: molrs.core.Atomistic) -> molrs.core.Atomistic:
                    return mol


class TestTypifierClass:
    def test_base_typify_without_match_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Typifier().typify(_pair())

    def test_subclass_without_match_raises_not_implemented(self) -> None:
        class _NoMatch(Typifier):
            pass

        with pytest.raises(NotImplementedError):
            _NoMatch().typify(_pair())


class TestNativeTypifierSubclass:
    """A native typifier can be extended, but its hooks run in Rust."""

    def test_a_native_typifier_subclass_typifies_as_the_native(self) -> None:
        class _Tagged(Mmff94Typifier):
            tag = "mine"

        typed = _Tagged().typify(_ethane())
        reference = Mmff94Typifier().typify(_ethane())
        assert typed.to_frame()["atoms"]["type"].tolist() == (
            reference.to_frame()["atoms"]["type"].tolist()
        )

    @pytest.mark.parametrize("hook", ["assign", "source_forcefield"])
    def test_a_native_subclass_overriding_a_hook_is_rejected(self, hook: str) -> None:
        with pytest.raises(TypeError, match=hook):
            type("_Overrides", (Mmff94Typifier,), {hook: lambda self, *a: None})

    def test_the_native_typifiers_live_in_the_typifier_module(self) -> None:
        assert Mmff94Typifier.__module__ == "molrs.ff.typifier"


class TestNativeTypifierMatch:
    def test_mmff94_match_returns_match(self) -> None:
        assert isinstance(Mmff94Typifier().assign(_ethane()), TypeAssignment)
