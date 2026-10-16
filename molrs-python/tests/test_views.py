"""The graph classes and their live views are the native classes, once."""

import copy
import pickle

import molrs
import numpy as np
import pytest
from molrs import _native

VIEW_CLASSES = (
    "NodeRef",
    "Atom",
    "VirtualSite",
    "DrudeParticle",
    "MasslessSite",
    "Bead",
    "RelationRef",
    "Bond",
    "Angle",
    "Dihedral",
    "Improper",
    "Port",
    "CgBond",
    "Refs",
)


class _SubAtomistic(molrs.core.Atomistic):
    """Module level, so pickle can find it."""


class _SubCoarseGrain(molrs.core.CoarseGrain):
    """Module level, so pickle can find it."""


def _roundtrip(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _ethanol_skeleton() -> tuple[molrs.core.Atomistic, list[molrs.core.Atom]]:
    graph = molrs.core.Atomistic()
    atoms = [
        graph.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0),
        graph.add_atom(atomic_number=6, x=1.5, y=0.0, z=0.0),
        graph.add_atom(atomic_number=8, x=2.0, y=1.4, z=0.0),
    ]
    graph.add_bond(atoms[0], atoms[1])
    graph.add_bond(atoms[1], atoms[2])
    return graph, atoms


class TestOneClass:
    def test_graph_classes_are_the_native_classes(self) -> None:
        assert molrs.core.Atomistic is _native.Atomistic
        assert molrs.core.CoarseGrain is _native.CoarseGrain
        assert molrs.core.MolGraph is _native.MolGraph

    @pytest.mark.parametrize("name", VIEW_CLASSES)
    def test_view_classes_are_the_native_classes(self, name: str) -> None:
        assert getattr(molrs.core, name) is getattr(_native, name)

    @pytest.mark.parametrize("cls", [molrs.core.Atomistic, molrs.core.CoarseGrain])
    def test_leaf_graph_classes_can_be_subclassed(self, cls: type) -> None:
        """Core data classes are extensible; a subclass is still the base."""
        sub = type("Sub", (cls,), {})
        assert isinstance(sub(), cls)

    @pytest.mark.parametrize("cls", [_SubAtomistic, _SubCoarseGrain])
    def test_leaf_graph_subclasses_pickle_as_themselves(self, cls: type) -> None:
        graph = cls(name="probe")
        graph.tag = "kept"
        back = pickle.loads(pickle.dumps(graph))
        assert type(back) is cls
        assert back.tag == "kept"
        assert back.props["name"] == "probe"

    def test_there_is_no_python_view_module(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            __import__("molrs.views")

    def test_graph_out_paths_return_the_one_class(self) -> None:
        from_smiles = molrs.io.smiles.SmilesIr.parse("CO").to_atomistic()
        for graph in (
            from_smiles,
            copy.copy(from_smiles),
            copy.deepcopy(from_smiles),
            molrs.core.Atomistic.from_frame(from_smiles.to_frame()),
        ):
            assert type(graph) is molrs.core.Atomistic
            assert len(graph.atoms) == 2
        assert type(molrs.perceive.add_hydrogens(from_smiles)) is molrs.core.Atomistic


class TestProps:
    def test_constructor_keywords_are_the_props(self) -> None:
        assert molrs.core.Atomistic(name="water").props == {"name": "water"}
        assert molrs.core.CoarseGrain(name="lipid").props == {"name": "lipid"}
        assert molrs.core.Atomistic().props == {}

    def test_copy_keeps_props_in_an_independent_dict(self) -> None:
        graph = molrs.core.Atomistic(label="x")
        copied = copy.copy(graph)
        assert copied.props == {"label": "x"}
        copied.props["label"] = "y"
        assert graph.props == {"label": "x"}

    def test_perception_output_keeps_props(self) -> None:
        graph = molrs.io.smiles.SmilesIr.parse("CO").to_atomistic()
        graph.props["label"] = "methanol"
        perceived = molrs.perceive.add_hydrogens(graph)
        assert perceived.props == {"label": "methanol"}

    def test_coarse_grain_copy_keeps_props(self) -> None:
        assert copy.copy(molrs.core.CoarseGrain(label="cg")).props == {"label": "cg"}

    def test_pickle_keeps_props(self) -> None:
        assert _roundtrip(molrs.core.Atomistic(label="x")).props == {"label": "x"}


class TestBeadMembership:
    def test_bead_members_are_the_source_atom_views(self) -> None:
        source, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        bead = cg.add_bead(bead_type="CC")
        cg.set_bead_members(bead, (atoms[0], atoms[1]))
        assert cg.bead_members(bead) == (atoms[0], atoms[1])
        assert all(atom.world is source for atom in cg.bead_members(bead))
        assert cg.beads_of_atom(atoms[0]) == [bead]
        assert cg.beads_of_atom(atoms[2]) == []
        assert cg.bead_members(cg.add_bead(bead_type="empty")) == ()
        # A membership is no field of the bead.
        assert "atoms" not in bead

    def test_set_bead_members_replaces_and_clears(self) -> None:
        _, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        bead = cg.add_bead(bead_type="CC")
        cg.set_bead_members(bead, atoms[:2])
        cg.set_bead_members(bead, [atoms[2]])
        assert cg.bead_members(bead) == (atoms[2],)
        cg.set_bead_members(bead, [])
        assert cg.bead_members(bead) == ()

    def test_copy_keeps_bead_membership(self) -> None:
        _, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        cg.set_bead_members(cg.add_bead(bead_type="CC"), (atoms[0], atoms[1]))
        copied = copy.copy(cg)
        assert copied.bead_members(copied.beads[0]) == (atoms[0], atoms[1])

    def test_pickle_keeps_bead_membership(self) -> None:
        source, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        cg.set_bead_members(cg.add_bead(bead_type="CO"), (atoms[1], atoms[2]))
        restored_source, restored_cg = _roundtrip((source, cg))
        members = restored_cg.bead_members(restored_cg.beads[0])
        assert members == (restored_source.atoms[1], restored_source.atoms[2])

    def test_membership_atoms_come_from_one_world(self) -> None:
        _, left = _ethanol_skeleton()
        _, right = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        bead = cg.add_bead(bead_type="X")
        with pytest.raises(ValueError, match="same source graph"):
            cg.set_bead_members(bead, (left[0], right[0]))
        with pytest.raises(ValueError, match="another graph"):
            cg.beads_of_atom(right[0])

    def test_memberships_read_from_a_frame_have_no_member_world(self) -> None:
        _, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        cg.set_bead_members(cg.add_bead(bead_type="CC", mol_id=1), atoms[:2])
        back = molrs.core.CoarseGrain.from_frame(cg.to_frame())
        with pytest.raises(ValueError, match="set_bead_members"):
            back.bead_members(back.beads[0])


class TestFactories:
    def test_factories_return_interned_live_refs(self) -> None:
        graph = molrs.core.Atomistic()
        carbon = graph.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
        oxygen = graph.add_atom(atomic_number=8, x=1.0, y=0.0, z=0.0)
        bond = graph.add_bond(carbon, oxygen, order=2.0)

        assert type(carbon) is molrs.core.Atom
        assert type(bond) is molrs.core.Bond
        assert graph.atoms[0] is carbon
        assert graph.bonds[0] is bond
        assert bond.itom is carbon
        assert bond.jtom is oxygen
        assert bond.endpoints == (carbon, oxygen)
        assert carbon["x", "y", "z"] == [0.0, 0.0, 0.0]
        assert bond["order"] == 2.0

    def test_add_bond_stamps_both_bond_facts(self) -> None:
        graph = molrs.core.Atomistic()
        a = graph.add_atom(atomic_number=6)
        b = graph.add_atom(atomic_number=6)
        bond = graph.add_bond(a, b)
        assert bond["bond_type"] == 1
        assert bond["bond_number"] == 1

    def test_add_bond_keeps_a_stated_class_as_given(self) -> None:
        graph = molrs.core.Atomistic()
        a = graph.add_atom(atomic_number=6)
        b = graph.add_atom(atomic_number=8)
        bond = graph.add_bond(a, b, bond_type=2, bond_number=2)
        assert bond["bond_type"] == 2
        assert bond["bond_number"] == 2

    def test_higher_relations_are_the_native_kinds(self) -> None:
        graph, (a, b, c) = _ethanol_skeleton()
        d = graph.add_atom(atomic_number=1)
        angle = graph.add_angle(a, b, c, theta0=1.9)
        dihedral = graph.add_dihedral(a, b, c, d)
        improper = graph.add_improper(b, a, c, d)

        assert (type(angle), type(dihedral), type(improper)) == (
            molrs.core.Angle,
            molrs.core.Dihedral,
            molrs.core.Improper,
        )
        assert graph.angles[0] is angle
        assert angle.endpoints == (a, b, c)
        assert angle["theta0"] == 1.9
        assert (graph.n_angles, graph.n_dihedrals, graph.n_impropers) == (1, 1, 1)

    def test_refs_have_no_detached_constructor(self) -> None:
        with pytest.raises(TypeError):
            molrs.core.Atom(atomic_number=6)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            molrs.core.Bond(object(), object())  # type: ignore[call-arg]

    def test_cross_world_relation_is_rejected(self) -> None:
        left = molrs.core.Atomistic()
        right = molrs.core.Atomistic()
        a = left.add_atom(atomic_number=6)
        b = right.add_atom(atomic_number=6)
        with pytest.raises(ValueError, match="another graph"):
            left.add_bond(a, b)
        with pytest.raises(ValueError, match="another graph"):
            left.add_angle(a, b, a)
        assert left.n_bonds == 0
        assert left.n_angles == 0

    def test_virtual_site_class_follows_its_stored_kind(self) -> None:
        graph = molrs.core.Atomistic()
        graph.add_atom(atomic_number=8)
        graph.add_atom(kind=molrs.core.DrudeParticle, atomic_number=0, charge=-1.0)
        graph.add_atom(kind=molrs.core.MasslessSite, atomic_number=0)
        graph.add_atom(kind=molrs.core.VirtualSite, atomic_number=0)

        # The views are dropped; re-interning reads the stored ``vsite``.
        assert [type(atom) for atom in graph.atoms] == [
            molrs.core.Atom,
            molrs.core.DrudeParticle,
            molrs.core.MasslessSite,
            molrs.core.VirtualSite,
        ]
        assert [atom.get("vsite") for atom in graph.atoms] == [
            None,
            "drude",
            "massless",
            "virtual",
        ]

    def test_virtual_site_kind_is_given_once(self) -> None:
        graph = molrs.core.Atomistic()
        with pytest.raises(ValueError, match="not both"):
            graph.add_atom(kind=molrs.core.DrudeParticle, atomic_number=0, vsite="drude")
        with pytest.raises(TypeError, match="kind must be"):
            graph.add_atom(kind=molrs.core.Atom, atomic_number=0)
        assert len(graph.atoms) == 0

    def test_add_port_returns_the_port_view(self) -> None:
        graph = molrs.core.Atomistic()
        anchor = graph.add_atom(atomic_number=8)
        handle = graph.add_atom(atomic_number=1)
        graph.add_bond(anchor, handle)
        port = graph.add_port(anchor, handle, "$")
        assert type(port) is molrs.core.Port
        assert graph.ports[0] is port
        assert port.anchor is anchor
        assert port.handle_atom is handle
        assert port["port_kind"] == "$"
        assert graph.port_on(anchor, handle) is port
        with pytest.raises(ValueError, match="fixed"):
            port["port_label"] = "A"

    def test_coarse_grain_factories(self) -> None:
        cg = molrs.core.CoarseGrain()
        a = cg.add_bead(bead_type="P1", x=0.0, y=0.0, z=0.0)
        b = cg.add_bead(bead_type="P1", x=1.0, y=0.0, z=0.0)
        bond = cg.add_bond(a, b, order=1.0)
        assert type(a) is molrs.core.Bead
        assert type(bond) is molrs.core.CgBond
        assert cg.beads[1] is b
        assert cg.bonds[0] is bond
        assert bond.endpoints == (a, b)
        assert bond["order"] == 1.0


class TestRemoval:
    def test_del_atom_cascades_and_leaves_a_stale_view(self) -> None:
        graph, (a, _, _) = _ethanol_skeleton()
        graph.del_atom(a)
        assert len(graph.atoms) == 2
        assert graph.n_bonds == 1
        # The view outlives its atom; a read through it raises.
        with pytest.raises(ValueError, match="stale"):
            _ = a["atomic_number"]

    def test_del_atoms_is_atomic(self) -> None:
        graph, atoms = _ethanol_skeleton()
        graph.del_atom(atoms[0])
        with pytest.raises(ValueError):
            graph.del_atoms([atoms[1], atoms[0]])  # atoms[0] is gone
        assert len(graph.atoms) == 2
        graph.del_atoms(graph.atoms)
        assert graph.n_atoms == 0 and graph.n_bonds == 0

    @pytest.mark.parametrize("cls", ["Atomistic", "CoarseGrain"])
    def test_del_bond_and_del_bonds(self, cls: str) -> None:
        graph = getattr(molrs.core, cls)()
        if cls == "Atomistic":
            a, b, c = (graph.add_atom(atomic_number=6) for _ in range(3))
        else:
            a, b, c = (graph.add_bead(bead_type="P") for _ in range(3))
        first = graph.add_bond(a, b)
        graph.add_bond(b, c)
        graph.del_bond(first)
        assert graph.n_bonds == 1
        graph.del_bonds(graph.bonds)
        assert graph.n_bonds == 0

    def test_del_bond_refuses_a_foreign_or_mistyped_relation(self) -> None:
        graph, (a, b, c) = _ethanol_skeleton()
        other, _ = _ethanol_skeleton()
        with pytest.raises(ValueError, match="another graph"):
            graph.del_bond(other.bonds[0])
        angle = graph.add_angle(a, b, c)
        with pytest.raises(ValueError, match="'bonds'"):
            graph.del_bond(angle)
        graph.del_angle(angle)
        assert graph.n_angles == 0 and graph.n_bonds == 2


class TestRefs:
    def test_len_index_slice_and_membership(self) -> None:
        graph, atoms = _ethanol_skeleton()
        refs = graph.atoms
        assert len(refs) == 3
        assert refs[-1] is atoms[2]
        assert list(refs[1:]) == atoms[1:]
        assert atoms[0] in refs
        assert graph.bonds[0] not in refs
        assert list(refs) == atoms

    def test_field_reads_are_columns(self) -> None:
        graph, _ = _ethanol_skeleton()
        np.testing.assert_array_equal(graph.atoms["x"], [0.0, 1.5, 2.0])
        np.testing.assert_array_equal(
            graph.atoms["x", "y"], [[0.0, 0.0], [1.5, 0.0], [2.0, 1.4]]
        )
        assert graph.atoms["atomic_number"].tolist() == [6, 6, 8]
        assert graph.bonds["bond_type"].tolist() == [1, 1]

    def test_terms_are_plain_relation_views_of_their_block(self) -> None:
        graph, (a, b, c) = _ethanol_skeleton()
        constraint = graph.add_term("constraints", [a, b], r0=1.5)
        assert type(constraint) is molrs.core.RelationRef
        assert constraint.kind == "constraints"
        assert constraint.endpoints == (a, b)
        assert list(graph.terms("constraints")) == [constraint]
        assert graph.n_terms("constraints") == 1
        assert graph.term_blocks() == ["constraints"]
        constraint["r0"] = 1.6
        assert graph.terms("constraints")["r0"].tolist() == [1.6]
        del constraint["r0"]
        assert "r0" not in constraint
        second = graph.add_term("constraints", [b, c])
        graph.del_term(constraint)
        assert list(graph.terms("constraints")) == [second]
        graph.del_terms(graph.terms("constraints"))
        assert graph.n_terms("constraints") == 0

    def test_a_typed_block_is_no_term_block(self) -> None:
        graph, (a, b, _) = _ethanol_skeleton()
        with pytest.raises(ValueError, match="typed methods"):
            graph.add_term("bonds", [a, b])
        with pytest.raises(ValueError, match="typed methods"):
            graph.terms("angles")

    def test_an_empty_refs_reads_as_empty_columns(self) -> None:
        graph, _ = _ethanol_skeleton()
        empty = graph.impropers
        assert len(empty) == 0
        assert empty["type"].shape == (0,)
        assert "impropers" in repr(empty)


class TestRefMapping:
    def test_node_ref_is_a_mapping_over_its_fields(self) -> None:
        graph = molrs.core.Atomistic()
        atom = graph.add_atom(atomic_number=6, charge=0.5)
        assert dict(atom) == {"atomic_number": 6, "charge": 0.5}
        assert set(atom.keys()) == {"atomic_number", "charge"}
        assert len(atom) == 2
        assert "charge" in atom and "mass" not in atom
        assert atom.get("mass", 12.0) == 12.0
        atom.update(mass=12.011)
        assert atom["mass"] == 12.011
        with pytest.raises(TypeError, match="del view"):
            atom["charge"] = None
        assert atom["charge"] == 0.5
        del atom["charge"]
        assert "charge" not in atom
        del atom["mass"]
        assert dict(atom.items()) == {"atomic_number": 6}
        with pytest.raises(KeyError):
            _ = atom["mass"]

    def test_tuple_keys_read_and_write_several_fields(self) -> None:
        graph = molrs.core.Atomistic()
        atom = graph.add_atom(atomic_number=6)
        atom["x", "y", "z"] = (1.0, 2.0, 3.0)
        assert atom["x", "y", "z"] == [1.0, 2.0, 3.0]
        assert ("x", "y", "z") in atom
        with pytest.raises(ValueError, match="needs 3 values"):
            atom["x", "y", "z"] = (1.0, 2.0)


class TestSpeciesInvariant:
    """Every door that writes an atom keeps a valid ``atomic_number``."""

    def test_an_atom_without_an_atomic_number_is_refused(self) -> None:
        graph = molrs.core.Atomistic()
        with pytest.raises(ValueError, match="atomic_number"):
            graph.add_atom(element="C", x=0.0)
        with pytest.raises(ValueError, match="atomic_number"):
            graph.add_atom(atomic_number=119)
        assert len(graph.atoms) == 0

    def test_a_non_scalar_value_is_refused_before_any_write(self) -> None:
        graph = molrs.core.Atomistic()
        with pytest.raises(TypeError):
            graph.add_atom(atomic_number=6, charge=None)  # type: ignore[arg-type]
        assert len(graph.atoms) == 0

    def test_a_view_cannot_break_the_atomic_number(self) -> None:
        graph = molrs.core.Atomistic()
        atom = graph.add_atom(atomic_number=6)
        with pytest.raises(ValueError):
            atom["atomic_number"] = 200
        with pytest.raises(ValueError):
            del atom["atomic_number"]
        with pytest.raises(TypeError):
            atom["atomic_number"] = None
        with pytest.raises(ValueError):
            atom["atomic_number"] = "C"
        assert atom["atomic_number"] == 6

    def test_a_bond_class_is_written_only_by_set_bond_class(self) -> None:
        graph, _ = _ethanol_skeleton()
        bond = graph.bonds[0]
        with pytest.raises(ValueError, match="set_bond_class"):
            bond["bond_type"] = 2
        with pytest.raises(ValueError):
            del bond["bond_number"]
        graph.set_bond_class(bond, molrs.core.BondOrder.Double, molrs.core.BondNumber.Double)
        assert bond["bond_type", "bond_number"] == [2, 2]

    def test_a_bead_keeps_a_string_bead_type(self) -> None:
        cg = molrs.core.CoarseGrain()
        with pytest.raises(ValueError, match="bead_type"):
            cg.add_bead(x=0.0)
        bead = cg.add_bead(bead_type="W")
        with pytest.raises(ValueError, match="bead_type"):
            bead["bead_type"] = 3
        with pytest.raises(ValueError, match="bead_type"):
            del bead["bead_type"]
        assert bead["bead_type"] == "W"
        assert len(cg.beads) == 1

    def test_an_add_refused_part_way_leaves_the_graph_unchanged(self) -> None:
        """The core writes a bag key by key; a value that contradicts an
        existing column's type is refused after the node or relation exists,
        and the binding removes it again."""
        graph = molrs.core.Atomistic()
        a = graph.add_atom(atomic_number=6, label="a")
        b = graph.add_atom(atomic_number=6, label="b")
        graph.add_bond(a, b, tag="single")
        graph.add_angle(a, b, a, tag="bent")
        with pytest.raises(ValueError):
            graph.add_atom(atomic_number=8, label=3)
        with pytest.raises(ValueError):
            graph.add_bond(a, b, tag=1)
        with pytest.raises(ValueError):
            graph.add_angle(a, b, a, tag=1.5)
        assert len(graph.atoms) == 2
        assert len(graph.bonds) == 1
        assert len(graph.angles) == 1
        assert graph.atoms["label"].tolist() == ["a", "b"]


class TestPickle:
    def test_graph_and_views_pickle_as_one_object_graph(self) -> None:
        graph, atoms = _ethanol_skeleton()
        graph.del_atom(graph.add_atom(atomic_number=1))  # a hole in the handles
        drude = graph.add_atom(kind=molrs.core.DrudeParticle, atomic_number=0)
        refs = [*atoms, drude, graph.bonds[1]]
        restored, restored_refs, restored_atoms = _roundtrip((graph, refs, graph.atoms))
        assert [type(ref) for ref in restored_refs] == [type(ref) for ref in refs]
        assert all(ref.world is restored for ref in restored_refs)
        assert restored_refs[:4] == list(restored.atoms)
        assert restored_refs[4] is restored.bonds[1]
        assert list(restored_atoms) == list(restored.atoms)
        assert restored_refs[4].endpoints == (restored_refs[1], restored_refs[2])
