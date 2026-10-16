"""The graph classes and their live views are the native classes, once."""

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
        graph.def_atom(element="C", x=0.0, y=0.0, z=0.0),
        graph.def_atom(element="C", x=1.5, y=0.0, z=0.0),
        graph.def_atom(element="O", x=2.0, y=1.4, z=0.0),
    ]
    graph.def_bond(atoms[0], atoms[1])
    graph.def_bond(atoms[1], atoms[2])
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
        from_smiles = molrs.io.smiles.SmilesIr("CO").to_atomistic()
        for graph in (
            from_smiles,
            from_smiles.copy(),
            molrs.core.Atomistic.from_frame(from_smiles.to_frame()),
            molrs.perceive.assign_rings(from_smiles),
        ):
            assert type(graph) is molrs.core.Atomistic
            assert len(graph.atoms) == 2


class TestProps:
    def test_constructor_keywords_are_the_props(self) -> None:
        assert molrs.core.Atomistic(name="water").props == {"name": "water"}
        assert molrs.core.CoarseGrain(name="lipid").props == {"name": "lipid"}
        assert molrs.core.Atomistic().props == {}

    def test_copy_keeps_props_in_an_independent_dict(self) -> None:
        graph = molrs.core.Atomistic(label="x")
        copy = graph.copy()
        assert copy.props == {"label": "x"}
        copy.props["label"] = "y"
        assert graph.props == {"label": "x"}

    def test_perception_output_keeps_props(self) -> None:
        graph = molrs.io.smiles.SmilesIr("CO").to_atomistic()
        graph.props["label"] = "methanol"
        perceived = molrs.perceive.assign_rings(graph)
        assert perceived.props == {"label": "methanol"}

    def test_coarse_grain_copy_keeps_props(self) -> None:
        assert molrs.core.CoarseGrain(label="cg").copy().props == {"label": "cg"}

    def test_pickle_keeps_props(self) -> None:
        assert _roundtrip(molrs.core.Atomistic(label="x")).props == {"label": "x"}


class TestBeadMembership:
    def test_bead_atoms_are_the_source_atom_views(self) -> None:
        source, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        bead = cg.def_bead(type="CC", atoms=(atoms[0], atoms[1]))
        assert bead["atoms"] == (atoms[0], atoms[1])
        assert all(atom.world is source for atom in bead["atoms"])
        assert "atoms" in bead
        assert cg.def_bead(type="empty").get("atoms") is None

    def test_copy_keeps_bead_membership(self) -> None:
        _, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        cg.def_bead(type="CC", atoms=(atoms[0], atoms[1]))
        copied = cg.copy()
        assert copied.beads[0]["atoms"] == (atoms[0], atoms[1])

    def test_pickle_keeps_bead_membership(self) -> None:
        source, atoms = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        cg.def_bead(type="CO", atoms=(atoms[1], atoms[2]))
        restored_source, restored_cg = _roundtrip((source, cg))
        members = restored_cg.beads[0]["atoms"]
        assert members == (restored_source.atoms[1], restored_source.atoms[2])

    def test_membership_atoms_come_from_one_world(self) -> None:
        _, left = _ethanol_skeleton()
        _, right = _ethanol_skeleton()
        cg = molrs.core.CoarseGrain()
        with pytest.raises(ValueError, match="same source world"):
            cg.def_bead(type="X", atoms=(left[0], right[0]))


class TestFactories:
    def test_factories_return_interned_live_refs(self) -> None:
        graph = molrs.core.Atomistic()
        carbon = graph.def_atom(element="C", x=0.0, y=0.0, z=0.0)
        oxygen = graph.def_atom(element="O", x=1.0, y=0.0, z=0.0)
        bond = graph.def_bond(carbon, oxygen, order=2.0)

        assert type(carbon) is molrs.core.Atom
        assert type(bond) is molrs.core.Bond
        assert graph.atoms[0] is carbon
        assert graph.bonds[0] is bond
        assert bond.itom is carbon
        assert bond.jtom is oxygen
        assert bond.endpoints == (carbon, oxygen)
        assert carbon["x", "y", "z"] == [0.0, 0.0, 0.0]
        assert bond["order"] == 2.0

    def test_def_bond_stamps_both_bond_facts(self) -> None:
        graph = molrs.core.Atomistic()
        a = graph.def_atom(element="C")
        b = graph.def_atom(element="C")
        bond = graph.def_bond(a, b)
        assert bond["bond_type"] == 1
        assert bond["bond_number"] == 1

    def test_higher_relations_are_the_native_kinds(self) -> None:
        graph, (a, b, c) = _ethanol_skeleton()
        d = graph.def_atom(element="H")
        angle = graph.def_angle(a, b, c, theta0=1.9)
        dihedral = graph.def_dihedral(a, b, c, d)
        improper = graph.def_improper(b, a, c, d)

        assert (type(angle), type(dihedral), type(improper)) == (
            molrs.core.Angle,
            molrs.core.Dihedral,
            molrs.core.Improper,
        )
        assert graph.angles[0] is angle
        assert angle.endpoints == (a, b, c)
        assert angle["theta0"] == 1.9
        assert graph.n_relations("angles") == 1
        assert graph.n_relations("dihedrals") == 1
        assert graph.n_relations("impropers") == 1

    def test_refs_have_no_detached_constructor(self) -> None:
        with pytest.raises(TypeError):
            molrs.core.Atom(element="C")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            molrs.core.Bond(object(), object())  # type: ignore[call-arg]

    def test_cross_world_relation_is_rejected(self) -> None:
        left = molrs.core.Atomistic()
        right = molrs.core.Atomistic()
        a = left.def_atom(element="C")
        b = right.def_atom(element="C")
        with pytest.raises(ValueError, match="belong to this graph"):
            left.def_bond(a, b)
        with pytest.raises(ValueError, match="belong to this graph"):
            left.def_angle(a, b, a)

    def test_virtual_site_class_follows_its_stored_kind(self) -> None:
        graph = molrs.core.Atomistic()
        graph.def_atom(element="O")
        graph.def_virtual_site(kind=molrs.core.DrudeParticle, charge=-1.0)
        graph.def_virtual_site(kind=molrs.core.MasslessSite)
        graph.def_virtual_site()

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

    def test_def_port_returns_the_port_view(self) -> None:
        graph = molrs.core.Atomistic()
        anchor = graph.def_atom(element="O")
        handle = graph.def_atom(element="H")
        graph.def_bond(anchor, handle)
        port = graph.def_port(anchor, handle, "$")
        assert type(port) is molrs.core.Port
        assert graph.ports[0] is port
        assert port.anchor is anchor
        assert port.handle_atom is handle
        assert port["port_kind"] == "$"

    def test_coarse_grain_factories(self) -> None:
        cg = molrs.core.CoarseGrain()
        a = cg.def_bead(bead_type="P1", x=0.0, y=0.0, z=0.0)
        b = cg.def_bead(bead_type="P1", x=1.0, y=0.0, z=0.0)
        bond = cg.def_cgbond(a, b, order=1.0)
        assert type(a) is molrs.core.Bead
        assert type(bond) is molrs.core.CgBond
        assert cg.beads[1] is b
        assert cg.cgbonds[0] is bond
        assert bond.endpoints == (a, b)
        assert bond["order"] == 1.0


class TestRemoval:
    def test_del_atom_cascades_and_leaves_a_stale_view(self) -> None:
        graph, (a, _, _) = _ethanol_skeleton()
        graph.del_atom(a)
        assert len(graph.atoms) == 2
        assert graph.n_relations("bonds") == 1
        # The view outlives its atom; a read through it finds no field.
        with pytest.raises(KeyError):
            _ = a["element"]

    @pytest.mark.parametrize("cls", ["Atomistic", "CoarseGrain"])
    def test_remove_link_removes_the_relations(self, cls: str) -> None:
        graph = getattr(molrs.core, cls)()
        if cls == "Atomistic":
            a, b = graph.def_atom(element="C"), graph.def_atom(element="C")
            bond = graph.def_bond(a, b)
        else:
            a, b = graph.def_bead(type="P"), graph.def_bead(type="P")
            bond = graph.def_cgbond(a, b)
        graph.remove_link(bond)
        assert graph.n_relations("bonds") == 0

    def test_remove_link_refuses_a_foreign_relation(self) -> None:
        graph, _ = _ethanol_skeleton()
        other, _ = _ethanol_skeleton()
        with pytest.raises(ValueError, match="another graph"):
            graph.remove_link(other.bonds[0])


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
        assert graph.atoms["element"].tolist() == ["C", "C", "O"]
        assert graph.bonds["bond_type"].tolist() == [1, 1]

    def test_exact_bucket_selects_one_kind(self) -> None:
        graph, (a, b, c) = _ethanol_skeleton()
        angle = graph.def_angle(a, b, c)
        assert list(graph.links.exact_bucket(molrs.core.Angle)) == [angle]
        assert len(graph.links.exact_bucket(molrs.core.Bond)) == 2
        assert len(graph.links.exact_bucket(molrs.core.Improper)) == 0

    def test_an_empty_exact_bucket_reads_as_empty_columns(self) -> None:
        graph, _ = _ethanol_skeleton()
        empty = graph.links.exact_bucket(molrs.core.Improper)
        assert len(empty) == 0
        assert empty["type"].shape == (0,)
        assert "impropers" in repr(empty)

    def test_an_unregistered_kind_is_a_generic_relation(self) -> None:
        graph, (a, b, _) = _ethanol_skeleton()
        graph.register_kind("constraints", 2)
        graph.add_relation("constraints", [a.handle, b.handle])
        (constraint,) = graph.links.exact_bucket(molrs.core.RelationRef)
        assert type(constraint) is molrs.core.RelationRef
        assert constraint.kind == "constraints"
        assert constraint.endpoints == (a, b)

    def test_from_frame_reads_back_drudes_and_virtual_sites(self) -> None:
        graph = molrs.core.Atomistic()
        o = graph.def_atom(element="O", x=0.0, y=0.0, z=0.0)
        h1 = graph.def_atom(element="H", x=0.96, y=0.0, z=0.0)
        h2 = graph.def_atom(element="H", x=-0.24, y=0.93, z=0.0)
        graph.def_bond(o, h1)
        graph.def_bond(o, h2)
        shell = graph.def_virtual_site(kind=molrs.core.DrudeParticle, charge=-1.0)
        m = graph.def_virtual_site(kind=molrs.core.MasslessSite, charge=-1.0)
        graph.register_kind("drudes", 2)
        rh = graph.add_relation("drudes", [o.handle, shell.handle])
        graph.set_relation_prop("drudes", rh, "type", "OW-DOW")
        graph.set_relation_prop("drudes", rh, "k", 500.0)
        graph.register_kind("virtual_sites", 4)
        rh = graph.add_relation(
            "virtual_sites", [m.handle, o.handle, h1.handle, h2.handle]
        )
        graph.set_relation_prop("virtual_sites", rh, "style", "average3")

        frame = graph.to_frame()
        back = molrs.core.Atomistic.from_frame(frame)
        assert back.n_relations("drudes") == 1
        assert back.n_relations("virtual_sites") == 1
        again = back.to_frame()
        for block in ("drudes", "virtual_sites"):
            for key in frame[block].keys():
                np.testing.assert_array_equal(again[block][key], frame[block][key])
        assert isinstance(back.atoms[3], molrs.core.DrudeParticle)


class TestRefMapping:
    def test_node_ref_is_a_mapping_over_its_fields(self) -> None:
        graph = molrs.core.Atomistic()
        atom = graph.def_atom(element="C", charge=0.5)
        assert dict(atom) == {"element": "C", "charge": 0.5}
        assert set(atom.keys()) == {"element", "charge"}
        assert len(atom) == 2
        assert "charge" in atom and "mass" not in atom
        assert atom.get("mass", 12.0) == 12.0
        atom.update(mass=12.011)
        assert atom["mass"] == 12.011
        atom["charge"] = None
        assert "charge" not in atom
        del atom["mass"]
        assert dict(atom.items()) == {"element": "C"}
        with pytest.raises(KeyError):
            _ = atom["mass"]

    def test_tuple_keys_read_and_write_several_fields(self) -> None:
        graph = molrs.core.Atomistic()
        atom = graph.def_atom(element="C")
        atom["x", "y", "z"] = (1.0, 2.0, 3.0)
        assert atom["x", "y", "z"] == [1.0, 2.0, 3.0]
        assert ("x", "y", "z") in atom
        with pytest.raises(ValueError, match="needs 3 values"):
            atom["x", "y", "z"] = (1.0, 2.0)


class TestPickle:
    def test_graph_and_views_pickle_as_one_object_graph(self) -> None:
        graph, atoms = _ethanol_skeleton()
        graph.del_atom(graph.def_atom(element="H"))  # a hole in the handles
        drude = graph.def_virtual_site(kind=molrs.core.DrudeParticle)
        refs = [*atoms, drude, graph.bonds[1]]
        restored, restored_refs, restored_atoms = _roundtrip((graph, refs, graph.atoms))
        assert [type(ref) for ref in restored_refs] == [type(ref) for ref in refs]
        assert all(ref.world is restored for ref in restored_refs)
        assert restored_refs[:4] == list(restored.atoms)
        assert restored_refs[4] is restored.bonds[1]
        assert list(restored_atoms) == list(restored.atoms)
        assert restored_refs[4].endpoints == (restored_refs[1], restored_refs[2])
