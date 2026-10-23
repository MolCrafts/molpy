import copy
import pickle

import molpy as mp
import numpy as np
import pytest


def _unit_cube_mesh() -> "mp.core.TriMesh":
    """Closed unit cube, 12 triangles, outward winding."""
    v = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    # vertex index = x + 2*y + 4*z
    faces = np.array(
        [
            [0, 4, 6],
            [0, 6, 2],  # -x
            [1, 3, 7],
            [1, 7, 5],  # +x
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 6, 7],
            [2, 7, 3],  # +y
            [0, 2, 3],
            [0, 3, 1],  # -z
            [4, 5, 7],
            [4, 7, 6],  # +z
        ],
        dtype=np.uint32,
    )
    return mp.core.TriMesh(v, faces)


def roundtrip(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def test_storage_units_and_observables_pickle_by_logical_state() -> None:
    block = mp.core.Block()
    block.insert("sample", np.array([1, 2], dtype=np.int16))
    block.insert("label", ["left", "right"])
    block.set_shape([1, 2])
    restored = roundtrip(block)
    assert type(restored) is mp.core.Block
    assert restored.dtype("sample") == "i16"
    assert restored["sample"].tolist() == [1, 2]
    assert restored["label"].tolist() == ["left", "right"]
    assert restored.structural_shape == [1, 2]

    empty_rows = mp.core.Block()
    empty_rows.resize(3)
    assert roundtrip(empty_rows).n_rows == 3

    frame = mp.core.Frame({"grid": block}, meta={"nested": {"ok": True}})
    frame.box = mp.core.Box(mp.core.Cell.lengths([4.0, 4.0, 4.0]))
    restored_frame = roundtrip(frame)
    assert type(restored_frame) is mp.core.Frame
    assert restored_frame["grid"].dtype("sample") == "i16"
    assert restored_frame.meta["nested"] == {"ok": True}
    assert restored_frame.box.volume() == 64.0

    registry = mp.core.UnitRegistry()
    registry.define("smoot", 1.7018, [1, 0, 0, 0, 0, 0, 0])
    registry.note = "custom registry"
    restored_registry = roundtrip(registry)
    assert restored_registry.note == "custom registry"
    assert restored_registry.smoot.factor_to(restored_registry.m) == 1.7018

    quantity = roundtrip(2.5 * registry.smoot)
    assert quantity.value == 2.5
    assert str(quantity.unit) == "smoot"
    assert roundtrip(mp.core.UnitPreset("real")).name == "real"
    assert roundtrip(mp.core.Element.by_symbol("C")) == mp.core.Element.by_number(6)

    observable = mp.core.ObservableRecord(
        "force",
        "vector",
        np.array([[1.0, 2.0, 3.0]]),
        "force vector",
        "kJ/mol/nm",
        ["atom", "xyz"],
        True,
        "every_step",
        "particle",
        "atoms",
    )
    restored_observable = roundtrip(observable)
    np.testing.assert_array_equal(restored_observable.values, observable.values)
    assert restored_observable.description == "force vector"
    assert restored_observable.axes == ["atom", "xyz"]
    assert restored_observable.target == "atoms"
    assert restored_observable.kind == "vector"
    other = roundtrip(mp.core.ObservableRecord("s", "spectrum", np.array([0.5])))
    assert other.kind == "spectrum"

    trajectory = mp.core.Trajectory(
        [frame], step=np.array([7], dtype=np.int64), time=np.array([0.5])
    )
    restored_trajectory = roundtrip(trajectory)
    assert restored_trajectory.step.tolist() == [7]
    assert restored_trajectory.time.tolist() == [0.5]
    assert restored_trajectory.frames[0].box.volume() == 64.0


def test_spatial_types_preserve_queries_and_region_behavior() -> None:
    box = mp.core.Box(
        mp.core.Cell.lengths([10.0, 10.0, 10.0]), pbc=np.array([True, False, True])
    )
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]])

    restored_box = roundtrip(box)
    np.testing.assert_array_equal(restored_box.pbc, box.pbc)
    np.testing.assert_array_equal(restored_box.matrix, box.matrix)

    neighbor_list = mp.core.NeighborList(1.0, brute_force=True)
    neighbor_list.build(points, box)
    restored_list = roundtrip(neighbor_list)
    restored_pairs = restored_list.neighbors(disp=False)
    assert restored_pairs.query_point_indices().tolist() == [0]
    assert restored_pairs.point_indices().tolist() == [1]
    assert restored_pairs.disp() is None

    table = roundtrip(neighbor_list.neighbors(dist_sq=False, disp=True))
    assert table.dist_sq() is None
    np.testing.assert_allclose(table.disp(), [[0.5, 0.0, 0.0]])

    query = roundtrip(mp.core.NeighborQuery(box, points, 1.0))
    assert query.query_self().n_pairs == 1
    assert query.query(np.array([[0.25, 0.0, 0.0]])).n_pairs == 2

    skin = mp.core.VerletSkin(
        mp.core.NeighborList(1.2),
        1.0,
        points,
        box,
        skin=0.2,
        every=2,
    )
    assert skin.update(points + 0.01) is False
    restored_skin = roundtrip(skin)
    assert restored_skin.ago == 1
    assert restored_skin.n_rebuilds == 0
    assert restored_skin.n_edges == skin.n_edges

    cube_mesh = _unit_cube_mesh()
    primitives = [
        mp.core.Sphere(np.zeros(3), 2.0),
        mp.core.Cuboid(np.zeros(3), np.ones(3)),
        mp.core.Parallelepiped(mp.core.Cell.matrix(np.eye(3)), np.zeros(3)),
        mp.core.HalfSpace(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5])),
        mp.core.Cylinder(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, 2.0),
        mp.core.Ellipsoid(np.zeros(3), np.array([1.0, 2.0, 0.5])),
        mp.core.Polyhedron(cube_mesh),
        mp.core.SphereUnion(np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), 0.8),
        mp.core.SphereUnion(
            np.array([[0.2, 0.2, 0.2]]),
            np.array([0.5]),
            box=mp.core.Box(
                mp.core.Cell.lengths([3.0, 3.0, 3.0]),
                np.zeros(3),
                np.array([True, True, True]),
            ),
        ),
    ]
    composed = (primitives[0] & ~primitives[1]) | (primitives[3] & primitives[7])
    restored_mesh = roundtrip(cube_mesh)
    assert isinstance(restored_mesh, mp.core.TriMesh)
    np.testing.assert_array_equal(restored_mesh.faces(), cube_mesh.faces())
    probes = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.5, 0.0, 0.0]])
    for region in [*primitives, composed, mp.core.Region(primitives[0])]:
        restored = roundtrip(region)
        assert type(restored) is type(region)
        np.testing.assert_array_equal(
            restored.contains(probes), region.contains(probes)
        )


def test_graphs_views_and_extraction_pickle_as_one_object_graph() -> None:
    graph = mp.core.MolGraph()
    graph.register_kind("empty", 3)
    graph.register_kind("links", 2)
    first, removed, last = [graph.add_node() for _ in range(3)]
    graph.del_node(removed)
    graph.set_node(first, "name", "first")
    graph.set_node(last, "x", 2.0)
    relation = graph.add_relation("links", [first, last])
    graph.set_relation_prop("links", relation, "order", 2)
    restored_graph = roundtrip(graph)
    handles = restored_graph.node_ids()
    assert restored_graph.kind_arity("empty") == 3
    assert restored_graph.get_node(handles[0]) == {"name": "first"}
    assert restored_graph.get_relation(
        "links", restored_graph.relation_ids("links")[0]
    ) == (handles, {"order": 2})

    molecule = mp.core.Atomistic(label="typed refs")
    atoms = [molecule.add_atom(atomic_number=6) for _ in range(4)]
    virtual = molecule.add_atom(kind=mp.core.VirtualSite, atomic_number=0)
    drude = molecule.add_atom(kind=mp.core.DrudeParticle, atomic_number=0)
    massless = molecule.add_atom(kind=mp.core.MasslessSite, atomic_number=0)
    refs = [
        *atoms,
        virtual,
        drude,
        massless,
        molecule.add_bond(atoms[0], atoms[1]),
        molecule.add_angle(*atoms[:3]),
        molecule.add_dihedral(*atoms),
        molecule.add_improper(*atoms),
    ]
    restored_molecule, restored_refs, restored_lazy = roundtrip(
        (molecule, refs, molecule.atoms)
    )
    assert restored_molecule.props == {"label": "typed refs"}
    assert [type(ref) for ref in restored_refs] == [type(ref) for ref in refs]
    assert all(ref.world is restored_molecule for ref in restored_refs)
    assert all(ref.world is restored_molecule for ref in restored_lazy)

    coarse = mp.core.CoarseGrain(label="cg")
    bead = coarse.add_bead(bead_type="P")
    coarse.set_bead_members(bead, atoms[:2])
    coarse.add_bead(bead_type="Q")
    coarse.add_bond(coarse.beads[0], coarse.beads[1])
    restored_source, restored_coarse, restored_bead = roundtrip(
        (molecule, coarse, bead)
    )
    assert restored_bead is restored_coarse.beads[0]
    members = restored_coarse.bead_members(restored_bead)
    assert members == tuple(restored_source.atoms[:2])

    chain = mp.core.Atomistic()
    chain_atoms = [chain.add_atom(atomic_number=6) for _ in range(3)]
    chain.add_bond(chain_atoms[0], chain_atoms[1])
    chain.add_bond(chain_atoms[1], chain_atoms[2])
    extracted = roundtrip(chain.extract_subgraph([chain_atoms[1]], 1))
    assert len(extracted.graph.atoms) == 3
    assert set(extracted.parent_of) == set(extracted.graph.atoms)

    fragment = mp.core.Atomistic()
    fragment_anchor = fragment.add_atom(atomic_number=8)
    fragment_handle = fragment.add_atom(atomic_number=1)
    fragment.add_bond(fragment_anchor, fragment_handle)
    fragment.add_port(fragment_anchor, fragment_handle, "$")
    fragment_anchor["frag_id"] = 2
    restored_fragment = roundtrip(fragment)
    assert type(restored_fragment) is mp.core.Atomistic
    assert restored_fragment.n_ports == 1
    assert restored_fragment.ports[0]["port_kind"] == "$"
    assert restored_fragment.atoms[0]["frag_id"] == 2

    # A partially labelled fragment keeps its holes: an unlabelled atom comes
    # back unlabelled, never as a stated ``frag_id`` of 0.
    partial = mp.core.Atomistic()
    labelled = partial.add_atom(atomic_number=6)
    middle = partial.add_atom(atomic_number=6)
    capping = partial.add_atom(atomic_number=1)
    partial.add_bond(labelled, middle)
    partial.add_bond(middle, capping)
    labelled["frag_id"] = 7
    restored_partial = roundtrip(partial)
    assert [atom.get("frag_id") for atom in restored_partial.atoms] == [7, None, None]

    # Term blocks come back through the typed `add_term`.
    termed = mp.core.Atomistic()
    ends = [termed.add_atom(atomic_number=6) for _ in range(2)]
    termed.add_term("exclusions", ends, weight=0.5)
    restored_termed = roundtrip(termed)
    assert restored_termed.term_blocks() == ["exclusions"]
    (exclusion,) = restored_termed.terms("exclusions")
    assert exclusion["weight"] == 0.5
    assert exclusion.endpoints == tuple(restored_termed.atoms)

    # Unpickling adds every node through the leaf's checked add: a state whose
    # atom has no valid ``atomic_number`` (or bead no ``bead_type``) raises.
    with pytest.raises(ValueError, match="atomic_number"):
        mp.core.Atomistic().__setstate__(({}, ([{"x": 0.0}], [], [])))
    with pytest.raises(ValueError, match="bead_type"):
        mp.core.CoarseGrain().__setstate__(({}, ([{"x": 0.0}], []), None, []))

    assert type(roundtrip(mp.core.Atomistic())) is mp.core.Atomistic
    assert type(roundtrip(mp.core.CoarseGrain())) is mp.core.CoarseGrain
    assert roundtrip(mp.perceive.Reaction("[C:1]>>[C:1]")).forming_bonds == []


def test_schema_and_metadata_pickle_as_value_types() -> None:
    column = mp.core.schema.columns[0]
    restored_column = roundtrip(column)
    assert type(restored_column) is type(column)
    assert restored_column.key == column.key
    assert restored_column.const_name == column.const_name
    assert restored_column.dtype == column.dtype
    assert restored_column.shape == column.shape
    assert restored_column.unit == column.unit
    assert restored_column.doc == column.doc
    assert restored_column.numpy_dtype == column.numpy_dtype

    block = next(spec for spec in mp.core.schema.blocks if spec.endpoint_columns)
    restored_block = roundtrip(block)
    assert type(restored_block) is type(block)
    assert restored_block.name == block.name
    assert restored_block.row_kind == block.row_kind
    assert restored_block.endpoint_target == block.endpoint_target
    assert restored_block.endpoint_columns == block.endpoint_columns
    assert restored_block.required == block.required
    assert restored_block.optional == block.optional
    assert restored_block.open == block.open
    assert restored_block.doc == block.doc

    assert [spec.key for spec in roundtrip(list(mp.core.schema.columns))] == [
        spec.key for spec in mp.core.schema.columns
    ]

    text = roundtrip(mp.core.MetaValue("string", "hello"))
    assert text.dtype == "string"
    assert text.value == "hello"
    nested = roundtrip(mp.core.MetaValue("json", {"ok": True}))
    assert nested.dtype == "json"
    assert nested.value == {"ok": True}


def test_frame_json_meta_roundtrip_stays_json_and_frozen() -> None:
    frame = mp.core.Frame(meta={"nested": {"ok": True, "tags": [1, 2]}})
    restored = roundtrip(frame)
    assert restored.meta.dtype("nested") == "json"
    assert isinstance(restored.meta["nested"], mp.core.MetaDocument)
    assert isinstance(restored.meta["nested"], dict) is False
    assert restored.meta["nested"] == {"ok": True, "tags": (1, 2)}


def test_meta_value_json_payload_is_plain_and_vector_payload_is_tuple() -> None:
    vector = mp.core.MetaValue("f64x6", [1, 2, 3, 4, 5, 6])
    assert isinstance(vector.value, tuple)
    assert vector.value == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    document = mp.core.MetaValue("json", {"ok": True, "tags": [1, 2]})
    assert isinstance(document.value, dict)
    assert isinstance(document.value["tags"], list)
    assert document.value == {"ok": True, "tags": [1, 2]}
    assert not isinstance(document.value, mp.core.MetaDocument)


def test_document_assigned_across_frames_keeps_json_dtype() -> None:
    src = mp.core.Frame()
    src.meta["run"] = {"step": 1, "tags": [1, 2], "inner": {"a": [3]}}
    dst = mp.core.Frame()
    dst.meta["copied"] = src.meta["run"]
    assert dst.meta.dtype("copied") == "json"
    assert isinstance(dst.meta["copied"], mp.core.MetaDocument)
    assert dst.meta["copied"] == {"step": 1, "tags": (1, 2), "inner": {"a": (3,)}}
    assert isinstance(dst.meta["copied"]["tags"], tuple)
    assert isinstance(dst.meta["copied"]["inner"]["a"], tuple)


def test_a_meta_document_pickles_by_content() -> None:
    frame = mp.core.Frame(meta={"doc": {"ok": True, "tags": [1, 2], "inner": {"a": 1}}})
    doc = frame.meta["doc"]
    assert roundtrip(doc) == {"ok": True, "tags": (1, 2), "inner": {"a": 1}}


def test_a_meta_document_deep_copies_by_content() -> None:
    frame = mp.core.Frame(meta={"doc": {"ok": True, "tags": [1, 2], "inner": {"a": 1}}})
    doc = frame.meta["doc"]
    assert copy.deepcopy(doc) == {"ok": True, "tags": (1, 2), "inner": {"a": 1}}


def test_a_meta_dict_snapshot_with_a_document_pickles() -> None:
    frame = mp.core.Frame(meta={"doc": {"ok": True, "tags": [1, 2]}, "step": 3})
    restored = roundtrip(dict(frame.meta))
    assert restored == {"doc": {"ok": True, "tags": (1, 2)}, "step": 3}


def test_custom_term_declarations_survive_pickle():
    molecule = mp.core.Atomistic()
    atoms = [molecule.add_atom(atomic_number=6) for _ in range(2)]
    molecule.declare_term("empty_custom", 3)
    molecule.declare_term("custom", 2)
    molecule.add_term("custom", atoms)
    restored = roundtrip(molecule)
    assert restored.term_blocks() == ["empty_custom", "custom"]
    assert restored.n_terms("custom") == 1
    with pytest.raises(ValueError):
        restored.add_term("empty_custom", list(restored.atoms))
