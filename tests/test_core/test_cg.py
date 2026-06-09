"""
Test CoarseGrain class API for creating and managing coarse-grained structures.

Tests the redesigned CG module per spec
``docs/specs/cg-atomistic-mapping-redesign.md``:

- ``Bead`` is a structurally empty :class:`Entity` (no mandatory fields).
- ``bead["atoms"]`` is a soft convention key (``tuple[Atom, ...]``); MolPy
  neither enforces nor writes it.
- ``CGBond`` is a two-endpoint :class:`Link` between two beads.
- ``CoarseGrain`` mirrors :class:`Atomistic` plus the single extra method
  :meth:`CoarseGrain.beads_of` for reverse lookup.
- No ``from_atomistic`` / ``to_atomistic`` / ``Bead.atomistic`` field.
"""

import copy

import numpy as np
import pytest

from molpy.core.atomistic import Atomistic
from molpy.core.cg import Bead, CGBond, CoarseGrain


# ---------------------------------------------------------------------------
# Bead — structurally empty Entity
# ---------------------------------------------------------------------------


class TestBeadEntity:
    """Test Bead entity class — no required fields, dict-like."""

    def test_bead_is_empty_entity(self):
        """Bead() is creatable with zero arguments and has no required keys."""
        bead = Bead()

        assert isinstance(bead, Bead)
        # The underlying data dict starts empty.
        assert dict(bead) == {}

    def test_bead_dict_interface(self):
        """Bead exposes attributes through the dict interface."""
        bead = Bead(type="A", x=1.0)

        assert bead["type"] == "A"
        assert bead["x"] == 1.0

    def test_bead_deepcopy_produces_independent_object(self):
        """Default Python deepcopy semantics apply: the clone is a separate
        Bead with an independent dict. Bead has no custom ``__deepcopy__``
        and therefore makes no opinion about which conventional keys
        should be shared vs. cloned — that's the user's call."""
        ato = Atomistic()
        a = ato.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
        b = ato.def_atom(symbol="H", x=1.0, y=0.0, z=0.0)

        bead = Bead(atoms=(a, b), type="CH")
        clone = copy.deepcopy(bead)

        assert clone is not bead
        assert clone["type"] == "CH"
        # dict is independent: mutating the clone does not affect the original
        clone["type"] = "MUTATED"
        assert bead["type"] == "CH"

    def test_create_bead_with_attributes(self):
        """Beads carry arbitrary attributes via kwargs."""
        bead = Bead(type="PEO", x=1.0, y=2.0, z=3.0)

        assert isinstance(bead, Bead)
        assert bead.get("type") == "PEO"
        assert bead.get("x") == 1.0
        assert bead.get("y") == 2.0
        assert bead.get("z") == 3.0

    def test_bead_repr_contains_class_name(self):
        """Bead repr identifies the class."""
        bead = Bead(type="PEO")
        assert "Bead" in repr(bead)


# ---------------------------------------------------------------------------
# CGBond — two-endpoint Link
# ---------------------------------------------------------------------------


class TestCGBondLink:
    """Test CGBond link class."""

    def test_create_cgbond(self):
        """Create a CGBond between two beads with attributes."""
        bead1 = Bead(type="A")
        bead2 = Bead(type="B")

        bond = CGBond(bead1, bead2, type="harmonic")

        assert isinstance(bond, CGBond)
        assert bond.ibead is bead1
        assert bond.jbead is bead2
        assert bond.get("type") == "harmonic"

    def test_cgbond_endpoints(self):
        """CGBond.ibead / .jbead expose the endpoints in order."""
        bead1 = Bead(type="A")
        bead2 = Bead(type="B")
        bond = CGBond(bead1, bead2)

        assert bond.ibead is bead1
        assert bond.jbead is bead2

    def test_cgbond_repr(self):
        """CGBond repr identifies the class."""
        bead1 = Bead(type="A")
        bead2 = Bead(type="B")
        bond = CGBond(bead1, bead2)

        assert "CGBond" in repr(bond)

    def test_cgbond_requires_beads(self):
        """CGBond rejects non-Bead endpoints."""
        bead = Bead(type="A")
        not_a_bead = object()

        with pytest.raises(AssertionError):
            CGBond(bead, not_a_bead)


# ---------------------------------------------------------------------------
# def_bead / def_cgbond registration — parallel to Atomistic.def_atom / def_bond
# ---------------------------------------------------------------------------


class TestCoarseGrainFactoryMethods:
    """Test def_* factory methods that create and register entities."""

    def test_def_bead_registers_in_entities(self):
        """def_bead returns the new Bead and it appears in ``cg.beads``."""
        cg = CoarseGrain()
        bead = cg.def_bead(type="PEO", x=0, y=0, z=0)

        assert isinstance(bead, Bead)
        assert bead.get("type") == "PEO"
        assert len(cg.beads) == 1
        assert bead in cg.beads

    def test_def_cgbond_registers_in_links(self):
        """def_cgbond creates a CGBond between two beads and registers it."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")

        bond = cg.def_cgbond(b1, b2, type="harmonic")

        assert isinstance(bond, CGBond)
        assert bond.ibead is b1
        assert bond.jbead is b2
        assert bond.get("type") == "harmonic"
        assert len(cg.cgbonds) == 1
        assert bond in cg.cgbonds


class TestCoarseGrainAddMethods:
    """Test add_* methods that register pre-built entity objects."""

    def test_add_bead_adds_existing(self):
        """add_bead registers an already-constructed Bead."""
        cg = CoarseGrain()
        bead = Bead(type="PEO", x=0, y=0, z=0)

        result = cg.add_bead(bead)

        assert result is bead
        assert len(cg.beads) == 1
        assert next(iter(cg.beads)) is bead

    def test_add_cgbond_adds_existing(self):
        """add_cgbond registers an already-constructed CGBond."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")
        bond = CGBond(b1, b2, type="harmonic")

        result = cg.add_cgbond(bond)

        assert result is bond
        assert len(cg.cgbonds) == 1
        assert next(iter(cg.cgbonds)) is bond


class TestCoarseGrainBatchMethods:
    """Test batch operations for creating and adding multiple entities."""

    def test_def_beads_batch_create(self):
        """def_beads creates multiple beads at once."""
        cg = CoarseGrain()

        beads = cg.def_beads(
            [
                {"type": "PEO", "x": 0, "y": 0, "z": 0},
                {"type": "PMA", "x": 5, "y": 0, "z": 0},
                {"type": "PEO", "x": 10, "y": 0, "z": 0},
            ]
        )

        assert len(beads) == 3
        assert all(isinstance(b, Bead) for b in beads)
        assert len(cg.beads) == 3
        assert np.array_equal(cg.beads["type"], ["PEO", "PMA", "PEO"])

    def test_def_cgbonds_batch_create(self):
        """def_cgbonds creates multiple bonds at once."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")
        b3 = cg.def_bead(type="C")

        bonds = cg.def_cgbonds(
            [
                (b1, b2, {"type": "harmonic"}),
                (b2, b3, {"type": "harmonic"}),
            ]
        )

        assert len(bonds) == 2
        assert all(isinstance(b, CGBond) for b in bonds)
        assert len(cg.cgbonds) == 2

    def test_add_beads_batch_add(self):
        """add_beads adds multiple existing Bead objects."""
        cg = CoarseGrain()
        beads = [
            Bead(type="A", x=0, y=0, z=0),
            Bead(type="B", x=5, y=0, z=0),
        ]

        result = cg.add_beads(beads)

        assert result == beads
        assert len(cg.beads) == 2

    def test_add_cgbonds_batch_add(self):
        """add_cgbonds adds multiple existing CGBond objects."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")
        b3 = cg.def_bead(type="C")

        bonds = [
            CGBond(b1, b2),
            CGBond(b2, b3),
        ]

        result = cg.add_cgbonds(bonds)

        assert result == bonds
        assert len(cg.cgbonds) == 2


# ---------------------------------------------------------------------------
# beads_of — the single extra core method (reverse lookup)
# ---------------------------------------------------------------------------


class TestBeadsOf:
    """Test ``CoarseGrain.beads_of(atom)`` reverse lookup over the conventional
    ``bead["atoms"]`` key."""

    def test_beads_of_disjoint(self):
        """In a disjoint partition each atom maps to exactly one bead."""
        ato = Atomistic()
        a = ato.def_atom(symbol="C", x=0, y=0, z=0)
        b = ato.def_atom(symbol="O", x=5, y=0, z=0)

        cg = CoarseGrain()
        bead_a = cg.def_bead(atoms=(a,), type="A")
        bead_b = cg.def_bead(atoms=(b,), type="B")

        assert cg.beads_of(a) == (bead_a,)
        assert cg.beads_of(b) == (bead_b,)

    def test_beads_of_overlap(self):
        """An atom shared by two beads is returned in both."""
        ato = Atomistic()
        a = ato.def_atom(symbol="C", x=0, y=0, z=0)
        b = ato.def_atom(symbol="O", x=5, y=0, z=0)
        c = ato.def_atom(symbol="N", x=10, y=0, z=0)

        cg = CoarseGrain()
        bead1 = cg.def_bead(atoms=(a, b), type="AB")
        bead2 = cg.def_bead(atoms=(b, c), type="BC")

        result = cg.beads_of(b)

        assert len(result) == 2
        assert bead1 in result
        assert bead2 in result

    def test_beads_of_unknown_returns_empty(self):
        """Unknown atom (not in any bead's ``atoms`` key) gives an empty tuple."""
        ato = Atomistic()
        a = ato.def_atom(symbol="C", x=0, y=0, z=0)
        unknown = ato.def_atom(symbol="X", x=99, y=0, z=0)

        cg = CoarseGrain()
        cg.def_bead(atoms=(a,), type="A")

        assert cg.beads_of(unknown) == ()

    def test_beads_of_skips_beads_without_atoms_key(self):
        """Beads lacking an ``atoms`` key are silently skipped (not errors)."""
        ato = Atomistic()
        a = ato.def_atom(symbol="C", x=0, y=0, z=0)

        cg = CoarseGrain()
        # Bead with no AA precursor (pure CG bead)
        cg.def_bead(type="pure_cg", x=0, y=0, z=0)
        bead_with_atoms = cg.def_bead(atoms=(a,), type="mapped")

        result = cg.beads_of(a)

        assert result == (bead_with_atoms,)


# ---------------------------------------------------------------------------
# Spatial / composition — parallel to Atomistic
# ---------------------------------------------------------------------------


class TestCoarseGrainSpatialOperations:
    """Test spatial transformation operations."""

    def test_move_translates_bead_xyz(self):
        """move() translates all bead positions via the SpatialMixin."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)
        cg.def_bead(type="B", x=1, y=0, z=0)

        result = cg.move([5, 10, 15])

        assert result is cg  # Returns self for chaining
        positions_x = cg.beads["x"]
        assert np.allclose(positions_x, [5, 6])

    def test_spatial_operations_chain(self):
        """move/scale chain and return self."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)

        result = cg.move([1, 0, 0]).scale(2.0).move([0, 5, 0])

        assert result is cg
        bead = list(cg.beads)[0]
        assert bead.get("x") != 0 or bead.get("y") != 0


class TestCoarseGrainSystemComposition:
    """Test system composition operations (merge / replicate / select)."""

    def test_merge_two_coarsegrain(self):
        """``+=`` merges another CoarseGrain in place."""
        cg1 = CoarseGrain()
        cg1.def_bead(type="A", x=0, y=0, z=0)

        cg2 = CoarseGrain()
        cg2.def_bead(type="B", x=10, y=0, z=0)

        result = cg1.__iadd__(cg2)

        assert result is cg1
        assert len(cg1.beads) == 2

    def test_add_creates_new_structure(self):
        """``+`` returns a new merged CoarseGrain, leaving inputs unchanged."""
        cg1 = CoarseGrain()
        cg1.def_bead(type="A", x=0, y=0, z=0)

        cg2 = CoarseGrain()
        cg2.def_bead(type="B", x=10, y=0, z=0)

        cg3 = cg1 + cg2

        assert cg3 is not cg1
        assert cg3 is not cg2
        assert len(cg3.beads) == 2
        assert len(cg1.beads) == 1
        assert len(cg2.beads) == 1

    def test_replicate_creates_n_copies(self):
        """replicate(n, transform) makes n copies, original untouched."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)

        result = cg.replicate(5, lambda mol, i: mol.move([i * 5, 0, 0]))

        assert len(result.beads) == 5
        assert len(cg.beads) == 1  # Original unchanged

    def test_select_returns_new_coarsegrain_with_filtered_beads(self):
        """select(predicate) returns a new CG containing only matching beads."""
        cg = CoarseGrain()
        cg.def_bead(type="A", x=0, y=0, z=0)
        cg.def_bead(type="B", x=1, y=0, z=0)
        cg.def_bead(type="A", x=2, y=0, z=0)

        sub = cg.select(lambda b: b.get("type") == "A")

        assert sub is not cg
        assert isinstance(sub, CoarseGrain)
        assert len(sub.beads) == 2
        # original is preserved (immutability)
        assert len(cg.beads) == 3

    def test_len_returns_bead_count(self):
        """``len(cg)`` is the bead count."""
        cg = CoarseGrain()
        assert len(cg) == 0

        cg.def_bead(type="A")
        assert len(cg) == 1

        cg.def_bead(type="B")
        assert len(cg) == 2

    def test_repr_shows_structure_summary(self):
        """repr summarises bead and bond counts."""
        cg = CoarseGrain()
        cg.def_bead(type="PEO")
        cg.def_bead(type="PMA")

        repr_str = repr(cg)

        assert "CoarseGrain" in repr_str
        assert "2 beads" in repr_str


# ---------------------------------------------------------------------------
# Generality / arbitrary attrs
# ---------------------------------------------------------------------------


class TestGeneralImplementation:
    """Test that implementation is general without hardcoding."""

    def test_arbitrary_bead_types(self):
        """Bead types are free-form labels."""
        cg = CoarseGrain()
        cg.def_bead(type="CustomType1")
        cg.def_bead(type="AnotherType")
        cg.def_bead(type="X")

        assert len(cg.beads) == 3

    def test_arbitrary_bond_attributes(self):
        """CGBonds support arbitrary attributes."""
        cg = CoarseGrain()
        b1 = cg.def_bead(type="A")
        b2 = cg.def_bead(type="B")

        bond = cg.def_cgbond(
            b1, b2, custom_attr="value", strength=100.0, another_field=[1, 2, 3]
        )

        assert bond.get("custom_attr") == "value"
        assert bond.get("strength") == 100.0
        assert bond.get("another_field") == [1, 2, 3]

    def test_arbitrary_bead_attributes(self):
        """Beads support arbitrary attributes."""
        cg = CoarseGrain()
        bead = cg.def_bead(
            type="Custom", mass=50.0, charge=-1.5, metadata={"key": "value"}
        )

        assert bead.get("mass") == 50.0
        assert bead.get("charge") == -1.5
        assert bead.get("metadata") == {"key": "value"}


# ---------------------------------------------------------------------------
# Coverage fill-in — repr branches, edit/select error paths, spatial chaining
# ---------------------------------------------------------------------------


class TestBeadReprFallbacks:
    """Bead.__repr__ branches when ``type`` is absent."""

    def test_repr_uses_name_when_type_missing(self):
        bead = Bead(name="POPC")
        assert "POPC" in repr(bead)
        assert "Bead" in repr(bead)

    def test_repr_falls_back_to_id_when_type_and_name_missing(self):
        bead = Bead()
        rep = repr(bead)
        assert "Bead" in rep
        assert str(id(bead)) in rep


class TestCoarseGrainPostInit:
    """Subclass __post_init__ template hook is honoured (parallel to Atomistic)."""

    def test_post_init_called_on_subclass(self):
        captured: dict = {}

        class Sub(CoarseGrain):
            def __post_init__(self, **props):
                captured["props"] = props

        Sub(name="lipid", model="martini3")
        assert captured["props"] == {"name": "lipid", "model": "martini3"}


class TestCoarseGrainReprManyTypes:
    """__repr__ collapses to '<N types>' beyond five distinct bead types."""

    def test_repr_collapses_when_more_than_five_types(self):
        cg = CoarseGrain()
        for t in "ABCDEFG":
            cg.def_bead(type=t)
        rep = repr(cg)
        assert "7 types" in rep


class TestCoarseGrainDeletion:
    """del_bead / del_cgbond delegate to the underlying Struct."""

    def test_del_bead_removes_bead(self):
        cg = CoarseGrain()
        a = cg.def_bead(type="A")
        b = cg.def_bead(type="B")
        cg.def_cgbond(a, b)
        cg.del_bead(a)
        assert a not in list(cg.beads)
        # incident bond is removed transitively
        assert len(cg.cgbonds) == 0

    def test_del_cgbond_removes_only_bond(self):
        cg = CoarseGrain()
        a = cg.def_bead(type="A")
        b = cg.def_bead(type="B")
        bond = cg.def_cgbond(a, b)
        cg.del_cgbond(bond)
        assert len(cg.cgbonds) == 0
        assert len(cg.beads) == 2


class TestCoarseGrainBatchFactory:
    """def_cgbonds accepts both 2-tuple and 3-tuple specs."""

    def test_def_cgbonds_two_tuple_no_attrs(self):
        cg = CoarseGrain()
        a, b, c = (cg.def_bead(type=t) for t in "ABC")
        bonds = cg.def_cgbonds([(a, b), (b, c)])
        assert len(bonds) == 2
        assert all(bond.data == {} for bond in bonds)

    def test_def_cgbonds_three_tuple_with_attrs(self):
        cg = CoarseGrain()
        a, b = cg.def_bead(type="A"), cg.def_bead(type="B")
        bonds = cg.def_cgbonds([(a, b, {"k": 100.0})])
        assert bonds[0].get("k") == 100.0


class TestCoarseGrainEditing:
    """rename_type and set_property update both beads and links."""

    def test_rename_type_updates_matching_beads(self):
        cg = CoarseGrain()
        cg.def_bead(type="P4")
        cg.def_bead(type="P4")
        cg.def_bead(type="C1")
        n = cg.rename_type("P4", "Q4")
        assert n == 2
        types = sorted(b["type"] for b in cg.beads)
        assert types == ["C1", "Q4", "Q4"]

    def test_rename_type_on_links(self):
        cg = CoarseGrain()
        a, b = cg.def_bead(type="A"), cg.def_bead(type="B")
        cg.def_cgbond(a, b, type="elastic")
        n = cg.rename_type("elastic", "harmonic", kind=CGBond)
        assert n == 1
        assert list(cg.cgbonds)[0]["type"] == "harmonic"

    def test_set_property_callable_selector(self):
        cg = CoarseGrain()
        cg.def_bead(type="A", x=1.0)
        cg.def_bead(type="A", x=-1.0)
        n = cg.set_property(lambda b: b["x"] > 0, "region", "right")
        assert n == 1
        assert next(b for b in cg.beads if b["x"] > 0)["region"] == "right"

    def test_set_property_rejects_non_callable(self):
        cg = CoarseGrain()
        with pytest.raises(TypeError):
            cg.set_property("string-selector", "k", 1)


class TestCoarseGrainSelectErrors:
    """select rejects non-callable predicates."""

    def test_select_rejects_non_callable(self):
        cg = CoarseGrain()
        with pytest.raises(TypeError):
            cg.select("smarts-string")


class TestCoarseGrainSpatialChaining:
    """rotate / align return self for chaining (parallel to Atomistic)."""

    def test_rotate_returns_self(self):
        cg = CoarseGrain()
        cg.def_bead(type="A", x=1.0, y=0.0, z=0.0)
        result = cg.rotate(axis=[0, 0, 1], angle=np.pi / 2)
        assert result is cg

    def test_scale_returns_self(self):
        cg = CoarseGrain()
        cg.def_bead(type="A", x=1.0, y=0.0, z=0.0)
        result = cg.scale(2.0)
        assert result is cg

    def test_align_returns_self(self):
        cg = CoarseGrain()
        a = cg.def_bead(type="A", x=0.0, y=0.0, z=0.0)
        b = cg.def_bead(type="B", x=1.0, y=0.0, z=0.0)
        result = cg.align(a, b, a_dir=[1, 0, 0], b_dir=[0, 1, 0])
        assert result is cg


class TestCoarseGrainToFrame:
    """to_frame mirrors Atomistic.to_frame."""

    def test_to_frame_produces_beads_block(self):
        cg = CoarseGrain()
        cg.def_bead(type="P4", x=0.0, y=0.0, z=0.0, charge=-1.0)
        cg.def_bead(type="C1", x=4.7, y=0.0, z=0.0, charge=0.0)

        frame = cg.to_frame()
        beads = frame["beads"]
        assert len(beads["type"]) == 2
        assert list(beads["type"]) == ["P4", "C1"]
        assert beads["x"][0] == 0.0
        assert beads["x"][1] == 4.7

    def test_to_frame_produces_cgbonds_block_with_indices(self):
        cg = CoarseGrain()
        a = cg.def_bead(type="A", x=0.0, y=0.0, z=0.0)
        b = cg.def_bead(type="B", x=1.0, y=0.0, z=0.0)
        c = cg.def_bead(type="C", x=2.0, y=0.0, z=0.0)
        cg.def_cgbond(a, b, k=120.0)
        cg.def_cgbond(b, c, k=80.0)

        frame = cg.to_frame()
        cgbonds = frame["cgbonds"]
        assert list(cgbonds["ibead"]) == [0, 1]
        assert list(cgbonds["jbead"]) == [1, 2]
        assert list(cgbonds["k"]) == [120.0, 80.0]

    def test_to_frame_omits_cgbonds_block_when_empty(self):
        cg = CoarseGrain()
        cg.def_bead(type="A")
        frame = cg.to_frame()
        with pytest.raises(KeyError):
            frame["cgbonds"]

    def test_to_frame_field_whitelist(self):
        cg = CoarseGrain()
        ato = Atomistic()
        a = ato.def_atom(symbol="C")
        cg.def_bead(type="P4", x=1.0, y=2.0, z=3.0, atoms=(a,))

        frame = cg.to_frame(bead_fields=["x", "y", "z", "type"])
        cols = set(frame["beads"].keys())
        assert cols == {"x", "y", "z", "type"}
        # atoms key was excluded
        assert "atoms" not in cols

    def test_to_frame_drops_non_numpy_atoms_key(self):
        cg = CoarseGrain()
        ato = Atomistic()
        a = ato.def_atom(symbol="C")
        b = ato.def_atom(symbol="H")
        cg.def_bead(type="P4", atoms=(a, b))

        frame = cg.to_frame()
        # numpy-only Store: the ragged ``atoms`` mapping (tuple of Atom handles
        # per bead) has no numpy representation and is dropped from the numeric
        # frame — the bead→atom mapping lives on the CoarseGrain struct.
        assert "atoms" not in frame["beads"].keys()
        assert list(frame["beads"]["type"]) == ["P4"]

    def test_cross_world_bond_endpoint_rejected(self):
        """Handle views are world-local: a bond over beads from another world
        cannot be registered (molrs rejects the foreign node handles up front),
        so the cross-world reference fails fast rather than surfacing later in
        ``to_frame``."""
        cg = CoarseGrain()
        a = cg.def_bead(type="A")
        b = cg.def_bead(type="B")
        bond = CGBond(a, b)
        cg2 = CoarseGrain()

        with pytest.raises(ValueError):
            cg2.add_cgbond(bond)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
