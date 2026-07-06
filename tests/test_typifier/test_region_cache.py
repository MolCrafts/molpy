"""Region-scoped typing (:func:`typify_region`) + the hash-keyed
:class:`RetypeCache` that dedups identical polymer junctions.

Covers the ``incremental-typify-02-cache`` acceptance contract:

- **ac-001** — a region's non-boundary (interior) atoms get the SAME types as the
  whole-graph ``typify`` assigns them; ``RegionTypes`` is canonical-order keyed,
  frozen, and holds no live ``Entity`` references.
- **ac-002** — two isomorphic regions call the underlying atom typifier ONCE (the
  second is a hash hit confirmed by ``is_isomorphic``); write-back lines the
  cached types up via canonical order + ``entity_map``.
- **ac-003** — retyping ``N`` identical junctions costs O(#distinct), not O(N):
  the atom-typifier call count stays flat as ``N`` grows.
- **ac-004** — a product typed through the region+cache path in ``Reacter.run``
  has the same atom types as the whole-graph path; a typifier without
  ``typify_region`` falls back cleanly.
"""

from __future__ import annotations

import dataclasses

import pytest

from molpy.core.affected_region import AffectedRegion, region_radius
from molpy.core.atomistic import Atomistic
from molpy.io.forcefield.xml import read_oplsaa_forcefield
from molpy.reacter import (
    Reacter,
    find_port,
    form_single_bond,
    select_hydrogens,
    select_self,
)
from molpy.typifier import OplsTypifier
from molpy.typifier.cache import RetypeCache
from molpy.typifier.region import RegionTypes, typify_region


# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def opls() -> OplsTypifier:
    ff = read_oplsaa_forcefield("oplsaa.xml")
    return OplsTypifier(ff, strict_typing=True)


def _alkane(n: int) -> Atomistic:
    """Linear CnH(2n+2) with explicit hydrogens, no coordinates."""
    s = Atomistic()
    carbons = [s.def_atom({"element": "C", "symbol": "C"}) for _ in range(n)]
    for i in range(n - 1):
        s.def_bond(carbons[i], carbons[i + 1])
    for i, c in enumerate(carbons):
        for _ in range(2 if 0 < i < n - 1 else 3):
            s.def_bond(c, s.def_atom({"element": "H", "symbol": "H"}))
    return s.get_topo(gen_angle=True, gen_dihe=True)


def _mid_region(struct: Atomistic, typifier: OplsTypifier) -> AffectedRegion:
    """Region around the central C-C bond of an alkane."""
    carbons = [a for a in struct.atoms if a.get("element") == "C"]
    mid = len(carbons) // 2
    seeds = [carbons[mid - 1], carbons[mid]]
    return AffectedRegion._from(struct, seeds, region_radius(typifier))


class _CallCounter:
    """Wrap ``obj.method`` with an invocation counter (restore via ``stop``)."""

    def __init__(self, obj: object, method: str) -> None:
        self.count = 0
        self._obj = obj
        self._method = method
        self._orig = getattr(obj, method)

        def wrapper(*args: object, **kwargs: object) -> object:
            self.count += 1
            return self._orig(*args, **kwargs)

        setattr(obj, method, wrapper)

    def stop(self) -> None:
        setattr(self._obj, self._method, self._orig)


def _ethyl(port: str) -> Atomistic:
    """CH3-CH2- fragment whose terminal carbon carries ``port``."""
    s = Atomistic()
    c1 = s.def_atom({"element": "C", "symbol": "C"})
    c2 = s.def_atom({"element": "C", "symbol": "C"})
    s.def_bond(c1, c2)
    for _ in range(3):
        s.def_bond(c1, s.def_atom({"element": "H", "symbol": "H"}))
    for _ in range(3):
        s.def_bond(c2, s.def_atom({"element": "H", "symbol": "H"}))
    c2["port"] = port
    return s


def _couple(typifier: OplsTypifier | None):
    reacter = Reacter(
        name="C-C_coupling",
        anchor_selector_left=select_self,
        anchor_selector_right=select_self,
        leaving_selector_left=select_hydrogens(1),
        leaving_selector_right=select_hydrogens(1),
        bond_former=form_single_bond,
    )
    left = _ethyl(">")
    right = _ethyl("<")
    return reacter.run(
        left,
        right,
        port_atom_L=find_port(left, ">"),
        port_atom_R=find_port(right, "<"),
        typifier=typifier,
    )


# --------------------------------------------------------------------------
# ac-001 — region interior types == whole-graph types
# --------------------------------------------------------------------------
def test_region_interior_types_match_whole(opls: OplsTypifier) -> None:
    struct = _alkane(16)
    whole = opls.typify(struct)
    handle_to_type = {
        pa.handle: wa.get("type")
        for pa, wa in zip(struct.atoms, whole.atoms, strict=True)
    }

    region = _mid_region(struct, opls)
    assert len(region.boundary) > 0, "test needs a real (truncated) boundary shell"

    region_types = typify_region(opls, region)
    canon = region.canonical_order()
    handle_to_parent = {r.handle: p for r, p in region.entity_map.items()}

    boundary_handles = {b.handle for b in region.boundary}
    checked = 0
    for canon_index, info in region_types.atoms:
        region_handle = canon[canon_index]
        assert region_handle not in boundary_handles, "boundary must not be stored"
        parent = handle_to_parent[region_handle]
        assert info.type == handle_to_type[parent.handle]
        checked += 1
    assert checked == len(list(region.atoms)) - len(region.boundary)


def test_region_types_is_frozen_and_holds_no_entities(opls: OplsTypifier) -> None:
    region = _mid_region(_alkane(16), opls)
    region_types = typify_region(opls, region)

    assert dataclasses.is_dataclass(RegionTypes)
    assert RegionTypes.__dataclass_params__.frozen is True

    for canon_index, info in region_types.atoms:
        assert isinstance(canon_index, int)
        assert isinstance(info.type, str)
        for key, value in info.params:
            assert isinstance(key, str)
            assert isinstance(value, (str, int, float, bool))
    # frozen dataclass with only scalar/tuple fields must be hashable
    assert isinstance(hash(region_types), int)


# --------------------------------------------------------------------------
# ac-002 — identical junctions type once; write-back via canonical order
# --------------------------------------------------------------------------
def test_cache_hit_types_identical_junction_once(opls: OplsTypifier) -> None:
    region_a = _mid_region(_alkane(16), opls)
    region_b = _mid_region(_alkane(16), opls)
    assert hash(region_a) == hash(region_b)
    assert region_a == region_b  # is_isomorphic confirm

    spy = _CallCounter(opls.atom_typifier, "typify")
    try:
        cache = RetypeCache(opls)
        types_a = cache.retype(region_a)
        types_b = cache.retype(region_b)
    finally:
        spy.stop()

    assert spy.count == 1, "second identical junction must be a cache hit"
    assert types_b is types_a


def test_write_back_maps_cached_types_onto_parent(opls: OplsTypifier) -> None:
    # Types come from region_a; applied onto region_b's DIFFERENT parent.
    region_a = _mid_region(_alkane(16), opls)
    parent_b = _alkane(16)
    region_b = _mid_region(parent_b, opls)

    whole_b = opls.typify(parent_b)
    handle_to_type = {
        pa.handle: wa.get("type")
        for pa, wa in zip(parent_b.atoms, whole_b.atoms, strict=True)
    }

    cache = RetypeCache(opls)
    types_a = cache.retype(region_a)
    # region_b is a cache hit -> cached types_a applied via region_b's own order
    cached = cache.retype(region_b)
    assert cached is types_a
    cache.apply(cached, region_b)

    boundary = {b.handle for b in region_b.boundary}
    for region_atom, parent_atom in region_b.entity_map.items():
        if region_atom.handle in boundary:
            continue
        assert parent_atom.get("type") == handle_to_type[parent_atom.handle]


# --------------------------------------------------------------------------
# ac-003 — growth cost is O(#distinct), not O(N)
# --------------------------------------------------------------------------
def test_growth_cost_bounded_by_distinct_junctions(opls: OplsTypifier) -> None:
    spy = _CallCounter(opls.atom_typifier, "typify")
    try:
        cache = RetypeCache(opls)
        # N identical junctions -> exactly one underlying typing.
        for _ in range(16):
            cache.retype(_mid_region(_alkane(16), opls))
        assert spy.count == 1

        # A structurally-distinct junction adds exactly one more.
        distinct = _mid_region(_alkane(6), opls)
        assert hash(distinct) != hash(_mid_region(_alkane(16), opls))
        for _ in range(8):
            cache.retype(distinct)
        assert spy.count == 2
    finally:
        spy.stop()


# --------------------------------------------------------------------------
# ac-004 — region+cache product == whole-graph path; clean fallback
# --------------------------------------------------------------------------
def test_reacter_region_path_matches_whole_graph(opls: OplsTypifier) -> None:
    result = _couple(opls)
    product = result.product

    product_types = [a.get("type") for a in product.atoms]
    assert all(t is not None for t in product_types)

    baseline = opls.typify(product.get_topo(gen_angle=True, gen_dihe=True))
    baseline_types = [a.get("type") for a in baseline.atoms]
    assert product_types == baseline_types


def test_reacter_falls_back_without_typify_region() -> None:
    # A composite typifier lacking ``typify_region`` must use the old whole-graph
    # path unchanged and still fully type the product.
    from molpy.typifier.base import TypifierBase

    class StubAtom(TypifierBase):
        def __init__(self) -> None:
            self.ff = None  # type: ignore[assignment]
            self.strict = False

        def typify(self, elem: Atomistic) -> Atomistic:
            new = elem.copy()
            for atom in new.atoms:
                atom.data["type"] = f"type_{atom.get('symbol', 'X')}"
            return new

    class StubComposite(TypifierBase):
        def __init__(self) -> None:
            self.ff = None  # type: ignore[assignment]
            self.strict = False
            self.atom_typifier = StubAtom()

        def typify(self, elem: Atomistic) -> Atomistic:
            raise NotImplementedError

    assert not hasattr(StubComposite(), "typify_region")
    result = _couple(StubComposite())
    for atom in result.product.atoms:
        assert atom.get("type") is not None
