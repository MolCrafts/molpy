"""ac-003 (incremental-typify-03): the ``AmberTools`` region path types an
``AffectedRegion`` through antechamber and caches the result by the region's
structural hash, so a recurring identical junction skips the subprocess; the
GAFF types land on the parent graph via ``region.entity_map``.

The ``Atomistic -> PDB`` writer (``write_antechamber_input_pdb``) is pre-broken
in this environment (an unrelated molrs object-dtype-string-column issue), so the
real end-to-end run is ``@pytest.mark.external``. A non-external test verifies the
plumbing (hash cache keying + entity_map write-back) with the antechamber run
stubbed.
"""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.builder.ambertools import AmberTools
from molpy.core.affected_region import AffectedRegion, region_radius
from molpy.typifier.region import RegionTypes, TypeInfo


def _methane() -> mp.Atomistic:
    """CH4 with explicit hydrogens (no coordinates needed for the region graph)."""
    s = mp.Atomistic()
    c = s.def_atom(element="C", symbol="C", name="C1")
    for i in range(4):
        h = s.def_atom(element="H", symbol="H", name=f"H{i + 1}")
        s.def_bond(c, h)
    return s


def _region(mol: mp.Atomistic) -> AffectedRegion:
    seeds = [a for a in mol.atoms if a.get("element") == "C"][:1]
    return AffectedRegion._from(mol, seeds, region_radius())


def _fake_gaff(region: AffectedRegion) -> RegionTypes:
    """A deterministic per-element GAFF snapshot in the region's canonical order.

    Mirrors how the real ``_region_gaff_types`` keys types by canonical position
    so an isomorphic region can reuse a cached snapshot.
    """
    region_atoms = list(region.atoms)
    canon = region.canonical_order()
    pos_of_handle = {a.handle: i for i, a in enumerate(region_atoms)}
    canon_of_pos = {pos_of_handle[h]: idx for idx, h in enumerate(canon)}
    boundary = {a.handle for a in region.boundary}
    entries: list[tuple[int, TypeInfo]] = []
    for pos, atom in enumerate(region_atoms):
        if atom.handle in boundary:
            continue
        gaff = "c3" if atom.get("element") == "C" else "hc"
        entries.append(
            (canon_of_pos[pos], TypeInfo(type=gaff, params=(("charge", 0.0),)))
        )
    entries.sort(key=lambda entry: entry[0])
    return RegionTypes(atoms=tuple(entries), bonds=(), angles=(), dihedrals=())


class _StubGaff:
    """Counting stub for ``AmberTools._region_gaff_types``."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, region: AffectedRegion, *, net_charge: int, name: str):
        self.calls += 1
        return _fake_gaff(region)


# --------------------------------------------------------------------------
# recurring identical region == cache hit (no second subprocess)
# --------------------------------------------------------------------------
def test_region_cache_hit_skips_second_run(tmp_path, monkeypatch) -> None:
    at = AmberTools(work_dir=tmp_path)
    stub = _StubGaff()
    monkeypatch.setattr(at, "_region_gaff_types", stub)

    region_a = _region(_methane())
    region_b = _region(_methane())  # isomorphic, different atoms
    assert hash(region_a) == hash(region_b) and region_a == region_b

    at.parameterize_region(region_a, net_charge=0, name="JUN")
    at.parameterize_region(region_b, net_charge=0, name="JUN")

    assert stub.calls == 1, "recurring identical region must be a cache hit"


def test_region_types_written_onto_parent_via_entity_map(tmp_path, monkeypatch) -> None:
    at = AmberTools(work_dir=tmp_path)
    monkeypatch.setattr(at, "_region_gaff_types", _StubGaff())

    parent = _methane()
    region = _region(parent)
    at.parameterize_region(region, net_charge=0, name="JUN")

    by_element = {a.get("element"): a.get("type") for a in parent.atoms}
    assert by_element["C"] == "c3"
    assert by_element["H"] == "hc"
    # every non-boundary parent atom is typed
    boundary = {a.handle for a in region.boundary}
    for region_atom, parent_atom in region.entity_map.items():
        if region_atom.handle in boundary:
            continue
        assert parent_atom.get("type") is not None


def test_cache_hit_writes_onto_second_regions_parent(tmp_path, monkeypatch) -> None:
    at = AmberTools(work_dir=tmp_path)
    monkeypatch.setattr(at, "_region_gaff_types", _StubGaff())

    parent_a = _methane()
    parent_b = _methane()
    region_a = _region(parent_a)
    region_b = _region(parent_b)

    at.parameterize_region(region_a, net_charge=0, name="JUN")
    at.parameterize_region(region_b, net_charge=0, name="JUN")  # cache hit

    # Cached snapshot lines up onto region_b's own atoms via canonical order.
    for parent in (parent_a, parent_b):
        elements = {a.get("element"): a.get("type") for a in parent.atoms}
        assert elements["C"] == "c3"
        assert elements["H"] == "hc"


# --------------------------------------------------------------------------
# whole-molecule path unchanged (region path is additive)
# --------------------------------------------------------------------------
def test_parameterize_still_present(tmp_path) -> None:
    at = AmberTools(work_dir=tmp_path)
    assert hasattr(at, "parameterize")  # whole-molecule path untouched
    assert hasattr(at, "parameterize_region")


# --------------------------------------------------------------------------
# real antechamber end-to-end (also exercises the pre-broken region PDB writer)
# --------------------------------------------------------------------------
@pytest.mark.external
def test_parameterize_region_end_to_end(tmp_path) -> None:
    at = AmberTools(work_dir=tmp_path)
    parent = _methane()
    for i, atom in enumerate(parent.atoms, start=1):
        atom["x"] = float(i) * 0.5
        atom["y"] = 0.0
        atom["z"] = 0.0
    region = _region(parent)

    at.parameterize_region(region, net_charge=0, name="JUN")

    boundary = {a.handle for a in region.boundary}
    for region_atom, parent_atom in region.entity_map.items():
        if region_atom.handle in boundary:
            continue
        assert parent_atom.get("type") is not None
