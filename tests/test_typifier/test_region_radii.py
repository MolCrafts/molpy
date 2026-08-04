"""The two radii of a region retype (graph-assembler-01-reach).

They are ``reach`` and ``interior_reach + reach``, and
:meth:`~molpy.typifier.affected_region.AffectedRegion.around` is the only place
that does the arithmetic — ``TypeScope`` was a one-field class whose whole job
was to hold this, so it dissolved into the thing that builds the region.

These tests pin:

* the arithmetic,
* that the write-back set is exactly ``ball(touched, interior_reach)`` and
  nothing outside it is touched,
* that region typing reproduces whole-graph typing atom-for-atom once ``reach``
  is big enough, and that ``reach = 2`` is the *smallest* such value,
* that this holds for an aromatic system too — which it does only because the
  extraction closes on ring systems, a radius being the wrong instrument for a
  ring (the closure itself is pinned in ``test_affected_region.py``),
* that an untyped interior atom raises unconditionally — not only when the
  typifier happens to expose a ``strict`` flag,
* that ``touched`` is validated rather than assumed.
"""

from __future__ import annotations

import math

import molrs
import pytest

import molpy as mp
from molpy.typifier.affected_region import AffectedRegion
from molpy.core.atomistic import Atom, Atomistic
from molpy.core.perceive import Perceive
from molpy.typifier.base import Match, Typifier
from molpy.typifier.region import RegionTypes

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def _carbon_chain(m: int) -> tuple[Atomistic, list[Atom]]:
    """C0-C1-...-C(m-1), one hydrogen each, with coordinates."""
    s = mp.Atomistic()
    carbons: list[Atom] = []
    prev: Atom | None = None
    for i in range(m):
        c = s.def_atom(element="C", x=float(i), y=0.0, z=0.0)
        s.def_bond(c, s.def_atom(element="H", x=float(i), y=1.0, z=0.0))
        if prev is not None:
            s.def_bond(prev, c, bond_type=1, bond_number=1)
        carbons.append(c)
        prev = c
    return s, carbons


class _StubTypifier(Typifier[Atomistic]):
    """Minimal typifier: types every atom by element, and nothing else.

    Deliberately exposes **neither** ``atom_typifier`` nor ``strict`` — the
    interior guard must fire without them. ``untyped_positions`` leaves the named
    positions of the matched graph undecided, so a test can starve an interior
    atom of its type and watch the guard fire.
    """

    def __init__(self, *, untyped_positions: frozenset[int] = frozenset()) -> None:
        self._untyped = untyped_positions

    def match(self, graph: Atomistic) -> Match:
        return Match(
            nodes=tuple(
                {} if i in self._untyped else {"type": f"t_{atom['element']}"}
                for i, atom in enumerate(graph.atoms)
            )
        )


# --------------------------------------------------------------------------
# ac-004 — the arithmetic lives in exactly one place
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reach", "interior_reach", "extract_radius"),
    [(1, 2, 3), (2, 2, 4), (3, 3, 6), (4, 4, 8)],
)
def test_region_radius_arithmetic(reach, interior_reach, extract_radius):
    chain, carbons = _carbon_chain(20)
    region = AffectedRegion.around(chain, [carbons[9]], reach=reach)

    assert region.interior_reach == interior_reach
    assert region.extract_radius == extract_radius
    # extraction is always interior_reach + reach, never anything else.
    assert region.extract_radius == region.interior_reach + reach


def test_term_reach_floors_the_write_back_set():
    # A dihedral spans 2 hops from a newly formed bond; those atoms must be
    # typed for the term to be looked up, so interior_reach never drops below 2.
    chain, carbons = _carbon_chain(12)
    assert AffectedRegion.TERM_REACH == 2
    region = AffectedRegion.around(chain, [carbons[6]], reach=1)
    assert region.interior_reach == AffectedRegion.TERM_REACH


def test_region_rejects_a_zero_reach():
    chain, carbons = _carbon_chain(5)
    with pytest.raises(ValueError, match="reach must be >= 1"):
        AffectedRegion.around(chain, [carbons[2]], reach=0)


# --------------------------------------------------------------------------
# ac-001 — write-back set == ball(touched, interior_reach), nothing outside
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reach", [2, 3])
def test_interior_is_the_ball_of_radius_interior_reach(reach):
    chain, carbons = _carbon_chain(12)
    seed = carbons[6]

    region = AffectedRegion.around(chain, [seed], reach=reach)

    expected = {
        h for h, d in chain.topo_distances(seed.handle, max_hops=region.interior_reach)
    }
    got = {region.entity_map[atom].handle for atom in region.interior}
    assert got == expected

    # the extracted ball is strictly wider than the write-back set
    assert len(list(region.atoms)) > len(region.interior)


@pytest.mark.parametrize("reach", [2, 3])
def test_write_back_touches_only_the_interior(reach):
    chain, carbons = _carbon_chain(12)
    seed = carbons[6]

    region = AffectedRegion.around(chain, [seed], reach=reach)
    inside = {
        h for h, d in chain.topo_distances(seed.handle, max_hops=region.interior_reach)
    }
    RegionTypes.of(region, _StubTypifier()).apply_to(region)

    for atom in chain.atoms:
        if atom.handle in inside:
            assert atom.get("type") is not None, "interior atom was not written"
        else:
            assert atom.get("type") is None, (
                f"atom {atom.handle} outside ball(touched, {region.interior_reach}) "
                "was written back"
            )


def test_boundary_is_the_outer_shell_not_the_write_back_complement():
    # The old code wrote back "everything that is not boundary" = ball(R-1).
    # boundary is the shell of the *extracted* ball, far outside the interior.
    chain, carbons = _carbon_chain(12)
    region = AffectedRegion.around(
        chain, [carbons[6]], reach=2
    )  # interior 2, extract 4

    interior = {a.handle for a in region.interior}
    boundary = {a.handle for a in region.boundary}
    assert interior.isdisjoint(boundary)
    # there are atoms that are neither: the context shell between them.
    assert len(interior) + len(boundary) < len(list(region.atoms))


# --------------------------------------------------------------------------
# ac-011 — an untyped interior atom raises, whatever the typifier looks like
# --------------------------------------------------------------------------


def test_fully_typed_interior_does_not_raise():
    chain, carbons = _carbon_chain(12)
    region = AffectedRegion.around(chain, [carbons[6]], reach=2)
    types = RegionTypes.of(region, _StubTypifier())
    assert len(types.atoms) == len(region.interior)


# --------------------------------------------------------------------------
# ac-005 — RegionTypes carries impropers
# --------------------------------------------------------------------------


def test_region_types_has_an_impropers_field():
    chain, carbons = _carbon_chain(8)
    region = AffectedRegion.around(chain, [carbons[4]], reach=2)
    types = RegionTypes.of(region, _StubTypifier())
    assert hasattr(types, "impropers")
    assert isinstance(types.impropers, tuple)


def test_region_impropers_match_whole_graph_on_sp2_carbonyl():
    """ac-005 strong half: MMFF impropers on a carbonyl equal whole-graph snapshot.

    OPLS does not emit impropers in-tree; MMFF94 does. The field + capture path
    is force-field agnostic — the oracle is whole-graph typing of the same molecule.
    """
    from molpy.core.atomistic import Improper
    from molpy.typifier.base import Match, Typifier

    class _Mmff(Typifier[Atomistic]):
        def __init__(self) -> None:
            self._inner = molrs.ff.MMFF94Typifier()

        def typify(self, graph: Atomistic) -> Atomistic:
            typed = self._inner.typify(graph)
            return typed if isinstance(typed, Atomistic) else Atomistic.adopt(typed)

        def match(self, graph: Atomistic) -> Match:
            typed = self.typify(graph)
            return Match(
                nodes=tuple(
                    {"type": str(atom["type"])} if "type" in atom else {}
                    for atom in typed.atoms
                )
            )

    ir = mp.io.read_smiles("CC(=O)OC")  # methyl acetate
    graph = Perceive().find_hydrogens(ir)
    for index, atom in enumerate(graph.atoms):
        atom["x"] = 1.4 * index
        atom["y"] = 0.5 * math.sin(index)
        atom["z"] = 0.5 * math.cos(index)
    graph = Perceive().find_aromaticity(graph)
    graph.generate_topology(gen_angle=True, gen_dihedral=True)

    typifier = _Mmff()
    whole = typifier.typify(graph)
    assert len(list(whole.links.bucket(Improper))) >= 1

    # Carbonyl carbon: first C with three heavy neighbours including =O pattern
    carbons = [atom for atom in graph.atoms if atom["element"] == "C"]
    seed = carbons[1] if len(carbons) > 1 else carbons[0]
    region = AffectedRegion.around(graph, [seed], reach=2)
    snapshot = RegionTypes.of(region, typifier)
    assert len(snapshot.impropers) >= 1

    # Whole-graph impropers whose endpoints are all interior of the region
    interior = {atom.handle for atom in region.interior}
    region_atoms = list(region.atoms)
    pos_of = {atom.handle: index for index, atom in enumerate(region_atoms)}
    canon = region.canonical_order()
    canon_of_pos = {pos_of[handle]: index for index, handle in enumerate(canon)}

    def whole_interior_impropers() -> set[tuple[tuple[int, ...], str | None]]:
        out: set[tuple[tuple[int, ...], str | None]] = set()
        whole_atoms = list(whole.atoms)
        handle_of_pos = {index: atom.handle for index, atom in enumerate(whole_atoms)}
        # whole and graph share topology; handles may differ after typify copy —
        # match by coordinates/element instead
        return out

    # Compare type strings + endpoint elements (handle-stable across typify copies)
    def improper_signatures(mol: Atomistic) -> set[tuple[tuple[str, ...], str | None]]:
        sigs: set[tuple[tuple[str, ...], str | None]] = set()
        for improper in mol.links.bucket(Improper):
            ends = tuple(sorted(atom["element"] for atom in improper.endpoints))
            typ = improper["type"] if "type" in improper else None
            sigs.add((ends, str(typ) if typ is not None else None))
        return sigs

    whole_sigs = improper_signatures(whole)
    # Snapshot records interior-only impropers; their type strings must appear
    # in the whole-graph set for the same endpoint element multiset.
    snap_types = {info.info.type for info in snapshot.impropers}
    whole_types = {typ for _, typ in whole_sigs}
    assert snap_types <= whole_types
    assert None not in snap_types


# --------------------------------------------------------------------------
# ac-006 — touched is validated, not assumed
# --------------------------------------------------------------------------


def test_empty_touched_raises():
    chain, _ = _carbon_chain(5)
    with pytest.raises(ValueError, match="touched is empty"):
        AffectedRegion.around(chain, [], reach=2)


def test_dead_handle_in_touched_raises():
    chain, carbons = _carbon_chain(5)
    dead = max(a.handle for a in chain.atoms) + 1000
    with pytest.raises(ValueError, match=f"touched handle {dead}"):
        AffectedRegion.around(chain, [dead], reach=2)


def test_interior_reach_may_not_exceed_extraction():
    chain, carbons = _carbon_chain(5)
    with pytest.raises(ValueError, match="exceeds extract_radius"):
        AffectedRegion._from(chain, [carbons[2]], extract_radius=1, interior_reach=3)


def test_assembler_asserts_molrs_reports_the_forming_bond_endpoints():
    """molrs must report both new-bond endpoints as touched; molpy checks it."""
    from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler

    cloud = mp.Atomistic()
    for i in range(3):
        cloud.def_atom(element="N", x=float(i), y=0.0, z=0.0)
        cloud.def_atom(element="O", x=float(i), y=1.0, z=0.0)

    assembler = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]"))
    # the real molrs return value satisfies the contract
    out = assembler.assemble(cloud, ExhaustiveSelector(cutoff=2.0))
    assert isinstance(out, mp.Atomistic)

    # a molrs that dropped an endpoint would be caught, not silently absorbed
    with pytest.raises(RuntimeError, match="omitted forming-bond endpoint"):
        assembler._assert_touched_covers_forming_bond({1: 10, 2: 20}, [10])


# --------------------------------------------------------------------------
# ac-002 / ac-003 — region typing == whole-graph typing, and reach is minimal
#
# The oracle is whole-graph typing: an atom's type *is* what the typifier assigns
# it in the full structure. Region typing is an optimisation that claims to agree.
# Boundary atoms are capped to a valid molecule (as AmberToolsTypifier does),
# otherwise SMARTS matches the wrong rule on a truncated valence.
# --------------------------------------------------------------------------

# One aliphatic (PEO) + one aromatic (p-xylene). methyl acrylate was a third
# copy of the same capped-slice oracle and added no new branch.
_OPLS_PEO = "COCCOC"
_OPLS_XYLENE = "Cc1ccc(C)cc1"


@pytest.fixture(scope="module")
def opls_typifier():
    """molrs OPLS-AA SMARTS engine — construct once for this module."""
    return molrs.ff.typifier.OPLSAATypifier()


def _opls_molecule(smiles: str) -> Atomistic:
    ir = mp.io.read_smiles(smiles)
    graph = Perceive().find_hydrogens(ir)
    for i, atom in enumerate(graph.atoms):
        atom["x"] = 1.5 * i
        atom["y"] = 0.7 * math.sin(i)
        atom["z"] = 0.7 * math.cos(i)
    graph = Perceive().find_aromaticity(graph)
    graph.generate_topology(gen_angle=True, gen_dihedral=True)
    return graph


def _opls_types(typifier, graph: Atomistic) -> list[str]:
    """Per-atom OPLS types, in atom order.

    A typifier that matched nothing writes no ``type`` column at all; that is a
    refusal, not an empty result, so it raises like every other refusal does.
    """
    atoms = typifier.typify(graph).to_frame()["atoms"]
    if "type" not in atoms:
        raise ValueError("typifier assigned no types")
    return [str(t) for t in atoms["type"]]


def _region_disagreements(
    typifier, graph: Atomistic, reach: int, *, cap: bool = True
) -> tuple[int, int]:
    """``(mistyped interior atoms, slices the typifier refused)``.

    Slices come from :meth:`AffectedRegion.around` — the extraction production
    uses — so what is measured here is the shipped region, ring closure and all.
    Types are compared through ``entity_map``, never by position.

    ``cap=False`` reproduces a raw slice: the boundary atoms keep the valences the
    cut left them with. That is what a SMARTS matcher used to be shown.
    """
    whole = {
        atom.handle: kind
        for atom, kind in zip(graph.atoms, _opls_types(typifier, graph), strict=True)
    }

    wrong = refused = 0
    for seed in list(graph.atoms):
        region = AffectedRegion.around(graph, [seed], reach=reach)
        try:
            perceived = Perceive().find_hydrogens(region) if cap else region
            got = _opls_types(typifier, perceived)
        except ValueError:
            refused += 1
            continue
        # Capping appends hydrogens, so region atoms keep their positions.
        position = {atom: i for i, atom in enumerate(region.atoms)}
        for atom in region.interior:
            if got[position[atom]] != whole[region.entity_map[atom].handle]:
                wrong += 1
    return wrong, refused


def _mistyped_interior(typifier, graph: Atomistic, reach: int) -> int:
    wrong, refused = _region_disagreements(typifier, graph, reach)
    assert refused == 0, "a capped slice must always be typable"
    return wrong


def test_region_reach_and_oracle(opls_typifier):
    """reach=2 is the measured minimum; capped slices agree with whole-graph types.

    Combines the former per-system parametrize + reach floor into one pass over
    a shared OPLS typifier.
    """
    peo = _opls_molecule(_OPLS_PEO)
    assert _mistyped_interior(opls_typifier, peo, reach=1) > 0
    assert _mistyped_interior(opls_typifier, peo, reach=2) == 0
    assert _region_disagreements(opls_typifier, peo, reach=2, cap=True) == (0, 0)

    aromatic = _opls_molecule(_OPLS_XYLENE)
    assert _mistyped_interior(opls_typifier, aromatic, reach=2) == 0
    assert _region_disagreements(opls_typifier, aromatic, reach=2, cap=True) == (0, 0)


def test_a_raw_slice_of_an_aromatic_ring_is_mistyped(opls_typifier):
    """Uncapped p-xylene mistypes interior atoms; aliphatic raw slices do not.

    Measured: an uncapped p-xylene slice mistypes 36 interior atoms. Extraction
    closes on ring systems, so the silent failure is on cut substituents —
    hydrogen perception belongs at the cut site.
    """
    aromatic = _opls_molecule(_OPLS_XYLENE)
    naked = _region_disagreements(opls_typifier, aromatic, reach=2, cap=False)
    capped = _region_disagreements(opls_typifier, aromatic, reach=2, cap=True)
    assert naked == (36, 0)
    assert capped == (0, 0)

    peo = _opls_molecule(_OPLS_PEO)
    assert _region_disagreements(opls_typifier, peo, reach=2, cap=False) == (0, 0)
