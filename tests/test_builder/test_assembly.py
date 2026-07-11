"""One assembly kernel, one variation point (graph-assembler-02-kernel).

Growing a chain and crosslinking a melt are the same algorithm with a different
pairing rule. These tests pin that literally: the same ``GraphAssembler`` runs
both, and the polymer builder's rule is provably the "explicit pairs" rule.
"""

from __future__ import annotations


import pytest

import molpy as mp
import molpy.core.atomistic as atomistic_module
from molpy.builder.assembly import (
    ExhaustiveSelector,
    ExplicitPairSelector,
    GraphAssembler,
    MatchContext,
    MonomerLibrary,
    PolymerBuilder,
    RandomSelector,
    SpacingSelector,
    TopologySelector,
)
from molpy.builder.assembly._placer import Placer
from molpy.core import fields
from molpy.parser.smiles import parse_cgsmiles
from molpy.typifier.region import RegionTypes
from molpy.typifier.base import Match, Typifier

ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"
NO_PLUS_O = "[N:1].[O:2]>>[N:1][O:2]"


class _ElementTypifier(Typifier):
    """Types every atom by element. Implements ``match`` and nothing else."""

    def match(self, graph) -> Match:
        return Match(
            nodes=tuple(
                {fields.TYPE.key: f"t_{atom[fields.ELEMENT]}"} for atom in graph.atoms
            )
        )


def _eo(*, typed: bool = True, charge: float | None = None):
    """Ethylene glycol: a capped, real molecule with two hydroxyl sites."""
    s = mp.Atomistic()
    heavy = [
        s.def_atom(element=e, x=float(i), y=0.0, z=0.0) for i, e in enumerate("OCCO")
    ]
    for a, b in zip(heavy, heavy[1:], strict=False):
        s.def_bond(a, b)
    for oxygen in (heavy[0], heavy[3]):
        s.def_bond(oxygen, s.def_atom(element="H", x=oxygen["x"], y=1.0, z=0.0))
    heavy[0][fields.SITE] = "a"
    heavy[3][fields.SITE] = "b"
    s.generate_topology(gen_angle=True, gen_dihedral=True)
    if typed:
        for atom in s.atoms:
            atom[fields.TYPE] = f"t_{atom[fields.ELEMENT]}"
    if charge is not None:
        for atom in s.atoms:
            atom[fields.CHARGE] = charge
    return s


def _no_cloud(n: int = 3):
    cloud = mp.Atomistic()
    for i in range(n):
        cloud.def_atom(element="N", x=float(i), y=0.0, z=0.0)
        cloud.def_atom(element="O", x=float(i), y=1.0, z=0.0)
    return cloud


def _builder(*, typifier=None, reach=None, **kwargs) -> PolymerBuilder:
    if typifier is not None and reach is None:
        reach = 2
    return PolymerBuilder(
        MonomerLibrary({"EO": _eo()}),
        mp.Reaction(ETHER),
        typifier=typifier,
        reach=reach,
        **kwargs,
    )


# --------------------------------------------------------------------------
# ac-002 — one kernel; crosslinking needs no class of its own
# --------------------------------------------------------------------------


def test_crosslinking_is_the_kernel_plus_a_proximity_selector():
    gel = GraphAssembler(mp.Reaction(NO_PLUS_O)).assemble(
        _no_cloud(), ExhaustiveSelector(cutoff=2.0)
    )
    assert len(list(gel.bonds)) == 3


def test_polymer_builder_is_a_graph_assembler():
    assert issubclass(PolymerBuilder, GraphAssembler)
    assert isinstance(_builder(), GraphAssembler)


def test_assemble_never_mutates_its_input():
    cloud = _no_cloud()
    GraphAssembler(mp.Reaction(NO_PLUS_O)).assemble(
        cloud, ExhaustiveSelector(cutoff=2.0)
    )
    assert len(list(cloud.bonds)) == 0


# --------------------------------------------------------------------------
# ac-003 — the merge, proved: the polymer rule IS explicit pairing
# --------------------------------------------------------------------------


def _context(builder: PolymerBuilder, topology):
    world = builder.library.expand(topology)
    labels = builder._labels(world)
    return MatchContext(
        world=world,
        occurrences=builder._match(world, labels),
        map_a=builder._map_a,
        map_b=builder._map_b,
        comp_a=builder._comp_a,
        comp_b=builder._comp_b,
    )


def test_topology_selection_equals_explicit_pair_selection():
    builder = _builder()
    topology = parse_cgsmiles("{[#EO]|5}").base_graph
    context = _context(builder, topology)

    by_topology = list(TopologySelector(topology).select(context))
    pairs = [(b[context.map_a], b[context.map_b]) for b in by_topology]
    by_pairs = list(ExplicitPairSelector(pairs).select(context))

    assert [set(b.values()) for b in by_topology] == [set(b.values()) for b in by_pairs]


def test_one_assembler_instance_runs_both_jobs():
    """The same kernel object, two selectors, two different jobs."""
    assembler = GraphAssembler(mp.Reaction(NO_PLUS_O))
    exhaustive = assembler.assemble(_no_cloud(3), ExhaustiveSelector(cutoff=2.0))
    random = assembler.assemble(
        _no_cloud(3), RandomSelector(conversion=1.0, seed=1, cutoff=2.0)
    )
    assert len(list(exhaustive.bonds)) == len(list(random.bonds)) == 3


# --------------------------------------------------------------------------
# ac-004 — matching is the kernel's (once, O(N)); pairing is the selector's
# --------------------------------------------------------------------------


def test_selector_receives_the_matches_and_never_rescans():
    builder = _builder()
    topology = parse_cgsmiles("{[#EO]|4}").base_graph
    context = _context(builder, topology)
    # one occurrence list per reactant component, already matched
    assert len(context.occurrences) == 2
    assert all(context.occurrences)


def test_selector_is_an_assemble_argument_not_a_constructor_argument():
    import inspect

    assert "selector" in inspect.signature(GraphAssembler.assemble).parameters
    assert "selector" not in inspect.signature(GraphAssembler.__init__).parameters


# --------------------------------------------------------------------------
# ac-005 — the retype cache lives on the assembler, across assemble() calls
# --------------------------------------------------------------------------


def test_cache_is_shared_across_builds_of_different_lengths():
    builder = _builder(typifier=_ElementTypifier())
    for n in (4, 7, 11):
        builder.build(f"{{[#EO]|{n}}}")
    junctions = (4 - 1) + (7 - 1) + (11 - 1)
    distinct = sum(len(bucket) for bucket in builder._cache._buckets.values())
    assert distinct < junctions
    # the count tracks distinct chemical environments, not the number of bonds
    assert distinct <= 4


# --------------------------------------------------------------------------
# ac-006 — residues, not build-time markers
# --------------------------------------------------------------------------


def test_expansion_stamps_contiguous_residue_ids_and_names():
    builder = _builder()
    chain = builder.build("{[#EO]|5}")
    ids = sorted({int(a[fields.RES_ID]) for a in chain.atoms})
    names = {str(a[fields.RES_NAME]) for a in chain.atoms}
    assert ids == [1, 2, 3, 4, 5]
    assert names == {"EO"}


def test_library_refuses_a_monomer_with_no_site():
    naked = mp.Atomistic()
    naked.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValueError, match="marks no reaction site"):
        MonomerLibrary({"X": naked})


def test_library_refuses_a_topology_naming_an_unknown_monomer():
    builder = _builder()
    with pytest.raises(ValueError, match="lacks"):
        builder.build("{[#ZZ]|2}")


# --------------------------------------------------------------------------
# ac-007 / ac-008 — placer is an argument; illegal states unrepresentable
# --------------------------------------------------------------------------


def test_placer_is_a_constructor_argument_not_a_subclass():
    import inspect

    assert "placer" in inspect.signature(GraphAssembler.__init__).parameters
    assert isinstance(Placer, type)  # an abstraction, not a builder variant


def test_spacing_and_explicit_pairs_are_separate_classes():
    """No constructor takes two mutually exclusive knobs and prefers one."""
    import inspect

    spacing = inspect.signature(SpacingSelector.__init__).parameters
    explicit = inspect.signature(ExplicitPairSelector.__init__).parameters
    assert "pairs" not in spacing
    assert "spacing" not in explicit


def test_spacing_must_be_positive():
    with pytest.raises(ValueError, match="spacing must be >= 1"):
        SpacingSelector(0)


# --------------------------------------------------------------------------
# ac-010 — no whole-graph topology rebuild on the assembly path
# --------------------------------------------------------------------------


def test_no_whole_graph_generate_topology_on_the_assembled_world(monkeypatch):
    seen: list[object] = []
    original = atomistic_module.Atomistic.generate_topology

    def spy(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(atomistic_module.Atomistic, "generate_topology", spy)
    chain = _builder(typifier=_ElementTypifier()).build("{[#EO]|6}")

    assert not [g for g in seen if g is chain], (
        "the assembled world was rebuilt whole-graph; only regions may be"
    )
    # the junction terms were still created, locally
    assert len(list(chain.angles)) == 19
    assert all(atom.get(fields.TYPE) for atom in chain.atoms)


# --------------------------------------------------------------------------
# ac-011 / ac-012 — construction-time rejection; disjoint bindings
# --------------------------------------------------------------------------


def test_something_that_is_not_a_typifier_is_rejected_at_construction():
    class NotATypifier:
        def typify(self, graph):
            return graph

    with pytest.raises(TypeError, match="is not a molpy.typifier.Typifier"):
        GraphAssembler(mp.Reaction(NO_PLUS_O), typifier=NotATypifier(), reach=2)


def test_a_typifier_without_a_reach_is_rejected_at_construction():
    """The radius belongs to whoever cuts the region; it may never be guessed."""
    with pytest.raises(TypeError, match="reach= is required"):
        GraphAssembler(mp.Reaction(NO_PLUS_O), typifier=_ElementTypifier())


def test_reaction_must_be_a_reaction_not_a_string():
    with pytest.raises(TypeError, match="must be a molpy.Reaction"):
        GraphAssembler(NO_PLUS_O)  # type: ignore[arg-type]


def test_assembler_takes_no_charges_argument():
    import inspect

    assert "charges" not in inspect.signature(GraphAssembler.__init__).parameters


def test_overlapping_bindings_raise():
    assembler = GraphAssembler(mp.Reaction(NO_PLUS_O))
    with pytest.raises(ValueError, match="both name atom"):
        assembler._assert_disjoint([{1: 10, 2: 11}, {1: 10, 2: 12}])


def test_empty_selection_warns_and_returns_the_world_unchanged():
    cloud = _no_cloud(2)
    assembler = GraphAssembler(mp.Reaction(NO_PLUS_O))
    # 0.5 A: the closest N-O pair is 1.0 A apart, so nothing is a candidate.
    # (A cutoff of 1e-6 panics inside molrs' link cell — molrs bug, see
    # .claude/notes/architecture.md; not worked around here, just avoided.)
    with pytest.warns(UserWarning, match="selected no bindings"):
        out = assembler.assemble(cloud, ExhaustiveSelector(cutoff=0.5))
    assert len(list(out.bonds)) == 0


# --------------------------------------------------------------------------
# ac-014 — charge conservation is checked, not restored
# --------------------------------------------------------------------------


def test_unfrozen_template_charges_raise_rather_than_leak_net_charge():
    """The reaction deletes a charged cap -> net charge would drift -> raise."""
    library = MonomerLibrary({"EO": _eo(charge=0.1)})
    builder = PolymerBuilder(library, mp.Reaction(ETHER))
    with pytest.raises(ValueError, match="changed the net charge"):
        builder.build("{[#EO]|3}")


def test_uncharged_graph_skips_the_charge_check():
    chain = _builder().build("{[#EO]|3}")
    assert len(list(chain.bonds)) == 11


# --------------------------------------------------------------------------
# ac-022 / ac-023 — malformed reaction raises; no fake distances
# --------------------------------------------------------------------------


def test_forming_bond_map_absent_from_reactants_raises():
    """The old `_find_component` returned 0, silently pairing the wrong sites."""
    with pytest.raises(ValueError, match="appears in no reactant pattern"):
        GraphAssembler._find_component([{1}, {2}], 7)


def test_distance_is_none_without_coordinates_not_zero():
    from molpy.builder.assembly._proximity import Candidate

    key_none = ExhaustiveSelector._sort_key(Candidate({1: 1}, {2: 2}, None))
    key_real = ExhaustiveSelector._sort_key(Candidate({1: 1}, {2: 2}, 3.0))
    # a missing distance is its own class, never a zero that outranks real ones
    assert key_real < key_none


def test_cutoff_without_coordinates_raises():
    graph = mp.Atomistic()
    graph.def_atom(element="N")
    graph.def_atom(element="O")
    with pytest.raises(ValueError, match="requires atom coordinates"):
        GraphAssembler(mp.Reaction(NO_PLUS_O)).assemble(
            graph, ExhaustiveSelector(cutoff=2.0)
        )


# --------------------------------------------------------------------------
# topology shapes: one monomer definition, many jobs
# --------------------------------------------------------------------------


def test_ring_closure_reuses_the_other_site_of_the_opening_residue():
    ring = _builder().build("{[#EO]1[#EO][#EO]1}")
    # a cycle: as many bonds as atoms
    assert len(list(ring.bonds)) == len(list(ring.atoms))


def test_a_difunctional_monomer_cannot_branch_and_forms_a_chain():
    chain = _builder().build("{[#EO]([#EO])[#EO]}")
    assert len(list(chain.bonds)) == len(list(chain.atoms)) - 1


def test_reproducible_random_network():
    assembler = GraphAssembler(mp.Reaction(NO_PLUS_O))
    a = assembler.assemble(
        _no_cloud(4), RandomSelector(conversion=0.5, seed=7, cutoff=5.0)
    )
    b = assembler.assemble(
        _no_cloud(4), RandomSelector(conversion=0.5, seed=7, cutoff=5.0)
    )
    assert len(list(a.bonds)) == len(list(b.bonds))


# --------------------------------------------------------------------------
# ResiduePlacer — guess the value (bond length), never the identity (element)
# --------------------------------------------------------------------------


def test_placer_lays_residues_out_without_overlap():
    import numpy as np

    from molpy.builder.assembly import ResiduePlacer

    stacked = _builder().build("{[#EO]|5}")
    spread = _builder(placer=ResiduePlacer()).build("{[#EO]|5}")

    def min_separation(graph):
        p = np.array([[a["x"], a["y"], a["z"]] for a in graph.atoms])
        d = np.linalg.norm(p[:, None] - p[None, :], axis=-1) + np.eye(len(p)) * 1e9
        return d.min()

    assert min_separation(stacked) < 1e-9  # template copies sit on top of each other
    assert min_separation(spread) > 0.5  # ...until they are placed


def test_placer_uses_summed_covalent_radii_as_the_initial_bond_length():
    import numpy as np

    from molpy.builder.assembly import ResiduePlacer
    from molpy.core.element import Element

    chain = _builder(placer=ResiduePlacer()).build("{[#EO]|4}")
    lengths = [
        float(
            np.linalg.norm(
                np.array([b.itom["x"], b.itom["y"], b.itom["z"]])
                - np.array([b.jtom["x"], b.jtom["y"], b.jtom["z"]])
            )
        )
        for b in chain.bonds
    ]
    ether = Element("C").covalent + Element("O").covalent
    assert max(lengths) == pytest.approx(ether, abs=1e-6)


def test_alignment_is_a_proper_rotation_never_a_reflection():
    """`-I` would mirror a residue and invert its chirality."""
    import numpy as np

    from molpy.builder.assembly import ResiduePlacer

    rng = np.random.default_rng(0)
    cases = [rng.normal(size=(2, 3)) for _ in range(200)]
    cases += [
        np.array([[1.0, 0, 0], [-1.0, 0, 0]]),  # antiparallel
        np.array([[0, 0, 1.0], [0, 0, -1.0]]),
        np.array([[1.0, 0, 0], [1.0, 0, 0]]),  # parallel
    ]
    for source, target in cases:
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        rotation = ResiduePlacer._align(source, target)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
        assert rotation @ source == pytest.approx(target, abs=1e-9)


def test_placement_refuses_to_guess_an_unknown_element():
    from molpy.builder.assembly import ResiduePlacer

    atom = _eo().atoms[0]
    del atom[fields.ELEMENT]
    with pytest.raises(KeyError, match="covalent radius"):
        ResiduePlacer._radius(atom)


# --------------------------------------------------------------------------
# ac-009 — the typifier never sees the whole graph
# --------------------------------------------------------------------------


def test_typifier_is_never_handed_the_assembled_world():
    seen: list[object] = []

    class _Spy(_ElementTypifier):
        def match(self, graph):
            seen.append(graph)
            return super().match(graph)

    chain = _builder(typifier=_Spy()).build("{[#EO]|6}")
    assert seen, "the typifier was never called at all"
    assert not [g for g in seen if g is chain]

    # Count heavy atoms, not atoms: the pipeline hands the typifier a
    # valence-completed region, and the cap hydrogens say nothing about how much
    # of the world it was shown.
    def heavy(graph):
        return sum(1 for atom in graph.atoms if atom[fields.ELEMENT] != "H")

    assert max(heavy(g) for g in seen) < heavy(chain)


# --------------------------------------------------------------------------
# ac-013 — the new bonded terms are complete: no missing, no duplicate
# --------------------------------------------------------------------------


def _term_sets(graph):
    return {
        kind: sorted(
            tuple(sorted(e.handle for e in t.endpoints)) for t in getattr(graph, kind)
        )
        for kind in ("angles", "dihedrals")
    }


@pytest.mark.parametrize("cgsmiles", ["{[#EO]|6}", "{[#EO]1[#EO][#EO]1}"])
def test_locally_inserted_terms_equal_the_whole_graph_oracle(cgsmiles):
    """A whole-graph rebuild is the oracle here — never the product path."""
    chain = _builder(typifier=_ElementTypifier()).build(cgsmiles)
    local = _term_sets(chain)

    chain.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
    oracle = _term_sets(chain)

    for kind in ("angles", "dihedrals"):
        assert local[kind] == oracle[kind], (
            f"{kind}: {len(local[kind])} inserted locally vs "
            f"{len(oracle[kind])} from a whole-graph rebuild"
        )


# --------------------------------------------------------------------------
# ac-012 — overlapping regions agree; regions see the final graph
# --------------------------------------------------------------------------


def test_regions_are_built_after_every_edit_and_overlap_consistently():
    """A junction's neighbours are themselves junctions in a short-monomer chain.

    If a region were captured before a later edit, the two would disagree about a
    shared interior atom's type. Types written by overlapping regions must match.
    """
    # reach=3 is wide enough that neighbouring junction regions overlap
    chain = _builder(typifier=_ElementTypifier(), reach=3).build("{[#EO]|4}")
    # every atom carries exactly the type its element implies — no atom was left
    # holding a stale type from a region captured too early
    for atom in chain.atoms:
        assert atom[fields.TYPE] == f"t_{atom[fields.ELEMENT]}"


# --------------------------------------------------------------------------
# ac-017 — the cache key covers the context shell, not just the interior
# --------------------------------------------------------------------------


def test_isomorphic_interiors_with_different_shells_do_not_share_a_cache_entry():
    from molpy.typifier.affected_region import AffectedRegion

    def chain(tail_element: str):
        s = mp.Atomistic()
        atoms = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(6)]
        atoms[-1][fields.ELEMENT] = tail_element  # only the far shell differs
        for a, b in zip(atoms, atoms[1:], strict=False):
            s.def_bond(a, b)
        return s, atoms

    a_graph, a_atoms = chain("C")
    b_graph, b_atoms = chain("N")

    # reach=2 -> interior 2, extract 4
    region_a = AffectedRegion.around(a_graph, [a_atoms[1]], reach=2)
    region_b = AffectedRegion.around(b_graph, [b_atoms[1]], reach=2)

    # the interiors (hops <= 2 from atom 1) are identical carbon chains...
    assert [a[fields.ELEMENT] for a in region_a.interior] == [
        a[fields.ELEMENT] for a in region_b.interior
    ]
    # ...but the context shell differs, so they are different cache keys
    assert region_a != region_b
    assert hash(region_a) != hash(region_b)
