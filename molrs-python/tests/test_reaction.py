"""Python-binding coverage for the native reaction-SMARTS (SMIRKS) engine.

Exposes the validated Rust ``Reaction`` (``crates/molrs/src/core/chem/smarts/reaction.rs``)
to Python: parse a Daylight reaction SMARTS ``reactants >> products``, read back
the reactant components + forming-bond map pairs, and apply the compiled
transform to one matched occurrence in place. The transform compilation itself
(map diff, unmapped-atom pairing, form/break/set-order) is exercised in Rust;
these tests assert the PyO3 surface and Daylight atom-map semantics.
"""

import copy

import molrs
import pytest
from molrs.core import Atom, BondNumber, BondOrder

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Br": 35}


def _atom(mol, symbol, x, y, z):
    """Add an atom of element ``symbol`` at ``(x, y, z)``; return its view."""
    return mol.add_atom(atomic_number=Z[symbol], x=x, y=y, z=z)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _amine_plus_ester():
    """A single Atomistic holding two disconnected fragments.

    Amine  : N0 with two explicit H (matches ``[N;H2:1]``).
    Ester  : carbonyl C0 =O1, single -O2- to methyl C3
             (matches ``[C:2](=O)OC``; leaving group = O2 + C3).
    """
    mol = molrs.core.Atomistic()
    # amine
    n0 = _atom(mol, "N", 0.0, 0.0, 0.0)
    h1 = _atom(mol, "H", 0.6, 0.8, 0.0)
    h2 = _atom(mol, "H", -0.6, 0.8, 0.0)
    mol.add_bond(n0, h1)
    mol.add_bond(n0, h2)
    # ester
    c0 = _atom(mol, "C", 5.0, 0.0, 0.0)  # carbonyl C
    o1 = _atom(mol, "O", 6.0, 0.8, 0.0)  # =O (preserved)
    o2 = _atom(mol, "O", 5.0, -1.3, 0.0)  # ester O (leaves)
    c3 = _atom(mol, "C", 5.0, -2.6, 0.0)  # alkyl C (leaves)
    bo = mol.add_bond(c0, o1)
    mol.set_bond_class(bo, BondOrder.Double, BondNumber.Double)  # carbonyl double bond
    mol.add_bond(c0, o2)
    mol.add_bond(o2, c3)
    return mol, {"n0": n0, "c0": c0, "o1": o1, "o2": o2, "c3": c3}


def _bonded(mol, atom):
    """The atoms bonded to ``atom``."""
    return {other for other, _ in mol.neighbor_bonds(atom)}


def _bind(reaction, mol):
    """Match each reactant component and merge the map->atom dicts."""
    binding = {}
    for pat in reaction.reactant_patterns:
        matches = pat.find_matches(mol)
        assert matches, f"pattern {pat!r} did not match the fixture"
        binding.update(matches[0].mapping)
    return binding


# ---------------------------------------------------------------------------
# ac-001: parse (>> split + multi-component LHS + pairwise maps + agent)
# ---------------------------------------------------------------------------


def test_parse_two_reactant_components_and_product():
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    pats = rxn.reactant_patterns
    assert len(pats) == 2
    assert all(isinstance(p, molrs.perceive.SmartsPattern) for p in pats)
    # component 0 is the amine (1 query atom), component 1 the ester (4 atoms)
    assert pats[0].n_query_atoms == 1
    assert pats[1].n_query_atoms == 4


def test_parse_tolerates_agent_field():
    # "A > agent > B" three-part form must not crash; the agent is ignored.
    rxn = molrs.perceive.Reaction("[C:1]=[C:2].[S;H1:3] > [Pt] > [C:1][C:2][S:3]")
    assert len(rxn.reactant_patterns) == 2


def test_parse_rejects_missing_arrow():
    with pytest.raises(ValueError):
        molrs.perceive.Reaction("[C:1][O:2]")


# ---------------------------------------------------------------------------
# ac-002: forming-bond map pairs + one-sided-map validation
# ---------------------------------------------------------------------------


def test_forming_bonds_amide():
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    assert rxn.forming_bonds == [(1, 2)]


def test_forming_bonds_thiol_ene_is_only_the_new_bond():
    # (1,2) merely changes order (2->1) so it is NOT a forming bond; (2,3) is.
    rxn = molrs.perceive.Reaction("[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]")
    assert rxn.forming_bonds == [(2, 3)]


def test_one_sided_map_is_an_error():
    # :2 appears only on the reactant side -> Daylight pairwise rule violated.
    with pytest.raises(ValueError):
        molrs.perceive.Reaction("[C:1][O:2] >> [C:1]")


# ---------------------------------------------------------------------------
# ac-003: single-occurrence apply + topology refresh
# ---------------------------------------------------------------------------


def test_apply_amide_forms_bond_and_drops_leaving_group():
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, _h = _amine_plus_ester()
    n_before = mol.n_atoms
    assert not molrs.perceive.SmartsPattern("[N][C]=O").has_match(mol)

    binding = _bind(rxn, mol)
    assert set(binding) == {1, 2}
    touched = rxn.apply(mol, binding)

    # apply reports the touched (surviving) atoms as a list of Atom views
    assert isinstance(touched, list)
    assert all(isinstance(t, Atom) for t in touched)
    # leaving atoms (ester O + alkyl C) removed -> exactly 2 fewer atoms
    assert mol.n_atoms == n_before - 2
    # new N-C(=O) amide linkage present; carbonyl O preserved (not re-added)
    assert molrs.perceive.SmartsPattern("[N][C]=O").has_match(mol)
    # topology regenerated around the new bond
    assert mol.n_angles > 0


def test_apply_reuses_core_and_leaves_binding_atoms_alive():
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, _h = _amine_plus_ester()
    binding = _bind(rxn, mol)
    touched = rxn.apply(mol, binding)
    assert isinstance(touched, list)
    # the mapped atoms survive and are now bonded
    assert binding[2] in _bonded(mol, binding[1])


# ---------------------------------------------------------------------------
# ac-004: unmapped RHS atom is added (species from template, no coordinates)
# ---------------------------------------------------------------------------


def test_apply_adds_unmapped_product_atom():
    # substitution: C-Br -> C-O ; Br leaves, O is a brand-new atom.
    rxn = molrs.perceive.Reaction("[C:1]Br >> [C:1]O")
    mol = molrs.core.Atomistic()
    c0 = _atom(mol, "C", 0.0, 0.0, 0.0)
    br = _atom(mol, "Br", 1.9, 0.0, 0.0)
    mol.add_bond(c0, br)

    binding = _bind(rxn, mol)
    assert set(binding) == {1}
    touched = rxn.apply(mol, binding)
    assert isinstance(touched, list)

    # Br removed, O added -> still 2 atoms
    assert mol.n_atoms == 2
    species = {atom["atomic_number"] for atom in mol.atoms}
    assert species == {6, 8}
    # the new O is bonded to C0 and carries no coordinates
    oxygen = next(atom for atom in mol.atoms if atom["atomic_number"] == 8)
    assert oxygen in _bonded(mol, c0)
    assert "x" not in oxygen


# ---------------------------------------------------------------------------
# ac-005: no regression of the chain-01 matcher
# ---------------------------------------------------------------------------


def test_smarts_matcher_still_works():
    mol = molrs.core.Atomistic()
    c = _atom(mol, "C", 0.0, 0.0, 0.0)
    o = _atom(mol, "O", 1.4, 0.0, 0.0)
    mol.add_bond(c, o)
    assert molrs.perceive.SmartsPattern("[C:1][O:2]").has_match(mol)


# ---------------------------------------------------------------------------
# region-support-02: apply reports the touched (surviving) atom handles
# ---------------------------------------------------------------------------


def test_apply_touched_amide_endpoints_and_leaving_neighbor():
    """ac-001: touched has the formed-bond endpoints N/C plus the leaving
    group's surviving neighbor; the deleted atoms are absent; the result is
    deduped."""
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, h = _amine_plus_ester()
    binding = _bind(rxn, mol)
    touched = rxn.apply(mol, binding)
    touched_set = set(touched)

    # formed N-C bond endpoints (C is also the ester O's surviving neighbor)
    assert binding[1] in touched_set  # amine N
    assert binding[2] in touched_set  # ester carbonyl C
    # deleted atoms (ester O + alkyl C) must NOT appear
    assert h["o2"] not in touched_set
    assert h["c3"] not in touched_set
    # deduped
    assert len(touched) == len(touched_set)


def test_apply_touched_includes_added_atom():
    """ac-002: a reaction adding an RHS atom includes the new atom; the
    surviving carbon is touched, the deleted Br is not."""
    rxn = molrs.perceive.Reaction("[C:1]Br >> [C:1]O")
    mol = molrs.core.Atomistic()
    c0 = _atom(mol, "C", 0.0, 0.0, 0.0)
    br = _atom(mol, "Br", 1.9, 0.0, 0.0)
    mol.add_bond(c0, br)

    binding = _bind(rxn, mol)
    touched = set(rxn.apply(mol, binding))

    oxygen = next(atom for atom in mol.atoms if atom["atomic_number"] == 8)
    assert oxygen in touched  # newly added atom
    assert c0 in touched  # surviving neighbor of the leaving Br
    assert br not in touched  # deleted atom absent
    assert {atom.handle for atom in touched} == {oxygen.handle, c0.handle}


def test_apply_touched_thiol_ene_two_carbons_and_sulfur():
    """ac-002: thiol-ene touched = the two carbons (order change) + the sulfur
    (formed bond); the sulfur's spectator H is not touched."""
    rxn = molrs.perceive.Reaction("[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]")
    mol = molrs.core.Atomistic()
    c1 = _atom(mol, "C", 0.0, 0.0, 0.0)
    c2 = _atom(mol, "C", 1.3, 0.0, 0.0)
    s3 = _atom(mol, "S", 3.0, 0.0, 0.0)
    hs = _atom(mol, "H", 3.0, 1.0, 0.0)  # explicit H so [S;H1] matches
    b = mol.add_bond(c1, c2)
    mol.set_bond_class(b, BondOrder.Double, BondNumber.Double)
    mol.add_bond(s3, hs)

    binding = _bind(rxn, mol)
    touched = set(rxn.apply(mol, binding))
    assert touched == {c1, c2, s3}


def test_apply_many_compiles_all_leaving_groups_before_mutation():
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    first, _ = _amine_plus_ester()
    second, _ = _amine_plus_ester()
    binding_first = _bind(rxn, first)
    binding_second = _bind(rxn, second)

    world = first
    row_of = {atom: row for row, atom in enumerate(second.atoms)}
    merged = world.merge(second)  # second's atoms, now in world, in row order
    binding_second = {
        label: merged[row_of[atom]] for label, atom in binding_second.items()
    }
    before = world.n_atoms

    batch = rxn.apply_many(world, [binding_first, binding_second], refresh=False)

    assert isinstance(batch, molrs.perceive.ReactionBatch)
    assert world.n_atoms == before - 4
    assert len(batch.touched) == len(batch.created) == 2
    assert batch.created == [[], []]
    assert all(
        set(binding.values()) <= set(seed)
        for binding, seed in zip(
            (binding_first, binding_second), batch.touched, strict=True
        )
    )


def test_apply_many_works_on_a_copy_and_leaves_the_original():
    rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, _ = _amine_plus_ester()
    work = copy.copy(mol)
    rxn.apply_many(work, [_bind(rxn, work)], refresh=False)
    assert (mol.n_atoms, work.n_atoms) == (7, 5)


def test_apply_many_reports_rhs_creation_order():
    reaction = molrs.perceive.Reaction("[N:1].[C:2]>>[N:1][C:2]([O])[S]")
    mol = molrs.core.Atomistic()
    n = mol.add_atom(atomic_number=7)
    c = mol.add_atom(atomic_number=6)

    batch = reaction.apply_many(mol, [{1: n, 2: c}], refresh=False)

    assert len(batch.touched) == len(batch.created) == 1
    assert len(batch.created[0]) == 2
    assert [atom["atomic_number"] for atom in batch.created[0]] == [8, 16]
