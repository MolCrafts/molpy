"""FFI seam of ``replicate(template, rotations, translations, frag_ids)``.

The leaf method grows *this* graph
by one rigid copy of ``template`` per row of ``rotations (N,3,3)`` /
``translations (N,3)``, stamping ``frag_id = frag_ids[c]`` on copy ``c``.
Column-wise copying, relation offsets and atomicity are proven by
``MolGraph::replicate`` in ``crates/molrs/src/core/molgraph.rs``; these tests
check only that the arrays cross, the graph grows, and bad shapes map to
``ValueError``. The one position asserted is a pure translation.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest

TWO_IDENTITIES = np.stack([np.eye(3), np.eye(3)])
TWO_SHIFTS = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
TWO_FRAG_IDS = np.array([7, 9], dtype=np.int32)


def _ch_template() -> molrs.core.Atomistic:
    """C at the origin bonded to H at x = 1.09."""
    template = molrs.core.Atomistic()
    c = template.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    h = template.add_atom(atomic_number=1, x=1.09, y=0.0, z=0.0)
    template.add_bond(c, h)
    return template


class TestAtomisticReplicate:
    def test_grows_the_graph_by_one_copy_per_transform_with_frag_ids_stamped(self):
        world = molrs.core.Atomistic()
        world.add_atom(atomic_number=8, x=-5.0, y=0.0, z=0.0)
        new = world.replicate(_ch_template(), TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)

        assert isinstance(new, molrs.core.Refs)
        assert len(new) == 4 and all(atom.world is world for atom in new)
        assert world.n_atoms == 1 + 2 * 2
        assert world.n_bonds == 2
        # Copy-major: copy 0's atoms come first.
        assert new["frag_id"].tolist() == [7, 7, 9, 9]
        # Copy 1 is the template shifted by (10, 0, 0): its carbon sits at x = 10.
        carbons = {
            atom["frag_id"]: atom["x"] for atom in new if atom["atomic_number"] == 6
        }
        assert carbons == {7: 0.0, 9: 10.0}

    def test_the_template_is_not_mutated(self):
        template = _ch_template()
        molrs.core.Atomistic().replicate(
            template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS
        )
        assert template.n_atoms == 2
        assert all("frag_id" not in atom for atom in template.atoms)

    @pytest.mark.parametrize(
        ("rotations", "translations", "frag_ids"),
        [
            (np.zeros((2, 3)), TWO_SHIFTS, TWO_FRAG_IDS),
            (TWO_IDENTITIES, np.zeros((2, 2)), TWO_FRAG_IDS),
            (TWO_IDENTITIES, TWO_SHIFTS[:1], TWO_FRAG_IDS),
            (TWO_IDENTITIES, TWO_SHIFTS, np.array([7, 9, 11], dtype=np.int32)),
        ],
        ids=[
            "rotations-2x3",
            "translations-2x2",
            "translations-count",
            "frag_ids-count",
        ],
    )
    def test_a_wrong_shape_is_a_value_error_and_leaves_the_graph_unchanged(
        self, rotations, translations, frag_ids
    ):
        world = molrs.core.Atomistic()
        with pytest.raises(ValueError):
            world.replicate(_ch_template(), rotations, translations, frag_ids)
        assert world.n_atoms == 0


class TestPortedReplicate:
    def test_copies_keep_the_template_ports(self):
        template = molrs.core.Atomistic()
        o = template.add_atom(atomic_number=8, x=0.0, y=0.0, z=0.0)
        h = template.add_atom(atomic_number=1, x=0.96, y=0.0, z=0.0)
        template.add_bond(o, h)
        template.add_port(o, h, "$")

        world = molrs.core.Atomistic()
        world.replicate(template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)

        assert world.n_atoms == 4
        assert world.n_ports == 2
        assert sorted(world.atoms["frag_id"].tolist()) == [7, 7, 9, 9]


class TestCoarseGrainReplicate:
    def test_grows_by_one_bead_per_transform(self):
        template = molrs.core.CoarseGrain()
        template.add_bead(bead_type="A", x=0.0, y=0.0, z=0.0)

        world = molrs.core.CoarseGrain()
        world.replicate(template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)

        assert world.n_beads == 2
        assert sorted(world.beads["frag_id"].tolist()) == [7, 9]


def test_a_graph_cannot_replicate_itself():
    template = _ch_template()
    with pytest.raises(ValueError, match="itself"):
        template.replicate(template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)
