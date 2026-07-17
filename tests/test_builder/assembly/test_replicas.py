"""Unit tests for :mod:`molpy.builder.assembly._replicas`."""

import pytest

from molpy.builder.assembly import Replicas
from molpy.core import fields


class TestReplicas:
    def test_strand_property_preserves_input_identity(self, eo_factory):
        strand = eo_factory()
        assert Replicas(strand).strand is strand

    def test_times_sets_one_based_molecule_ids(self, eo_factory):
        strand = eo_factory()
        world = Replicas(strand).times(3, spacing=4.0)
        assert {int(atom[fields.MOL_ID]) for atom in world.atoms} == {1, 2, 3}
        assert world.n_atoms == 3 * strand.n_atoms

    def test_grid_creates_n_cubed_copies(self, eo_factory):
        strand = eo_factory()
        world = Replicas(strand).grid(2, spacing=5.0, jitter=0.0, rotate=False, seed=0)
        assert world.n_atoms == 8 * strand.n_atoms
        assert {int(atom[fields.MOL_ID]) for atom in world.atoms} == set(range(1, 9))

    @pytest.mark.parametrize(
        ("method", "args", "message"),
        [
            ("times", (0,), "count must be >= 1"),
            ("grid", (0, 1.0), "grid size n must be >= 1"),
            ("grid", (1, 0.0), "spacing must be positive"),
        ],
    )
    def test_invalid_replication_shape_is_rejected(
        self, eo_factory, method, args, message
    ):
        with pytest.raises(ValueError, match=message):
            getattr(Replicas(eo_factory()), method)(*args)
