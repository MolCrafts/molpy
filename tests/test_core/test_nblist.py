import numpy as np
import numpy.testing as npt
import molpy as mp
import pytest
from molpy.core.neighborlist import NaiveNbList
import ase
from ase.neighborlist import neighbor_list


class TestNaiveNblist:

    @pytest.mark.parametrize(
        "cutoff, self_interaction",
        [
            (rc, self_interaction)
            for rc in [1, 3, 5, 7]
            for self_interaction in [True, False]
        ],
    )
    def test_free_space(self, ase_free_tests, cutoff, self_interaction):

        is_exclude_ii = not self_interaction
        is_exclude_ji = False

        for atoms in ase_free_tests:
            space = mp.Free()
            nblist = NaiveNbList()

            mapping, diff, actual_dist = nblist.build(
                atoms.positions,
                space,
                cutoff,
                exclude_ii=is_exclude_ii,
                exclude_ji=is_exclude_ji,
            )

            idx_i, idx_j, idx_S, expected_dist = neighbor_list(
                "ijSd", atoms, cutoff, self_interaction=self_interaction
            )
            npt.assert_allclose(np.sort(actual_dist), np.sort(expected_dist))

    @pytest.mark.parametrize(
        "cutoff, self_interaction",
        [(rc, self_interaction) for rc in [1, 3, 5, 7] for self_interaction in [False, True]],
    )
    def test_orth_space(self, ase_orth_tests, cutoff, self_interaction):
        is_exclude_ii = not self_interaction
        is_exclude_ji = False


        for atoms in ase_orth_tests:

            # dont support "self-interaction":
            # which means two more same pair idx in the mapping
            if np.any(cutoff > 0.5 * np.min(atoms.cell.array)):
                continue

            space = mp.Free()
            nblist = NaiveNbList()

            mapping, diff, actual_dist = nblist.build(
                atoms.positions,
                space,
                cutoff,
                exclude_ii=is_exclude_ii,
                exclude_ji=is_exclude_ji,
            )

            idx_i, idx_j, idx_S, expected_dist = neighbor_list(
                "ijSd", atoms, cutoff, self_interaction=self_interaction
            )
            npt.assert_equal(
                np.sort(np.unique(mapping, axis=0)),
                np.sort(np.unique(np.stack([idx_i, idx_j], axis=-1), axis=0)),
                err_msg=f"{np.unique(actual_dist)}"
            )
            npt.assert_equal(actual_dist, expected_dist)
            npt.assert_equal(mapping, np.stack([idx_i, idx_j], axis=-1), axis=0)