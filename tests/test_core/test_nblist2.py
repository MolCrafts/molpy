import pytest
from ase.build import bulk, molecule
import numpy as np
import molpy as mp
from ase.neighborlist import neighbor_list
from ase import Atoms

from molpy.core.neighborlist import NaiveNbList


def ase2data(frames):
    n_atoms = [0]
    pos = []
    cell = []
    pbc = []
    for ff in frames:
        n_atoms.append(len(ff))
        pos.append(ff.get_positions())
        cell.append(ff.get_cell().array)
        pbc.append(ff.get_pbc())
    pos = np.concatenate(pos)
    cell = np.concatenate(cell)
    pbc = np.concatenate(pbc)
    stride = np.cumsum(n_atoms)
    batch = np.zeros(pos.shape[0], dtype=int)
    for ii, (st, nd) in enumerate(zip(stride[:-1], stride[1:])):
        batch[st:nd] = ii
    n_atoms = n_atoms[1:]
    return (
        pos,
        cell,
        pbc,
        batch,
        n_atoms,
    )


# triclinic atomic structure
CaCrP2O7_mvc_11955_symmetrized = {
    "positions": [
        [3.68954016, 5.03568186, 4.64369552],
        [5.12301681, 2.13482791, 2.66220405],
        [1.99411973, 0.94691001, 1.25068234],
        [6.81843724, 6.22359976, 6.05521724],
        [2.63005662, 4.16863452, 0.86090529],
        [6.18250036, 3.00187525, 6.44499428],
        [2.11497733, 1.98032773, 4.53610884],
        [6.69757964, 5.19018203, 2.76979073],
        [1.39215545, 2.94386142, 5.60917746],
        [7.42040152, 4.22664834, 1.69672212],
        [2.43224207, 5.4571615, 6.70305327],
        [6.3803149, 1.71334827, 0.6028463],
        [1.11265639, 1.50166318, 3.48760997],
        [7.69990058, 5.66884659, 3.8182896],
        [3.56971588, 5.20836551, 1.43673437],
        [5.2428411, 1.96214426, 5.8691652],
        [3.12282634, 2.72812741, 1.05450432],
        [5.68973063, 4.44238236, 6.25139525],
        [3.24868468, 2.83997522, 3.99842386],
        [5.56387229, 4.33053455, 3.30747571],
        [2.60835346, 0.74421609, 5.3236629],
        [6.20420351, 6.42629368, 1.98223667],
    ],
    "cell": [
        [6.19330899, 0.0, 0.0],
        [2.4074486111396207, 6.149627748674982, 0.0],
        [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
    ],
    "numbers": [
        20,
        20,
        24,
        24,
        15,
        15,
        15,
        15,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
    ],
    "pbc": [True, True, True],
}


def bulk_metal():
    frames = [
        bulk("Si", "diamond", a=6, cubic=True),
        # bulk("Si", "diamond", a=6),
        # bulk("Cu", "fcc", a=3.6),
        # bulk("Si", "bct", a=6, c=3),
        # # test very skewed unit cell
        # bulk("Bi", "rhombohedral", a=6, alpha=20),
        # bulk("Bi", "rhombohedral", a=6, alpha=10),
        # bulk("Bi", "rhombohedral", a=6, alpha=5),
        # bulk("SiCu", "rocksalt", a=6),
        # bulk("SiFCu", "fluorite", a=6),
        # Atoms(**CaCrP2O7_mvc_11955_symmetrized),
    ]
    return frames


def atomic_structures():
    frames = [
        # molecule("H2O"),
        # molecule("OCHCHO"),
        # molecule("CH3CH2NH2"),
        # molecule("methylenecyclopropane"),
        
        # molecule("C3H9C"),
    ] + bulk_metal()
    return frames


# @pytest.mark.parametrize(
#     "frames, cutoff, self_interaction",
#     [
#         (atomic_structures(), rc, self_interaction)
#         for rc in [1, 3, 5, 7]
#         for self_interaction in [True, False]
#     ],
# )
# def test_neighborlist_n2(frames, cutoff, self_interaction):
#     """Check that torch_neighbor_list gives the same NL as ASE by comparing
#     the resulting sorted list of distances between neighbors."""
#     pos, cell, pbc, batch, n_atoms = ase2data(frames)

#     dds = []
#     mapping, batch_mapping, shifts_idx = NeighborList.compute_neighborlist_n2(
#         cutoff, pos, cell, pbc, batch, self_interaction
#     )
#     cell_shifts = NeighborList.compute_cell_shifts(cell, shifts_idx, batch_mapping)
#     dds = NeighborList.compute_distances(pos, mapping, cell_shifts)
#     dds = np.sort(dds.numpy())

#     dd_ref = []
#     for frame in frames:
#         idx_i, idx_j, idx_S, dist = neighbor_list(
#             "ijSd", frame, cutoff=cutoff, self_interaction=self_interaction
#         )
#         dd_ref.extend(dist)
#     dd_ref = np.sort(dd_ref)

#     np.testing.assert_allclose(dd_ref, dds)


class TestNaiveNblist:

    @pytest.mark.parametrize(
        "frames, cutoff, self_interaction",
        [
            (atomic_structures(), rc, self_interaction)
            for rc in [1, 3, 5, 7]
            for self_interaction in [True, False]
        ],
    )
    def test_build(self, frames, cutoff, self_interaction):
        pos, cell, pbc, batch, n_atoms = ase2data(frames)
        cell = cell.reshape(-1, 3, 3)

        dds = []

        nblist = NaiveNbList()

        # mapping, diff, dist = nblist.batch_build([pos[batch==i] for i in range(len(frames))], [mp.Box(cell[i], pbc=pbc[i]) for i in range(len(frames))], cutoff, exclude_ii=not self_interaction, exclude_ji=not self_interaction)
        for i, frame in enumerate(frames):
            print(frame)
            is_exclude_ii = not self_interaction

            mapping, diff, actual_dist = nblist.build(
                pos[batch == i],
                mp.Box(cell[i], pbc=pbc[i]),
                cutoff,
                exclude_ii=is_exclude_ii,
                exclude_ji=False,
            )
            idx_i, idx_j, idx_S, expected_dist = neighbor_list(
                "ijSd", frame, cutoff=cutoff, self_interaction=self_interaction
            )
            np.testing.assert_allclose(np.sort(actual_dist), np.sort(expected_dist))


# @pytest.mark.parametrize(
#     "frames, cutoff, self_interaction",
#     [
#         (atomic_structures(), rc, self_interaction)
#         # for rc in [1] #[1, 3, 5, 7]
#         # for self_interaction in [False]
#         for rc in [1, 3, 5, 7]
#         for self_interaction in [False, True]
#     ],
# )
# def test_neighborlist_linked_cell(frames, cutoff, self_interaction):
#     """Check that torch_neighbor_list gives the same NL as ASE by comparing
#     the resulting sorted list of distances between neighbors."""
#     pos, cell, pbc, batch, n_atoms = ase2data(frames)

#     dds = []
#     mapping, batch_mapping, shifts_idx = NeighborList.compute_neighborlist(
#         cutoff, pos, cell, pbc, batch, self_interaction
#     )
#     cell_shifts = NeighborList.compute_cell_shifts(cell, shifts_idx, batch_mapping)
#     dds = NeighborList.compute_distances(pos, mapping, cell_shifts)
#     dds = np.sort(dds)

#     dd_ref = []
#     for frame in frames:
#         idx_i, idx_j, idx_S, dist = neighbor_list(
#             "ijSd", frame, cutoff=cutoff, self_interaction=self_interaction
#         )
#         dd_ref.extend(dist)
#     # nice for understanding if something goes wrong
#     # idx_S = torch.from_numpy(idx_S).to(torch.float64)

#     print("idx_i", idx_i)
#     print("idx_j", idx_j)
#     missing_entries = []
#     for ineigh in range(idx_i.shape[0]):
#         mask = np.logical_and(
#             idx_i[ineigh] == mapping[0], idx_j[ineigh] == mapping[1]
#         )
#         print((idx_S[ineigh] == shifts_idx[mask]).shape)
#         if np.any(np.all(idx_S[ineigh] == shifts_idx[mask])):
#             pass
#         else:
#             missing_entries.append(
#                 (idx_i[ineigh], idx_j[ineigh], idx_S[ineigh])
#             )
#             print(missing_entries[-1])
#             print(
#                 NeighborList.compute_cell_shifts(
#                     cell,
#                     idx_S[ineigh].view((1, -1)),
#                     np.array([0]),
#                 )
#             )

#     dd_ref = np.sort(dd_ref)
#     print(dd_ref)
#     print(dds)
#     np.testing.assert_allclose(dd_ref, dds)
