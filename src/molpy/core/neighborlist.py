import itertools
from .box import Box
import numpy as np
import molpy as mp

NEIGHBOUR_GRID = np.array(
    [
        [-1, 1, 0],
        [-1, -1, 1],
        [-1, 0, 1],
        [-1, 1, 1],
        [0, -1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    np.int32,
)


class _NeighborList:

    def __init__(self, cutoff):
        self.cutoff = cutoff
        self._xyz = np.array([])

    def build(self, xyz):
        # xyz = frame.atoms.positions
        self._xyz = xyz
        self._cell_shape, self._cell_offset, self._all_cell_coords = self._init_cell(
            xyz, self.cutoff
        )
        self._xyz_cell_coords, self._xyz_cell_idx = self._add_to_cell(
            xyz, self._cell_offset
        )

    def update(self, frame):
        xyz = frame.atoms.positions
        # TODO: check if xyz and box change a lot
        if len(xyz) != len(self._xyz):
            self.build(frame)
        pairs = self.find_all_pairs(frame.box)
        frame[mp.Alias.idx_i] = pairs[:, 0]
        frame[mp.Alias.idx_j] = pairs[:, 1]
        frame[mp.Alias.Rij] = frame.box.diff(xyz[pairs[:, 0]], xyz[pairs[:, 1]])
        return frame

    def __call__(self, frame):
        return self.update(frame)

    def find_all_pairs(self, box):

        results = []
        for cell_coord in self._xyz_cell_coords:
            pairs = self._find_pairs_around_center_cell(cell_coord, box)
            results.append(pairs)
            pairs = self._find_pairs_in_center_cell(cell_coord, box)
            results.append(pairs)
        results = filter(lambda x: len(x) > 0, results)
        results = list(results)
        if len(results) == 0:
            return np.array([])
        results = np.concatenate(list(results))
        # NOTE: neighbor cell has been halved to avoid double counting
        results = np.sort(results, axis=-1)
        results = np.unique(results, axis=0)
        return results

    def _find_pairs_in_center_cell(self, center_cell_coord, box):

        center_xyz, center_id = self._find_atoms_in_cell(
            self._xyz,
            self._xyz_cell_idx,
            self._cell_coord_to_idx(center_cell_coord),
        )
        distance = np.linalg.norm(box.all_diff(center_xyz, center_xyz), axis=-1)
        pair_id = np.array(list(itertools.product(center_id, center_id)))
        cutoff_mask = np.logical_and(distance < self.cutoff, distance > 0)
        pairs = pair_id[cutoff_mask]
        return pairs[
            pairs[:, 0] < pairs[:, 1]
        ]  # halve the pairs to avoid double counting

    def _find_pairs_around_center_cell(self, center_cell_coord, box):
        nbor_cell = self._find_neighbor_cell(self._cell_shape, center_cell_coord)
        center_xyz, center_id = self._find_atoms_in_cell(
            self._xyz,
            self._xyz_cell_idx,
            self._cell_coord_to_idx(center_cell_coord),
        )
        around_xyz, around_id = self._find_atoms_in_cell(
            self._xyz, self._xyz_cell_idx, self._cell_coord_to_idx(nbor_cell)
        )
        distance = np.linalg.norm(
            box.all_diff(center_xyz, around_xyz), axis=-1
        )  # (N*M, )
        pair_id = np.array(list(itertools.product(center_id, around_id)))
        cutoff_mask = np.logical_and(distance < self.cutoff, distance > 0)
        pairs = pair_id[cutoff_mask]
        return pairs

    def _init_cell(
        self,
        xyz,
        cutoff,
    ):
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        space = max_xyz - min_xyz
        space = np.where(space == 0, 1, space)
        _cell_shape = np.ceil(space / cutoff).astype(int)
        _cell_offset = np.array(
            [_cell_shape[0] * _cell_shape[1], _cell_shape[0], 1], dtype=int
        )
        _all_cell_coords = np.array(list(np.ndindex(*_cell_shape)))
        return _cell_shape, _cell_offset, _all_cell_coords

    def _add_to_cell(self, xyz, cell_offset):
        _xyz_cell_coords = (xyz - np.min(xyz, axis=0)) // self.cutoff  # (N, D)
        _xyz_cell_idx = (_xyz_cell_coords * cell_offset).sum(axis=-1)  # (N,)
        return _xyz_cell_coords.astype(int), _xyz_cell_idx.astype(int)

    def _find_atoms_in_cell(self, xyz, xyz_cell_idx, which_cell_idx):
        mask = np.isin(xyz_cell_idx, which_cell_idx)
        return xyz[mask], np.where(mask)[0]

    def _find_neighbor_cell(self, cell_shape, center_cell_coord):
        cell_matrix = np.diag(cell_shape)
        nbor_cell = NEIGHBOUR_GRID + center_cell_coord
        reci_r = np.einsum("ij,nj->ni", np.linalg.inv(cell_matrix), nbor_cell)
        shifted_reci_r = reci_r - np.floor(reci_r)
        nbor_cell = np.einsum("ij,nj->ni", cell_matrix, shifted_reci_r)
        return nbor_cell

    def _cell_coord_to_idx(self, cell_coord):
        cell_idx = (cell_coord * self._cell_offset).sum(axis=-1)
        return cell_idx


class NeighborList:
    """ """

    def __init__(
        self,
        cutoff: float
    ):
        self.cutoff = cutoff

    def __call__(self, frame):
        R = frame.atoms.R
        return self.compute_neighborlist(
            self.cutoff, R, frame.box.matrix, frame.box.pbc, np.zeros(R.shape[0], dtype=int)
        )

    @staticmethod
    def compute_distances(
        pos: np.ndarray,
        mapping: np.ndarray,
        cell_shifts: np.ndarray | None = None,
    ):
        assert mapping.ndim == 2
        assert mapping.shape[0] == 2

        if cell_shifts is None:
            dr = pos[mapping[1]] - pos[mapping[0]]
        else:
            dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

        # return dr.norm(p=2, axis=1)
        return np.linalg.norm(dr, axis=1)

    @staticmethod
    def compute_cell_shifts(
        cell: np.ndarray, shifts_idx: np.ndarray, batch_mapping: np.ndarray
    ):
        if cell is None:
            cell_shifts = None
        else:
            cell_shifts = np.einsum(
                "jn,jnm->jm", shifts_idx, cell.reshape(-1, 3, 3)[batch_mapping]
            )
        return cell_shifts

    @staticmethod
    def strict_nl(
        cutoff: float,
        pos: np.ndarray,
        cell: np.ndarray,
        mapping: np.ndarray,
        batch_mapping: np.ndarray,
        shifts_idx: np.ndarray,
    ):
        """Apply a strict cutoff to the neighbor list defined in mapping.

        Parameters
        ----------
        cutoff : _type_
            _description_
        pos : _type_
            _description_
        cell : _type_
            _description_
        mapping : _type_
            _description_
        batch_mapping : _type_
            _description_
        shifts_idx : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        cell_shifts = NeighborList.compute_cell_shifts(cell, shifts_idx, batch_mapping)
        if cell_shifts is None:
            d2 = np.square(pos[mapping[0]] - pos[mapping[1]]).sum(axis=1)
        else:
            d2 = np.square(pos[mapping[0]] - pos[mapping[1]] - cell_shifts).sum(axis=1)

        mask = d2 < cutoff * cutoff
        mapping = mapping[:, mask]
        mapping_batch = batch_mapping[mask]
        shifts_idx = shifts_idx[mask]
        return mapping, mapping_batch, shifts_idx

    @staticmethod
    def ravel_3d(idx_3d: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """Convert 3d indices meant for an array of sizes `shape` into linear
        indices.

        Parameters
        ----------
        idx_3d : [-1, 3]
            _description_
        shape : [3]
            _description_

        Returns
        -------
        np.ndarray
            linear indices
        """
        idx_linear = idx_3d[:, 2] + shape[2] * (idx_3d[:, 1] + shape[1] * idx_3d[:, 0])
        return idx_linear.astype(int)

    @staticmethod
    def unravel_3d(idx_linear: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """Convert linear indices meant for an array of sizes `shape` into 3d indices.

        Parameters
        ----------
        idx_linear : np.ndarray [-1]

        shape : np.ndarray [3]


        Returns
        -------
        np.ndarray [-1, 3]

        """
        idx_3d = np.empty((idx_linear.shape[0], 3))
        idx_3d[:, 2] = np.remainder(idx_linear, shape[2])
        idx_3d[:, 1] = np.remainder(np.floor_divide(idx_linear, shape[2]), shape[1])
        idx_3d[:, 0] = np.floor_divide(idx_linear, shape[1] * shape[2])
        return idx_3d

    @staticmethod
    def get_linear_bin_idx(
        cell: np.ndarray, pos: np.ndarray, nbins_s: np.ndarray
    ) -> np.ndarray:
        """Find the linear bin index of each input pos given a box defined by its cell vectors and a number of bins, contained in the box, for each directions of the box.

        Parameters
        ----------
        cell : np.ndarray [3, 3]
            cell vectors
        pos : np.ndarray [-1, 3]
            set of positions
        nbins_s : np.ndarray [3]
            number of bins in each directions

        Returns
        -------
        np.ndarray
            linear bin index
        """
        scaled_pos = np.linalg.solve(cell.T, pos.T).T
        bin_index_s = np.floor(scaled_pos * nbins_s)
        bin_index_l = NeighborList.ravel_3d(bin_index_s, nbins_s)
        return bin_index_l

    @staticmethod
    def scatter_bin_index(
        nbins: int,
        max_n_atom_per_bin: int,
        n_images: int,
        bin_index: np.ndarray,
    ):
        """convert the linear table `bin_index` into the table `bin_id`. Empty entries in `bin_id` are set to `n_images` so that they can be removed later.

        Parameters
        ----------
        nbins : _type_
            total number of bins
        max_n_atom_per_bin : _type_
            maximum number of atoms per bin
        n_images : _type_
            total number of atoms counting the pbc replicas
        bin_index : _type_
            map relating `atom_index` to the `bin_index` that it belongs to such that `bin_index[atom_index] -> bin_index`.

        Returns
        -------
        bin_id : np.ndarray [nbins, max_n_atom_per_bin]
            relate `bin_index` (row) with the `atom_index` (stored in the columns).
        """
        sorted_id = np.argsort(bin_index)
        sorted_bin_index = bin_index[sorted_id]
        bin_id = np.full((nbins * max_n_atom_per_bin,), n_images)
        sorted_bin_id = np.remainder(np.arange(bin_index.shape[0]), max_n_atom_per_bin)
        sorted_bin_id = sorted_bin_index * max_n_atom_per_bin + sorted_bin_id
        # bin_id.scatter_(axis=0, index=sorted_bin_id, src=sorted_id)
        bin_id[sorted_bin_id] = sorted_id
        bin_id = bin_id.reshape((nbins, max_n_atom_per_bin))
        return bin_id

    @staticmethod
    def strides_of(v: np.ndarray) -> np.ndarray:
        v = v.flatten()
        stride = np.empty(v.shape[0] + 1, dtype=v.dtype)
        stride[0] = 0
        np.cumsum(v, axis=0, dtype=stride.dtype, out=stride[1:])
        return stride

    @staticmethod
    def get_number_of_cell_repeats(
        cutoff: float, cell: np.ndarray, pbc: np.ndarray
    ) -> np.ndarray:
        cell = cell.reshape((-1, 3, 3))
        pbc = pbc.reshape((-1, 3))

        has_pbc = pbc.prod(axis=1, dtype=bool)
        reciprocal_cell = np.zeros_like(cell)
        reciprocal_cell[has_pbc, :, :] = np.linalg.inv(cell[has_pbc, :, :]).transpose(
            0, 2, 1
        )
        # inv_distances = reciprocal_cell.norm(2, axis=-1)
        inv_distances = np.linalg.norm(reciprocal_cell, axis=-1)
        num_repeats = np.ceil(cutoff * inv_distances)
        num_repeats_ = np.where(pbc, num_repeats, np.zeros_like(num_repeats))
        return num_repeats_

    @staticmethod
    def get_cell_shift_idx(num_repeats: np.ndarray) -> np.ndarray:
        reps = []
        for ii in range(3):
            r1 = np.arange(-num_repeats[ii], num_repeats[ii] + 1)
            indices = np.argsort(np.abs(r1))
            reps.append(r1[indices])
        # shifts_idx = np.cartesian_prod(reps[0], reps[1], reps[2])
        shifts_idx = np.array(
            list(itertools.product(reps[0], reps[1], reps[2])), dtype=int
        )
        return shifts_idx

    @staticmethod
    def linked_cell(
        pos: np.ndarray,
        cell: np.ndarray,
        cutoff: float,
        num_repeats: np.ndarray,
        self_interaction: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine the atomic neighborhood of the atoms of a given structure for a particular cutoff using the linked cell algorithm.

        Parameters
        ----------
        pos : np.ndarray [n_atom, 3]
            atomic positions in the unit cell (positions outside the cell boundaries will result in an undifined behaviour)
        cell : np.ndarray [3, 3]
            unit cell vectors in the format V=[v_0, v_1, v_2]
        cutoff : float
            length used to determine neighborhood
        num_repeats : np.ndarray [3]
            number of unit cell repetitions in each directions required to account for PBC
        self_interaction : bool, optional
            to keep the original atoms as their own neighbor, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            neigh_atom : [2, n_neighbors]
                indices of the original atoms (neigh_atom[0]) with their neighbor index (neigh_atom[1]). The indices are meant to access the provided position array
            neigh_shift_idx : [n_neighbors, 3]
                cell shift indices to be used in reconstructing the neighbor atom positions.
        """
        dtype = pos.dtype
        n_atom = pos.shape[0]
        # find all the integer shifts of the unit cell given the cutoff and periodicity
        shifts_idx = NeighborList.get_cell_shift_idx(num_repeats)
        n_cell_image = shifts_idx.shape[0]
        shifts_idx = np.repeat(shifts_idx, n_atom, axis=0)
        batch_image = np.zeros((shifts_idx.shape[0]), dtype=int)
        cell_shifts = NeighborList.compute_cell_shifts(
            cell.reshape(-1, 3, 3), shifts_idx, batch_image
        )

        i_ids = np.arange(n_atom)
        i_ids = i_ids.repeat(n_cell_image)
        # compute the positions of the replicated unit cell (including the original)
        # they are organized such that: 1st n_atom are the non-shifted atom, 2nd n_atom are moved by the same translation, ...
        images = pos[i_ids] + cell_shifts
        n_images = images.shape[0]
        # create a rectangular box at [0,0,0] that encompases all the atoms (hence shifting the atoms so that they lie inside the box)
        b_min = images.min(axis=0)
        b_max = images.max(axis=0)
        images -= b_min - 1e-5
        box_length = b_max - b_min + 1e-3
        # divide the box into square bins of size cutoff in 3d
        nbins_s = np.maximum(np.ceil(box_length / cutoff), np.ones(3))
        # adapt the box lenghts so that it encompasses
        box_vec = np.diag(nbins_s * cutoff)
        nbins_s = nbins_s
        nbins = int(np.prod(nbins_s))
        # determine which bins the original atoms and the images belong to following a linear indexing of the 3d bins
        bin_index_j = NeighborList.get_linear_bin_idx(box_vec, images, nbins_s)
        n_atom_j_per_bin = np.bincount(bin_index_j, minlength=nbins)
        max_n_atom_per_bin = int(n_atom_j_per_bin.max())
        # convert the linear map bin_index_j into a 2d map. This allows for
        # fully vectorized neighbor assignment
        bin_id_j = NeighborList.scatter_bin_index(
            nbins, max_n_atom_per_bin, n_images, bin_index_j
        )

        # find which bins the original atoms belong to
        bin_index_i = bin_index_j[:n_atom]
        i_bins_l = np.unique(bin_index_i)
        i_bins_s = NeighborList.unravel_3d(i_bins_l, nbins_s)

        # find the bin indices in the neighborhood of i_bins_l. Since the bins have
        # a side length of cutoff only 27 bins are in the neighborhood
        # (including itself)
        dd = np.array([0, 1, -1])
        bin_shifts = np.array(list(itertools.product(dd, dd, dd)))
        n_neigh_bins = bin_shifts.shape[0]
        #  bin_shifts = bin_shifts.repeat((i_bins_s.shape[0], 1))
        bin_shifts = np.tile(bin_shifts, (i_bins_s.shape[0], 1))
        neigh_bins_s = (
            np.repeat(
                i_bins_s,
                n_neigh_bins,
                axis=0,
                # output_size=n_neigh_bins * i_bins_s.shape[0],
            )
            + bin_shifts
        )
        # some of the generated bin_idx might not be valid
        mask = np.all(
            np.logical_and(neigh_bins_s < nbins_s.reshape(1, 3), neigh_bins_s >= 0),
            axis=1,
        )

        # remove the bins that are outside of the search range, i.e. beyond the borders of the box in the case of non-periodic directions.
        neigh_j_bins_l = NeighborList.ravel_3d(neigh_bins_s[mask], nbins_s)

        max_neigh_per_atom = max_n_atom_per_bin * n_neigh_bins
        # the i_bin related to neigh_j_bins_l
        repeats = mask.reshape(-1, n_neigh_bins).sum(axis=1)
        neigh_i_bins_l = np.concatenate(
            [
                np.arange(rr) + i_bins_l[ii] * n_neigh_bins
                for ii, rr in enumerate(repeats)
            ],
            axis=0,
        )
        # the linear neighborlist. make it at large as necessary
        neigh_atom = np.empty((2, n_atom * max_neigh_per_atom), dtype=int)
        # fill the i_atom index
        neigh_atom[0] = np.tile(
            np.arange(n_atom).reshape(-1, 1), (1, max_neigh_per_atom)
        ).reshape(-1)
        # relate `bin_index` (row) with the `neighbor_atom_index` (stored in the columns). empty entries are set to `n_images`
        bin_id_ij = np.full(
            (nbins * n_neigh_bins, max_n_atom_per_bin),
            n_images,
        )
        # fill the bins with neighbor atom indices
        bin_id_ij[neigh_i_bins_l] = bin_id_j[neigh_j_bins_l]
        bin_id_ij = bin_id_ij.reshape((nbins, max_neigh_per_atom))
        # map the neighbors in the bins to the central atoms
        neigh_atom[1] = bin_id_ij[bin_index_i].reshape(-1)
        # remove empty entries
        neigh_atom = neigh_atom[:, neigh_atom[1] != n_images]

        if not self_interaction:
            # neighbor atoms are still indexed from 0 to n_atom*n_cell_image
            neigh_atom = neigh_atom[:, neigh_atom[0] != neigh_atom[1]]

        # sort neighbor list so that the i_atom indices increase
        sorted_ids = np.argsort(neigh_atom[0])
        neigh_atom = neigh_atom[:, sorted_ids]
        # get the cell shift indices for each neighbor atom
        neigh_shift_idx = shifts_idx[neigh_atom[1]]
        # make sure the j_atom indices access the original positions
        neigh_atom[1] = np.remainder(neigh_atom[1], n_atom)
        # print(neigh_atom)
        return neigh_atom, neigh_shift_idx

    @staticmethod
    def build_linked_cell_neighborhood(
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
        cutoff: float,
        n_atoms: np.ndarray,
        self_interaction: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the neighborlist of a given set of atomic structures using the linked cell algorithm.

        Parameters
        ----------
        positions : np.ndarray [-1, 3]
            set of atomic positions for each structures
        cell : np.ndarray [3*n_structure, 3]
            set of unit cell vectors for each structures
        pbc : np.ndarray [n_structures, 3] bool
            periodic boundary conditions to apply
        cutoff : float
            length used to determine neighborhood
        n_atoms : np.ndarray
            number of atoms in each structures
        self_interaction : bool
            to keep the original atoms as their own neighbor

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            mapping : [2, n_neighbors]
                indices of the neighbor list for the given positions array, mapping[0/1] correspond respectively to the central/neighbor atom (or node in the graph terminology)
            batch_mapping : [n_neighbors]
                indices mapping the neighbor atom to each structures
            cell_shifts_idx : [n_neighbors, 3]
                cell shift indices to be used in reconstructing the neighbor atom positions.
        """

        n_structure = n_atoms.shape[0]
        cell = cell.reshape((-1, 3, 3))
        pbc = pbc.reshape((-1, 3))
        # compute the number of cell replica necessary so that all the unit cell's atom have a complete neighborhood (no MIC assumed here)
        num_repeats = NeighborList.get_number_of_cell_repeats(cutoff, cell, pbc)

        stride = NeighborList.strides_of(n_atoms)

        mapping, batch_mapping, cell_shifts_idx = [], [], []
        for i_structure in range(n_structure):
            # compute the neighborhood with the linked cell algorithm
            neigh_atom, neigh_shift_idx = NeighborList.linked_cell(
                positions[stride[i_structure] : stride[i_structure + 1]],
                cell[i_structure],
                cutoff,
                num_repeats[i_structure],
                self_interaction,
            )

            batch_mapping.append(i_structure * np.ones(neigh_atom.shape[1]))
            # shift the mapping indices so that they can access positions
            mapping.append(neigh_atom + stride[i_structure])
            cell_shifts_idx.append(neigh_shift_idx)
        return (
            np.concatenate(mapping, axis=1).astype(int),
            np.concatenate(batch_mapping, axis=0).astype(int),
            np.concatenate(cell_shifts_idx, axis=0).astype(int),
        )

    @staticmethod
    def compute_neighborlist(
        cutoff: float,
        pos: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
        batch: np.ndarray,
        self_interaction: bool = False,
    ):
        """Compute the neighborlist for a set of atomic structures using the linked
        cell algorithm before applying a strict `cutoff`. The atoms positions `pos`
        should be wrapped inside their respective unit cells.

        Parameters
        ----------
        cutoff : float
            cutoff radius of used for the neighbor search
        pos : np.ndarray [n_atom, 3]
            set of atoms positions wrapped inside their respective unit cells
        cell : np.ndarray [3*n_structure, 3]
            unit cell vectors in the format [a_1, a_2, a_3]
        pbc : np.ndarray [n_structure, 3] bool
            periodic boundary conditions to apply. Partial PBC are not supported yet
        batch : np.ndarray torch.long [n_atom,]
            index of the structure in which the atom belongs to
        self_interaction : bool, optional
            to keep the center atoms as their own neighbor, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            mapping : [2, n_neighbors]
                indices of the neighbor list for the given positions array, mapping[0/1] correspond respectively to the central/neighbor atom (or node in the graph terminology)
            batch_mapping : [n_neighbors]
                indices mapping the neighbor atom to each structures
            shifts_idx : [n_neighbors, 3]
                cell shift indices to be used in reconstructing the neighbor atom positions.
        """
        n_atoms = np.bincount(batch)
        mapping, batch_mapping, shifts_idx = (
            NeighborList.build_linked_cell_neighborhood(
                pos, cell, pbc, cutoff, n_atoms, self_interaction
            )
        )

        mapping, mapping_batch, shifts_idx = NeighborList.strict_nl(
            cutoff, pos, cell, mapping, batch_mapping, shifts_idx
        )
        return mapping, mapping_batch, shifts_idx

class NaiveNeighborList:

    def __init__(self):
        ...

    def build(self, xyz: np.ndarray, box: Box, r_cutoff: float, ):
        """
        build from scratch
        """
        xyz = np.atleast_2d(xyz)  # (N, ndim)
        diff = xyz[:, None, :] - xyz[None, :, :]
        box.wrap(diff)
        dist = np.linalg.norm(diff, axis=-1)
        

    def update(self, xyz):
        ...

    def query(self, centers, around):
        ...