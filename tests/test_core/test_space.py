import ase.geometry
import molpy as mp
import numpy as np
import numpy.testing as npt
import ase
import pytest


class TestFree:

    def test_init_(self):

        free = mp.Free()
        assert free
        assert isinstance(free, mp.Boundary)

    def test_wrap(self, ase_free_tests):

        free = mp.Free()

        for atoms in ase_free_tests:
            pos = atoms.positions.copy()
            npt.assert_equal(free.wrap(pos), atoms.positions)

    def test_diff_dr(self, ase_free_tests):

        free = mp.Free()

        for atoms in ase_free_tests:
            dr = atoms.positions.copy() - np.zeros_like(atoms.positions)
            npt.assert_equal(free.diff_dr(dr), dr)

    def test_diff_pair(self, ase_free_tests):

        free = mp.Free()

        for atoms in ase_free_tests:
            pos = atoms.positions.copy()
            if len(pos) % 2:
                r1, r2 = np.split(pos[:-1], 2)
            else:
                r1, r2 = np.split(pos, 2)
            npt.assert_equal(free.diff_pair(r1, r2), r1 - r2)

    def test_diff_all(self, ase_free_tests):

        free = mp.Free()

        for atoms in ase_free_tests:
            pos = atoms.positions.copy()
            r1 = pos.copy()
            r2 = pos.copy()
            diff = free.diff_all(r1, r2)
            npt.assert_equal(diff, r1[:, None, :] - r2)

    def test_diff_self(self, ase_free_tests):

        free = mp.Free()

        for atoms in ase_free_tests:
            pos = atoms.positions.copy()
            diff = free.diff_self(pos)
            npt.assert_equal(diff, pos[:, None, :] - pos)


class TestOrthorhombicBox:

    def test_init(self):

        box = mp.OrthorhombicBox([10, 10, 10])
        assert box
        assert isinstance(box, mp.Box)
        assert isinstance(box, mp.Boundary)

    def test_wrap(self, ase_orth_tests):

        for atoms in ase_orth_tests:
            atoms.center()
            lengths = np.diag(atoms.cell.array)
            box = mp.OrthorhombicBox(lengths, pbc=atoms.get_pbc())
            pos = atoms.positions.copy()
            npt.assert_allclose(
                box.wrap(pos),
                ase.geometry.wrap_positions(pos, atoms.cell.array, atoms.pbc),
            )

    def test_diff_dr(self, ase_orth_tests):

        for atoms in ase_orth_tests:
            atoms.center()
            lengths = np.diag(atoms.cell.array)
            box = mp.OrthorhombicBox(lengths, pbc=atoms.get_pbc())
            dr = atoms.positions.copy() - np.zeros_like(atoms.positions)
            expected_dr, _ = ase.geometry.find_mic(
                dr, atoms.get_cell(), atoms.get_pbc()
            )
            npt.assert_allclose(
                box.diff_dr(dr),
                expected_dr,
                err_msg=f"{atoms.cell.array=}\n{atoms.pbc=}\n{dr=}\n{box.diff_dr(dr)=}",
            )

    def test_diff_pair(self, ase_orth_tests):

        for atoms in ase_orth_tests:
            atoms.center()
            lengths = np.diag(atoms.cell.array)
            box = mp.OrthorhombicBox(lengths, pbc=atoms.get_pbc())
            pos = atoms.positions.copy()
            if len(pos) % 2:
                r1, r2 = np.split(pos[:-1], 2)
            else:
                r1, r2 = np.split(pos, 2)
            npt.assert_allclose(
                box.diff_pair(r1, r2),
                box.diff_dr(r1 - r2),
            )

    def test_make_fractional(self, ase_orth_tests):

        for atoms in ase_orth_tests:
            atoms.center()
            lengths = np.diag(atoms.cell.array)
            box = mp.OrthorhombicBox(lengths, pbc=atoms.get_pbc())
            pos = atoms.positions.copy()
            frac = box.make_fractional(pos)
            npt.assert_allclose(
                frac * atoms.cell.lengths(),
                pos
            )

    def test_make_absolute(self, ase_orth_tests):

        for atoms in ase_orth_tests:
            atoms.center()
            lengths = np.diag(atoms.cell.array)
            box = mp.OrthorhombicBox(lengths, pbc=atoms.get_pbc())
            pos = atoms.positions.copy()
            npt.assert_allclose(
                pos,
                box.make_absolute(box.make_fractional(pos))
            )
