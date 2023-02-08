# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-08
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt


class TestPairs:

    def test_lj126(self):
        lj126 = mp.potential.LJ126(1.0, 0.0, True)
        natoms = 5
        xyz = np.array([[0, 0, 0], [0.5, 0, 0], [0.7, 0, 0], [0.9, 0, 0], [1.1, 0, 0]])
        pairs = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        types = np.array([0, 0, 1, 2, 3])
        epsilon = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        eps1 = np.repeat(epsilon, natoms).reshape(natoms, natoms)
        eps2 = eps1.T
        eps_mat = np.sqrt(eps1 * eps2)  # geometry
        sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sig1 = np.repeat(sigma, natoms).reshape(natoms, natoms)
        sig2 = sig1.T
        sig_mat = np.sqrt(sig1 * sig2)  # geometry

        eps_mat[0, 2] = eps_mat[2, 0] = 9

        box = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])  # No PBC effect
        tDistance = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no bond
        mScales = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # no bond

        npt.assert_allclose(
            lj126.energy(xyz, pairs, types, eps_mat, sig_mat, box, tDistance, mScales),
            np.array([16128.0, 360.61149, 59.725004, 0.0]),
        )  #

        force_expect = np.array(
                [
                    [-390144.0, -0.0, -0.0],
                    [-6594.0464, -0.0, -0.0],
                    [-1247.9355, -0.0, -0.0],
                    [-0.0, -0.0, -0.0],
                ]
            )

        force_actual = lj126.force(xyz, pairs, types, eps_mat, sig_mat, box, tDistance, mScales)

        npt.assert_allclose(
            force_actual,
            force_expect
        )
        try:
            import jax
            jit_force = jax.jit(lj126.force)
            npt.assert_allclose(jit_force(xyz, pairs, types, eps_mat, sig_mat, box, tDistance, mScales), force_expect)
        except ImportError:
            pass
