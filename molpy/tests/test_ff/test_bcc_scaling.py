"""BccModel on a long conjugated chain: equivalencing must not walk every path.

``-eq 1`` compares, per atom, the multiset of scores of every simple path out of
it. Thirty benzene rings joined para give each atom about 2^30 such paths, which
the original path-by-path walk could never finish; the histogram engine counts
them over the block-cut tree instead. The time bound is loose (CI machines,
debug-ish builds); what it guards against is the exponential blow-up.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import molpy as mp
import molpy.perceive

RINGS = 30


def _polyphenylene(rings: int) -> mp.Atomistic:
    """``rings`` benzene rings joined para; atom ``6k + j`` is carbon ``j`` of
    ring ``k`` (``j = 0`` bonds the previous ring, ``j = 3`` the next)."""
    smiles = "c1ccc(cc1)" + "-c1ccc(cc1)" * (rings - 1)
    return molpy.perceive.add_hydrogens(mp.io.smiles.read_smiles(smiles).to_atomistic())


def test_a_long_ring_chain_is_equivalenced_quickly_and_symmetrically():
    mol = _polyphenylene(RINGS)
    base = np.random.default_rng(7).normal(0.0, 0.2, mol.n_atoms)

    started = time.perf_counter()
    q = mp.ff.charge.BccModel(parameter_set="bcc").assign(mol, base)
    elapsed = time.perf_counter() - started

    assert elapsed < 30.0, f"BccModel.assign took {elapsed:.1f} s"
    assert q.sum() == pytest.approx(base.sum(), abs=1e-9)
    carbon = lambda ring, j: q[6 * ring + j]  # noqa: E731
    for ring in range(RINGS):
        # Mirror symmetry of the chain, and the two ortho / two meta carbons.
        for j in range(6):
            mirror = carbon(RINGS - 1 - ring, (3 - j) % 6)
            assert carbon(ring, j) == pytest.approx(mirror, abs=1e-12)
        assert carbon(ring, 1) == pytest.approx(carbon(ring, 5), abs=1e-12)
        assert carbon(ring, 2) == pytest.approx(carbon(ring, 4), abs=1e-12)
    # Depth along the chain is not symmetric: an end ring is not an inner ring.
    assert carbon(0, 0) != pytest.approx(carbon(1, 0), abs=1e-6)
