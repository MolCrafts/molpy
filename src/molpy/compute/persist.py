"""Pair-survival (persistence) time-correlation functions.

Measures how long pairs of particles remain within a distance cutoff as a
function of time lag — residence-time / hydrogen-bond-dynamics analysis. For a
reference species ``i`` and partner species ``j``::

    C(tau) = < (1/N_i) sum_i sum_j S_ij(t, t+tau) >_t ,

where ``S_ij in {0,1}`` is the survival indicator for the pair born at ``t`` and
observed at ``t+tau``. ``C(0)`` is the mean coordination number.

Three survival criteria (see :class:`molrs.transport.Persist`):

- ``continuous`` — within the survival cutoff at *every* frame since birth.
- ``intermittent`` — within the cutoff at ``t+tau`` (re-formation allowed).
- ``ssp`` — stable-state picture: born within inner cutoff ``r0``, continuously
  within outer cutoff ``r1`` (``r1 >= r0``) since.

The per-pair, per-frame survival accounting runs in Rust
(``molrs.transport.Persist``); this wrapper extracts per-species coordinates and
per-frame orthorhombic box edge lengths.

Adapted from the tame library (https://github.com/Roy-Kid/tame),
``tame/recipes/persist.py`` / ``tame/ops/time.py`` (``tpairsurvive``). The
published ``persist.py`` recipe is non-functional (undefined names); this port
implements the intended correlation with explicit survival criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molrs.transport import Persist as _MolrsPersist

from .base import Compute
from .result import PersistResult

if TYPE_CHECKING:
    from ..core.trajectory import Trajectory


def _parse_tag(tag: str) -> tuple[int, int, str, float, float]:
    """Parse ``"t1,t2:method:r0[,r1]"`` into (t1, t2, method, r0, r1)."""
    parts = tag.split(":")
    if len(parts) != 3:
        raise ValueError(f"persist tag must be 't1,t2:method:r0[,r1]', got {tag!r}")
    t1, t2 = (int(s) for s in parts[0].split(","))
    method = parts[1].strip().lower()
    cutoffs = [float(s) for s in parts[2].split(",")]
    r0 = cutoffs[0]
    r1 = cutoffs[1] if len(cutoffs) > 1 else cutoffs[0]
    return t1, t2, method, r0, r1


class Persist(Compute):
    """Compute pair-survival (persistence) time-correlation functions.

    Args:
        tags: Pair specifications ``"t1,t2:method:r0[,r1]"`` — e.g.
            ``"3,4:ssp:3.0,4.0"`` (cation-anion stable-state pairs born within
            3 A, surviving while within 4 A) or ``"1,1:continuous:3.5"``
            (like-species, single cutoff). ``method`` is one of
            ``continuous`` / ``intermittent`` / ``ssp``.
        max_dt: Maximum time lag in ps.
        dt: Timestep in ps.

    Examples:
        >>> from molpy.io import read_h5_trajectory
        >>> traj = read_h5_trajectory("electrolyte.h5")
        >>> p = Persist(tags=["3,4:ssp:3.0,4.0"], max_dt=30.0, dt=0.01)
        >>> result = p(traj)
        >>> result.correlations["3,4:ssp:3.0,4.0"]  # C(tau), shape (n_cache,)
    """

    def __init__(self, tags: list[str], max_dt: float, dt: float):
        super().__init__(tags=tags, max_dt=max_dt, dt=dt)
        self.tags = tags
        self.max_dt = max_dt
        self.dt = dt
        self.n_cache = int(max_dt / dt)

    def __call__(self, trajectory: "Trajectory") -> PersistResult:
        coords_list: list[NDArray] = []
        box_len_list: list[NDArray] = []
        elems: NDArray | None = None

        for frame in trajectory:
            if "atoms" not in frame:
                raise ValueError("Frame must contain 'atoms' block")
            atoms = frame["atoms"]
            for col in ("x", "y", "z", "type"):
                if col not in atoms:
                    raise ValueError(f"Atoms block must contain '{col}'")
            coords_list.append(np.column_stack([atoms["x"], atoms["y"], atoms["z"]]))
            if elems is None:
                elems = np.asarray(atoms["type"])
            if frame.simbox is None:
                raise ValueError("Frame must contain box information")
            box_len_list.append(np.asarray(frame.simbox.lengths, dtype=np.float64))

        coords_traj = np.asarray(coords_list, dtype=np.float64)  # (F, N, 3)
        box_lengths = np.ascontiguousarray(
            np.asarray(box_len_list, dtype=np.float64)
        )  # (F, 3)
        n_frames = coords_traj.shape[0]
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")
        assert elems is not None

        max_lag = self.n_cache - 1
        correlations: dict[str, NDArray] = {}
        for tag in self.tags:
            t1, t2, method, r0, r1 = _parse_tag(tag)
            ci = np.ascontiguousarray(coords_traj[:, elems == t1, :])
            cj = np.ascontiguousarray(coords_traj[:, elems == t2, :])
            res = _MolrsPersist.pair_survival_tcf(
                ci,
                cj,
                box_lengths,
                r0,
                r1,
                method,
                self.dt,
                max_lag,
                t1 == t2,
            )
            correlations[tag] = res["correlation"]

        time_array = np.arange(self.n_cache, dtype=np.float64) * self.dt
        return PersistResult(time=time_array, correlations=correlations)
