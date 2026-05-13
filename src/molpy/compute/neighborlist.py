"""Spatial neighbor list — molrs-backed.

Returns ``molrs.NeighborList`` directly (no molpy wrapper). Coordinates are
stacked once via ``frame["atoms"][["x", "y", "z"]]`` (the only unavoidable
copy, internal to ``Block.__getitem__(list)``); from that point through to
the returned indices/distances the path is zero-copy borrowed views into
the molrs Rust buffers.
"""

from __future__ import annotations

import molrs

from .base import Compute


class NeighborList(Compute):
    """Spatial neighbor-pair query within a cutoff radius.

    Parameters
    ----------
    cutoff : float
        Cutoff radius in Angstroms.

    Examples
    --------
    >>> nlist = NeighborList(cutoff=2.5)(frame)
    >>> nlist.n_pairs
    1834
    """

    def __init__(self, cutoff: float):
        super().__init__(cutoff=float(cutoff))
        self.cutoff = float(cutoff)

    def _compute(self, frame) -> molrs.NeighborList:
        if frame.box.is_free:
            raise ValueError("frame.box is required for spatial neighbor search")
        # Block list-indexing returns (N, 3) via np.column_stack — single
        # unavoidable copy. Box passes through directly (molpy.Box IS-A
        # molrs.Box).
        xyz = frame["atoms"][["x", "y", "z"]]
        return molrs.NeighborQuery(frame.box, xyz, self.cutoff).query_self()
