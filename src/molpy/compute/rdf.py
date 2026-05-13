"""Radial distribution function g(r) — molrs-backed.

Returns ``molrs.RDFResult`` directly (no molpy wrapper). The result is
finalized eagerly inside ``molrs.RDF.compute``, so ``result.rdf`` is the
normalized g(r) and not the raw histogram.
"""

from __future__ import annotations

import molrs

from .base import Compute


class RDF(Compute):
    """Histogram pair distances into g(r) over one or more frames.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    r_max : float
        Upper edge of the last bin in Angstroms.
    r_min : float, default 0.0
        Lower edge of bin 0 in Angstroms.

    Notes
    -----
    RDF takes two inputs (frames + neighbor lists) and therefore overrides
    ``__call__`` rather than ``_compute``. It is not directly compatible
    with single-input molexp workflows; chain via an explicit dataflow node
    if needed.
    """

    def __init__(self, n_bins: int, r_max: float, r_min: float = 0.0):
        super().__init__(n_bins=n_bins, r_max=r_max, r_min=r_min)
        self._inner = molrs.RDF(n_bins, r_max, r_min)

    def __call__(self, frames, neighbors) -> molrs.RDFResult:
        molpy_frames = self._as_list(frames)
        molrs_frames = [self._molrs_frame_view(f) for f in molpy_frames]
        return self._inner.compute(molrs_frames, self._as_list(neighbors))

    @staticmethod
    def _molrs_frame_view(frame) -> molrs.Frame:
        """Build a molrs.Frame whose simbox is the molpy frame's box.

        molrs.RDF.compute only reads ``simbox_ref()`` from the frame
        (coordinates come from the NeighborList), so this view does not
        copy any coordinate data — it just attaches the existing Box
        (which IS-A molrs.Box) to a fresh molrs.Frame.
        """
        if frame.box.is_free:
            raise ValueError("frame.box is required for RDF computation")
        mf = molrs.Frame()
        mf.simbox = frame.box
        return mf

    def _compute(self, input):  # pragma: no cover — RDF uses __call__ directly
        raise NotImplementedError(
            "RDF takes two inputs (frames, neighbors); call directly or use "
            "RDF.__call__(frames, neighbors)."
        )

    @staticmethod
    def _as_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]
